# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download and convert a PandaSet scene to npz for av_trainer.py.

Produces an npz file compatible with av_trainer.py containing camera images,
intrinsics, extrinsics, LiDAR points, and LiDAR sensor poses. The PandaSet
path in av_trainer uses the LiDAR points for sparse depth supervision; native
LiDAR rasterization is currently NCore-only.

Two input modes:
  1. From a local PandaSet directory (--pandaset-dir)
  2. Download a single scene from HuggingFace (--download, requires HF_TOKEN)

Default output is full-resolution (1920x1080).  Use --resolution WxH to
downsample (no upsampling allowed).

Examples:
    # Full-resolution scene 019 from local PandaSet directory
    python prepare_pandaset.py --pandaset-dir /data/pandaset --scene 019

    # Download scene 019 from HuggingFace and convert at full resolution
    python prepare_pandaset.py --download --scene 019

    # Reproduce the CI test asset (240x135, 8x downsampled)
    python prepare_pandaset.py --pandaset-dir /data/pandaset --scene 019 \\
        --resolution 240x135 --output assets/test_pandaset.npz

Usage with av_trainer.py:
    python av_trainer.py --scene pandaset_019.npz --max-steps 30000
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import pickle
import struct
import tempfile
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

# PandaSet camera names in canonical order
CAMERA_NAMES = [
    "front_camera",
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_camera",
]

# HuggingFace PandaSet URLs
# Primary: original dataset (gated, requires HF token)
HF_DATASET_URL_AUTH = (
    "https://huggingface.co/datasets/TommyZihwormo/PandaSet/resolve/main/pandaset_0.zip"
)
# Fallback: public mirror (no auth required)
HF_DATASET_URL_PUBLIC = (
    "https://huggingface.co/datasets/georghess/pandaset/resolve/main/pandaset.zip"
)


# ---------------------------------------------------------------------------
# HuggingFace zip download via HTTP range requests
# ---------------------------------------------------------------------------


def _http_read_range(
    url: str, start: int, length: int, token: str | None = None
) -> bytes:
    """Read a byte range from a URL via HTTP Range request.

    Verifies the server actually honored the ``Range`` header by checking
    for a ``206 Partial Content`` response. If a proxy strips the header
    and the origin returns ``200 OK`` with the full body instead, fail
    loudly rather than silently slurping the entire (potentially
    multi-gigabyte) zip into memory.
    """
    req = Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Range", f"bytes={start}-{start + length - 1}")
    with urlopen(req, timeout=60) as resp:
        if resp.status != 206:
            raise RuntimeError(
                f"Expected 206 Partial Content for Range request to {url}, "
                f"got {resp.status}. A proxy may be stripping the Range header; "
                "refusing to download the full body."
            )
        return resp.read()


def _http_get_size(url: str, token: str | None = None) -> int:
    """Get the total size of a remote file."""
    req = Request(url, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Expected 200 OK for HEAD {url}, got {resp.status}.")
        return int(resp.headers["Content-Length"])


def _read_zip_central_directory(
    url: str, file_size: int, token: str | None = None
) -> list[tuple[str, int, int]]:
    """Read the zip central directory to get file offsets without downloading
    the whole zip.  Returns list of (filename, local_header_offset, compressed_size)."""
    # Read the End of Central Directory record (last 64KB should be enough)
    eocd_size = min(65536, file_size)
    eocd_data = _http_read_range(url, file_size - eocd_size, eocd_size, token)

    # Try Zip64 EOCD locator first (0x07064b50), then standard EOCD (0x06054b50)
    zip64_locator_sig = b"\x50\x4b\x06\x07"
    zip64_eocd_sig = b"\x50\x4b\x06\x06"
    eocd_sig = b"\x50\x4b\x05\x06"

    locator_pos = eocd_data.rfind(zip64_locator_sig)
    if locator_pos != -1:
        # Zip64: read the Zip64 EOCD locator to find the Zip64 EOCD
        zip64_eocd_offset = struct.unpack_from("<Q", eocd_data, locator_pos + 8)[0]
        zip64_eocd_data = _http_read_range(url, zip64_eocd_offset, 56, token)
        if zip64_eocd_data[:4] != zip64_eocd_sig:
            raise ValueError("Invalid Zip64 EOCD record")
        cd_size = struct.unpack_from("<Q", zip64_eocd_data, 40)[0]
        cd_offset = struct.unpack_from("<Q", zip64_eocd_data, 48)[0]
    else:
        eocd_pos = eocd_data.rfind(eocd_sig)
        if eocd_pos == -1:
            raise ValueError("Could not find End of Central Directory record")
        eocd = eocd_data[eocd_pos:]
        cd_size = struct.unpack_from("<I", eocd, 12)[0]
        cd_offset = struct.unpack_from("<I", eocd, 16)[0]

    # Read the entire Central Directory
    cd_data = _http_read_range(url, cd_offset, cd_size, token)

    entries = []
    pos = 0
    cd_sig = b"\x50\x4b\x01\x02"
    while pos < len(cd_data):
        if cd_data[pos : pos + 4] != cd_sig:
            break
        compressed_size = struct.unpack_from("<I", cd_data, pos + 20)[0]
        fname_len = struct.unpack_from("<H", cd_data, pos + 28)[0]
        extra_len = struct.unpack_from("<H", cd_data, pos + 30)[0]
        comment_len = struct.unpack_from("<H", cd_data, pos + 32)[0]
        local_header_offset = struct.unpack_from("<I", cd_data, pos + 42)[0]
        fname = cd_data[pos + 46 : pos + 46 + fname_len].decode("utf-8")

        # Parse Zip64 extra field if sizes/offset are 0xFFFFFFFF
        extra_start = pos + 46 + fname_len
        extra_data = cd_data[extra_start : extra_start + extra_len]
        if compressed_size == 0xFFFFFFFF or local_header_offset == 0xFFFFFFFF:
            # Find Zip64 extra field (tag 0x0001)
            epos = 0
            while epos < len(extra_data) - 4:
                tag = struct.unpack_from("<H", extra_data, epos)[0]
                sz = struct.unpack_from("<H", extra_data, epos + 2)[0]
                if tag == 0x0001:
                    z64 = extra_data[epos + 4 : epos + 4 + sz]
                    z64_off = 0
                    # Fields appear in order: uncompressed, compressed, offset
                    uncomp_size_orig = struct.unpack_from("<I", cd_data, pos + 24)[0]
                    if uncomp_size_orig == 0xFFFFFFFF:
                        z64_off += 8  # skip uncompressed size
                    if compressed_size == 0xFFFFFFFF:
                        compressed_size = struct.unpack_from("<Q", z64, z64_off)[0]
                        z64_off += 8
                    if local_header_offset == 0xFFFFFFFF:
                        local_header_offset = struct.unpack_from("<Q", z64, z64_off)[0]
                    break
                epos += 4 + sz

        entries.append((fname, local_header_offset, compressed_size))
        pos += 46 + fname_len + extra_len + comment_len

    return entries


def _extract_file_from_zip(
    url: str,
    local_header_offset: int,
    compressed_size: int,
    token: str | None = None,
) -> bytes:
    """Extract a single file from a remote zip via range request."""
    # Read local file header (30 bytes) + variable fields
    header_data = _http_read_range(url, local_header_offset, 30, token)
    fname_len = struct.unpack_from("<H", header_data, 26)[0]
    extra_len = struct.unpack_from("<H", header_data, 28)[0]
    compression = struct.unpack_from("<H", header_data, 8)[0]

    data_offset = local_header_offset + 30 + fname_len + extra_len
    file_data = _http_read_range(url, data_offset, compressed_size, token)

    if compression == 0:  # stored
        return file_data
    elif compression == 8:  # deflated
        import zlib

        return zlib.decompress(file_data, -15)
    else:
        raise ValueError(f"Unsupported compression method: {compression}")


def download_scene_from_hf(
    scene_id: str, output_dir: str, token: str | None = None
) -> str:
    """Download a single PandaSet scene from HuggingFace to a local directory.
    Uses HTTP range requests to extract only the requested scene from the
    44.5 GB zip without downloading the full archive.
    If token is provided, uses the original gated dataset; otherwise falls
    back to the public mirror.
    Returns the path to the scene directory."""
    if token:
        url = HF_DATASET_URL_AUTH
        print("Connecting to HuggingFace dataset (authenticated)...")
    else:
        url = HF_DATASET_URL_PUBLIC
        print("Connecting to HuggingFace dataset (public mirror)...")
    file_size = _http_get_size(url, token)
    print(f"Zip size: {file_size / 1e9:.1f} GB")

    print("Reading zip central directory...")
    entries = _read_zip_central_directory(url, file_size, token)

    # Filter to our scene — handle optional top-level directory prefix
    # (e.g., "pandaset/019/..." or "019/...")
    scene_entries = []
    scene_prefix = None
    for candidate_prefix in [f"{scene_id}/", f"pandaset/{scene_id}/"]:
        scene_entries = [
            (name, offset, size)
            for name, offset, size in entries
            if name.startswith(candidate_prefix) and not name.endswith("/")
        ]
        if scene_entries:
            scene_prefix = candidate_prefix
            break
    print(f"Found {len(scene_entries)} files for scene {scene_id}")

    if not scene_entries:
        # Show available scenes for debugging
        top_dirs = sorted(
            set(
                e[0].split("/")[1]
                if e[0].startswith("pandaset/")
                else e[0].split("/")[0]
                for e in entries
                if "/" in e[0]
            )
        )
        raise ValueError(f"Scene '{scene_id}' not found. Available: {top_dirs[:10]}...")

    scene_dir = os.path.join(output_dir, scene_id)
    os.makedirs(scene_dir, exist_ok=True)

    assert scene_prefix is not None
    for i, (name, offset, comp_size) in enumerate(scene_entries):
        rel_path = name[len(scene_prefix) :]
        out_path = os.path.join(scene_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if i % 20 == 0 or i == len(scene_entries) - 1:
            print(f"  Downloading {i + 1}/{len(scene_entries)}: {rel_path}")

        data = _extract_file_from_zip(url, offset, comp_size, token)
        with open(out_path, "wb") as f:
            f.write(data)

    print(f"Scene {scene_id} downloaded to {scene_dir}")
    return scene_dir


# ---------------------------------------------------------------------------
# PandaSet scene loading from local directory
# ---------------------------------------------------------------------------


def load_camera_images(scene_dir: str, camera: str, frame_ids: list[int]) -> np.ndarray:
    """Load camera images for given frames. Returns [N, H, W, 3] uint8."""
    images = []
    cam_dir = os.path.join(scene_dir, "camera", camera)
    for fid in frame_ids:
        path = os.path.join(cam_dir, f"{fid:02d}.jpg")
        if not os.path.exists(path):
            path = os.path.join(cam_dir, f"{fid}.jpg")
        img = np.array(Image.open(path))
        images.append(img)
    return np.stack(images)


def load_camera_poses(scene_dir: str, camera: str) -> list[dict]:
    """Load camera poses.json. Returns list of 4x4 pose dicts."""
    path = os.path.join(scene_dir, "camera", camera, "poses.json")
    with open(path) as f:
        return json.load(f)


def load_camera_intrinsics(scene_dir: str, camera: str) -> dict:
    """Load camera intrinsics.json."""
    path = os.path.join(scene_dir, "camera", camera, "intrinsics.json")
    with open(path) as f:
        return json.load(f)


def load_lidar_frame(scene_dir: str, frame_id: int) -> np.ndarray:
    """Load a single LiDAR frame from .pkl.gz. Returns [M, 4] (x, y, z, i)."""
    path = os.path.join(scene_dir, "lidar", f"{frame_id:02d}.pkl.gz")
    if not os.path.exists(path):
        path = os.path.join(scene_dir, "lidar", f"{frame_id}.pkl.gz")
    with gzip.open(path, "rb") as f:
        df = pickle.load(f)
    points = np.column_stack(
        [df["x"].values, df["y"].values, df["z"].values, df["i"].values]
    ).astype(np.float32)
    return points


def load_lidar_poses(scene_dir: str) -> list[dict]:
    """Load LiDAR poses.json. Returns list of 4x4 pose dicts."""
    path = os.path.join(scene_dir, "lidar", "poses.json")
    with open(path) as f:
        return json.load(f)


def pose_dict_to_matrix(pose: dict) -> np.ndarray:
    """Convert PandaSet pose dict {"position": {x,y,z}, "heading": {w,x,y,z}}
    to a 4x4 homogeneous transformation matrix."""
    from scipy.spatial.transform import Rotation

    pos = np.array(
        [pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]]
    )
    quat = [
        pose["heading"]["x"],
        pose["heading"]["y"],
        pose["heading"]["z"],
        pose["heading"]["w"],
    ]
    R = Rotation.from_quat(quat).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def get_frame_ids(scene_dir: str) -> list[int]:
    """Discover available frame IDs from the LiDAR directory."""
    lidar_dir = os.path.join(scene_dir, "lidar")
    ids = []
    for f in os.listdir(lidar_dir):
        if f.endswith(".pkl.gz"):
            ids.append(int(f.replace(".pkl.gz", "")))
    return sorted(ids)


def downsample_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Downsample an image using PIL. Input/output are uint8 HWC arrays."""
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil_img)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_scene(
    scene_dir: str,
    output_path: str,
    resolution: tuple[int, int] | None,
    include_frame_ids: list[int] | None,
    skip_frame_ids: list[int] | None,
    test_frame_ids: list[int] | None,
    n_test_frames: int,
    lidar_subsample: int,
    lidar_every_n_frames: int,
    lidar_sensor_json: str | None,
) -> None:
    """Convert a PandaSet scene directory to npz."""
    all_frame_ids = get_frame_ids(scene_dir)
    print(
        f"Scene has {len(all_frame_ids)} frames: {all_frame_ids[0]}..{all_frame_ids[-1]}"
    )

    # Apply frame selection
    if include_frame_ids is not None:
        frame_ids = [f for f in include_frame_ids if f in all_frame_ids]
        missing = set(include_frame_ids) - set(all_frame_ids)
        if missing:
            print(
                f"  Warning: requested frame IDs not found in scene: {sorted(missing)}"
            )
    else:
        frame_ids = list(all_frame_ids)

    if skip_frame_ids is not None:
        frame_ids = [f for f in frame_ids if f not in skip_frame_ids]

    n_frames = len(frame_ids)
    print(f"Selected {n_frames} frames: {frame_ids[:5]}...{frame_ids[-5:]}")

    # Determine which cameras are available
    cameras = []
    for cam in CAMERA_NAMES:
        if os.path.isdir(os.path.join(scene_dir, "camera", cam)):
            cameras.append(cam)
    print(f"Cameras ({len(cameras)}): {cameras}")

    # Load one image to get original resolution
    sample_img = load_camera_images(scene_dir, cameras[0], [frame_ids[0]])[0]
    orig_h, orig_w = sample_img.shape[:2]
    print(f"Original resolution: {orig_w}x{orig_h}")

    # Determine target resolution
    if resolution is not None:
        target_w, target_h = resolution
        if target_w > orig_w or target_h > orig_h:
            raise ValueError(
                f"Requested resolution {target_w}x{target_h} exceeds original "
                f"{orig_w}x{orig_h}. Only downsampling is supported."
            )
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        downsample_factor = int(round(scale_x))
        if abs(scale_x - scale_y) > 0.01:
            print(
                f"  Warning: non-uniform scaling detected "
                f"(scale_x={scale_x:.3f}, scale_y={scale_y:.3f})"
            )
        print(
            f"Target resolution: {target_w}x{target_h} ({downsample_factor}x downsample)"
        )
    else:
        target_w, target_h = orig_w, orig_h
        downsample_factor = 1
        print(f"Using original resolution: {target_w}x{target_h}")

    # Load all camera images
    print("Loading camera images...")
    all_images = np.zeros(
        (n_frames, len(cameras), target_h, target_w, 3), dtype=np.uint8
    )
    for ci, cam in enumerate(cameras):
        imgs = load_camera_images(scene_dir, cam, frame_ids)
        for fi in range(n_frames):
            if resolution is not None:
                all_images[fi, ci] = downsample_image(imgs[fi], target_h, target_w)
            else:
                all_images[fi, ci] = imgs[fi]
        print(f"  {cam}: {imgs.shape[0]} frames loaded")

    # Load camera intrinsics and scale for resolution
    print("Loading camera intrinsics...")
    cam_intrinsics = np.zeros((len(cameras), 4), dtype=np.float32)
    for ci, cam in enumerate(cameras):
        intr = load_camera_intrinsics(scene_dir, cam)
        fx = intr["fx"]
        fy = intr["fy"]
        cx = intr["cx"]
        cy = intr["cy"]
        # Scale intrinsics for downsampled resolution
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        cam_intrinsics[ci] = [fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y]

    # Load camera extrinsics (cam_to_worlds)
    print("Loading camera poses...")
    cam_to_worlds = np.zeros((n_frames, len(cameras), 4, 4), dtype=np.float32)
    for ci, cam in enumerate(cameras):
        poses = load_camera_poses(scene_dir, cam)
        for fi, fid in enumerate(frame_ids):
            cam_to_worlds[fi, ci] = pose_dict_to_matrix(poses[fid])

    # Load LiDAR points
    print("Loading LiDAR points...")
    if lidar_every_n_frames > 1 or lidar_subsample > 1:
        print(
            f"  LiDAR subsampling: every {lidar_every_n_frames} frames, "
            f"every {lidar_subsample} points"
        )
    all_points = []
    all_frame_indices = []
    for fi, fid in enumerate(frame_ids):
        if fi % lidar_every_n_frames != 0:
            continue
        pts = load_lidar_frame(scene_dir, fid)
        if lidar_subsample > 1:
            pts = pts[::lidar_subsample]
        all_points.append(pts)
        all_frame_indices.append(np.full(len(pts), fi, dtype=np.int32))
        if fi % 10 == 0:
            print(f"  Frame {fid}: {len(pts)} points")
    lidar_points = np.concatenate(all_points, axis=0)
    lidar_frame_indices = np.concatenate(all_frame_indices, axis=0)
    print(f"Total LiDAR points: {len(lidar_points)}")

    # Load LiDAR poses
    print("Loading LiDAR poses...")
    lidar_poses = load_lidar_poses(scene_dir)
    lidar_to_worlds = np.zeros((n_frames, 4, 4), dtype=np.float32)
    for fi, fid in enumerate(frame_ids):
        lidar_to_worlds[fi] = pose_dict_to_matrix(lidar_poses[fid])

    # Determine test frames
    if test_frame_ids is not None:
        is_test = np.array([fid in test_frame_ids for fid in frame_ids])
    else:
        # Evenly space test frames
        test_indices = np.linspace(0, n_frames - 1, n_test_frames + 2, dtype=int)[1:-1]
        is_test = np.zeros(n_frames, dtype=bool)
        is_test[test_indices] = True
    print(f"Train/test split: {(~is_test).sum()} train, {is_test.sum()} test")

    # Build npz dict
    npz_data = {
        "images": all_images,
        "cam_intrinsics": cam_intrinsics,
        "cam_to_worlds": cam_to_worlds,
        "lidar_points": lidar_points,
        "lidar_frame_indices": lidar_frame_indices,
        "lidar_to_worlds": lidar_to_worlds,
        "is_test": is_test,
        "camera_names": np.array(cameras),
        "frame_ids": np.array(frame_ids, dtype=np.int32),
        "downsample": np.int32(downsample_factor),
    }

    # --lidar-sensor-json is reserved for future PandaSet LiDAR rendering;
    # av_trainer's PandaSet path currently consumes only `lidar_points` /
    # `lidar_frame_indices` (sparse depth supervision) and never reads any of
    # the per-sensor keys we used to write here, so writing them would just
    # bloat the npz with dead data and mislead readers of the call site into
    # thinking PandaSet LiDAR rasterization is wired up.
    if lidar_sensor_json:
        print(
            "note: --lidar-sensor-json is reserved for future PandaSet LiDAR "
            "rendering and is currently not consumed by av_trainer"
        )

    # Save
    print(f"Saving to {output_path}...")
    np.savez_compressed(output_path, **npz_data)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done: {output_path} ({size_mb:.1f} MB)")
    print(f"  Images: {all_images.shape}")
    print(f"  LiDAR: {lidar_points.shape[0]} points")
    print(f"  Resolution: {target_w}x{target_h} (downsample={downsample_factor}x)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PandaSet scene to npz for av_trainer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Full-resolution scene 019 from local PandaSet
  python prepare_pandaset.py --pandaset-dir /data/pandaset --scene 019

  # Download from HuggingFace (requires HF_TOKEN env var)
  HF_TOKEN=hf_... python prepare_pandaset.py --download --scene 019

  # Reproduce the CI test asset (64 frames at 240x135, subsampled LiDAR)
  python prepare_pandaset.py --pandaset-dir /data/pandaset --scene 019 \\
      --resolution 240x135 \\
      --skip-frame-ids 63 64 65 66 67 68 69 71 72 73 74 75 76 77 78 79 \\
      --test-frame-ids 10 30 50 70 \\
      --lidar-every-n-frames 2 --lidar-subsample 26 \\
      --output assets/test_pandaset.npz
""",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pandaset-dir",
        type=str,
        help="path to extracted PandaSet root directory (contains scene dirs)",
    )
    source.add_argument(
        "--download",
        action="store_true",
        help="download scene from HuggingFace via HTTP range requests (no auth needed)",
    )

    parser.add_argument(
        "--scene",
        type=str,
        default="019",
        help="PandaSet scene ID (default: 019)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="target WxH resolution for camera images; must be <= original "
        "(default: original resolution, 1920x1080 for PandaSet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output npz path (default: pandaset_{scene}.npz)",
    )
    parser.add_argument(
        "--frame-ids",
        type=int,
        nargs="+",
        default=None,
        help="explicit list of frame IDs to include (default: all frames in scene)",
    )
    parser.add_argument(
        "--skip-frame-ids",
        type=int,
        nargs="+",
        default=None,
        help="frame IDs to exclude (applied after --frame-ids if both given)",
    )
    parser.add_argument(
        "--n-test-frames",
        type=int,
        default=4,
        help="number of evenly-spaced held-out test frames (default: 4)",
    )
    parser.add_argument(
        "--test-frame-ids",
        type=int,
        nargs="+",
        default=None,
        help="explicit test frame IDs (overrides --n-test-frames)",
    )
    parser.add_argument(
        "--lidar-subsample",
        type=int,
        default=1,
        help="spatial subsample factor for LiDAR points: keep every Nth point "
        "(default: 1 = all points). The CI test asset uses 26.",
    )
    parser.add_argument(
        "--lidar-every-n-frames",
        type=int,
        default=1,
        help="keep LiDAR only for every Nth frame (default: 1 = all frames). "
        "The CI test asset uses 2 (every other frame).",
    )
    parser.add_argument(
        "--lidar-sensor-json",
        type=str,
        default=None,
        help="reserved for future PandaSet LiDAR rendering; currently a no-op "
        "(av_trainer's PandaSet path uses sparse-LiDAR depth supervision only).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for the original gated dataset. "
        "If not provided, falls back to a public mirror. "
        "Can also be set via HF_TOKEN env var.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="cache directory for HuggingFace downloads (default: temp dir)",
    )

    args = parser.parse_args()

    # Parse resolution
    resolution = None
    if args.resolution:
        parts = args.resolution.lower().split("x")
        if len(parts) != 2:
            parser.error("--resolution must be WxH (e.g., 480x270)")
        resolution = (int(parts[0]), int(parts[1]))

    # Default output path
    output = args.output or f"pandaset_{args.scene}.npz"

    # Get scene directory
    auto_cache = False
    if args.download:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if args.cache_dir:
            cache_dir = args.cache_dir
        else:
            cache_dir = tempfile.mkdtemp(prefix="pandaset_")
            auto_cache = True
            print(f"Using auto-created cache dir: {cache_dir}")
        scene_dir = download_scene_from_hf(args.scene, cache_dir, token)
    else:
        scene_dir = os.path.join(args.pandaset_dir, args.scene)
        if not os.path.isdir(scene_dir):
            parser.error(f"Scene directory not found: {scene_dir}")

    convert_scene(
        scene_dir=scene_dir,
        output_path=output,
        resolution=resolution,
        include_frame_ids=args.frame_ids,
        skip_frame_ids=args.skip_frame_ids,
        test_frame_ids=args.test_frame_ids,
        n_test_frames=args.n_test_frames,
        lidar_subsample=args.lidar_subsample,
        lidar_every_n_frames=args.lidar_every_n_frames,
        lidar_sensor_json=args.lidar_sensor_json,
    )

    # Clean up auto-created cache after successful conversion
    if auto_cache:
        import shutil

        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Cleaned up cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
