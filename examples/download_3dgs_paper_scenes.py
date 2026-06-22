# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download pretrained 3D Gaussian Splatting PLY scenes from INRIA.

Downloads the paper-comparison scenes (Mip-NeRF 360, Tanks & Temples, Deep
Blending) from the public 3D Gaussian Splatting pretrained models release at
INRIA. Uses HTTP range requests to extract only the requested scenes without
downloading the full 14.66 GB archive.

The downloaded ``point_cloud.ply`` files are compatible with
:func:`gsplat.exporter.load_ply_to_splats` and the
``examples/benchmarks/render_only_bench.py --ply-path`` option.

Usage:
    # Download a single Mip-NeRF 360 scene (~700 MB)
    python download_3dgs_paper_scenes.py --scenes bicycle --output-dir ./3dgs_models

    # Download all Mip-NeRF 360 outdoor scenes
    python download_3dgs_paper_scenes.py \\
        --scenes bicycle flowers garden stump treehill \\
        --output-dir ./3dgs_models

    # List available scenes
    python download_3dgs_paper_scenes.py --list

The range-extract helpers mirror those in ``prepare_pandaset.py``.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from urllib.request import Request, urlopen

MODELS_ZIP_URL = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting"
    "/datasets/pretrained/models.zip"
)

# Scene names match the FastGS / 3DGS paper convention. Per-scene Gaussian
# counts at the standard iteration_30000 checkpoint are reported only as a
# rough order-of-magnitude reference; actual counts vary slightly between
# trainer reruns.
MIPNERF360_OUTDOOR = ["bicycle", "flowers", "garden", "stump", "treehill"]
MIPNERF360_INDOOR = ["room", "counter", "kitchen", "bonsai"]
TANKS_AND_TEMPLES = ["truck", "train"]
DEEP_BLENDING = ["drjohnson", "playroom"]

ALL_SCENES = MIPNERF360_OUTDOOR + MIPNERF360_INDOOR + TANKS_AND_TEMPLES + DEEP_BLENDING


# ---------------------------------------------------------------------------
# HTTP range / zip central directory parsing
# ---------------------------------------------------------------------------


def _http_read_range(url: str, start: int, length: int) -> bytes:
    """Read a byte range from a URL via HTTP Range request."""
    req = Request(url)
    req.add_header("Range", f"bytes={start}-{start + length - 1}")
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def _http_get_size(url: str) -> int:
    """Get the total size of a remote file."""
    req = Request(url, method="HEAD")
    with urlopen(req, timeout=30) as resp:
        return int(resp.headers["Content-Length"])


def _read_zip_central_directory(url: str, file_size: int) -> list[tuple[str, int, int]]:
    """Return ``[(filename, local_header_offset, compressed_size), ...]``.

    Supports both standard and Zip64 EOCD records (Inria's models.zip is
    >4 GB so it uses Zip64).
    """
    eocd_size = min(65536, file_size)
    eocd_data = _http_read_range(url, file_size - eocd_size, eocd_size)

    zip64_locator_sig = b"\x50\x4b\x06\x07"
    zip64_eocd_sig = b"\x50\x4b\x06\x06"
    eocd_sig = b"\x50\x4b\x05\x06"

    locator_pos = eocd_data.rfind(zip64_locator_sig)
    if locator_pos != -1:
        zip64_eocd_offset = struct.unpack_from("<Q", eocd_data, locator_pos + 8)[0]
        zip64_eocd_data = _http_read_range(url, zip64_eocd_offset, 56)
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

    cd_data = _http_read_range(url, cd_offset, cd_size)

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

        # Parse Zip64 extra field if any sizes are sentinel 0xFFFFFFFF
        extra_start = pos + 46 + fname_len
        extra_data = cd_data[extra_start : extra_start + extra_len]
        if compressed_size == 0xFFFFFFFF or local_header_offset == 0xFFFFFFFF:
            epos = 0
            while epos < len(extra_data) - 4:
                tag = struct.unpack_from("<H", extra_data, epos)[0]
                sz = struct.unpack_from("<H", extra_data, epos + 2)[0]
                if tag == 0x0001:
                    z64 = extra_data[epos + 4 : epos + 4 + sz]
                    z64_off = 0
                    uncomp_size_orig = struct.unpack_from("<I", cd_data, pos + 24)[0]
                    if uncomp_size_orig == 0xFFFFFFFF:
                        z64_off += 8
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
) -> bytes:
    """Extract a single file from a remote zip via range request."""
    header_data = _http_read_range(url, local_header_offset, 30)
    fname_len = struct.unpack_from("<H", header_data, 26)[0]
    extra_len = struct.unpack_from("<H", header_data, 28)[0]
    compression = struct.unpack_from("<H", header_data, 8)[0]

    data_offset = local_header_offset + 30 + fname_len + extra_len
    file_data = _http_read_range(url, data_offset, compressed_size)

    if compression == 0:
        return file_data
    if compression == 8:
        import zlib

        return zlib.decompress(file_data, -15)
    raise ValueError(f"Unsupported compression method: {compression}")


# ---------------------------------------------------------------------------
# Scene download
# ---------------------------------------------------------------------------


def _scene_ply_filter(scene: str, fname: str) -> bool:
    """Return True if ``fname`` is the ``point_cloud.ply`` for ``scene``.

    The Inria release stores trained scenes as
    ``<dataset>/<scene>/point_cloud/iteration_30000/point_cloud.ply``. We
    match by scene segment + suffix to be robust to small layout shifts
    (e.g. trailing slashes, dataset-folder prefix differences).
    """
    parts = fname.strip("/").split("/")
    return (
        scene in parts and parts[-1] == "point_cloud.ply" and "iteration_30000" in parts
    )


def download_scene(
    scene: str,
    output_dir: str,
    entries: list[tuple[str, int, int]] | None = None,
) -> str:
    """Download the trained ``point_cloud.ply`` for one scene.

    Args:
        scene: Scene name (e.g. ``"bicycle"``). Must be in :data:`ALL_SCENES`.
        output_dir: Directory to write the PLY into. The file is stored as
            ``<output_dir>/<scene>/point_cloud.ply``.
        entries: Optional pre-computed central-directory listing of
            ``models.zip`` (saves one HEAD + range read when downloading
            multiple scenes in a loop).

    Returns:
        Absolute path to the downloaded ``point_cloud.ply``.
    """
    if scene not in ALL_SCENES:
        raise ValueError(f"Unknown scene {scene!r}; available scenes: {ALL_SCENES}")

    if entries is None:
        size = _http_get_size(MODELS_ZIP_URL)
        entries = _read_zip_central_directory(MODELS_ZIP_URL, size)

    matches = [e for e in entries if _scene_ply_filter(scene, e[0])]
    if not matches:
        raise RuntimeError(
            f"Could not locate point_cloud.ply for scene {scene!r} in "
            f"{MODELS_ZIP_URL}. The Inria release layout may have changed."
        )
    if len(matches) > 1:
        # Prefer the longest match (most specific path); unlikely but defensive.
        matches.sort(key=lambda e: len(e[0]), reverse=True)

    fname, offset, compressed_size = matches[0]
    scene_dir = os.path.join(output_dir, scene)
    os.makedirs(scene_dir, exist_ok=True)
    dest = os.path.join(scene_dir, "point_cloud.ply")

    print(
        f"  {scene}: extracting {fname} "
        f"({compressed_size / 1e6:.1f} MB compressed) ...",
        flush=True,
    )
    data = _extract_file_from_zip(MODELS_ZIP_URL, offset, compressed_size)
    with open(dest, "wb") as f:
        f.write(data)
    print(f"    -> {dest} ({len(data) / 1e6:.1f} MB uncompressed)", flush=True)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1] if "Usage:" in __doc__ else "",
    )
    p.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=(
            f"Scene names to download. Available: {' '.join(ALL_SCENES)}. "
            "Default: all scenes."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./3dgs_models",
        help="Directory to write per-scene PLYs into.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available scenes and exit.",
    )
    args = p.parse_args()

    if args.list:
        print("Available paper-comparison scenes:")
        print(f"  Mip-NeRF 360 outdoor: {' '.join(MIPNERF360_OUTDOOR)}")
        print(f"  Mip-NeRF 360 indoor : {' '.join(MIPNERF360_INDOOR)}")
        print(f"  Tanks & Temples     : {' '.join(TANKS_AND_TEMPLES)}")
        print(f"  Deep Blending       : {' '.join(DEEP_BLENDING)}")
        return

    scenes = args.scenes if args.scenes is not None else ALL_SCENES
    unknown = [s for s in scenes if s not in ALL_SCENES]
    if unknown:
        print(
            f"Error: unknown scene(s) {unknown}. " f"Available: {' '.join(ALL_SCENES)}",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Reading central directory of {MODELS_ZIP_URL} ...")
    size = _http_get_size(MODELS_ZIP_URL)
    entries = _read_zip_central_directory(MODELS_ZIP_URL, size)
    print(f"  zip size: {size / 1e9:.2f} GB, {len(entries)} entries")
    print()

    for scene in scenes:
        download_scene(scene, args.output_dir, entries=entries)

    print()
    print(f"Done. Scenes written under {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
