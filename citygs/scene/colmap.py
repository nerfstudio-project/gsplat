"""Ingest a COLMAP reconstruction into a scene manifest.

Reads ``<data_dir>/sparse[/0]`` with pycolmap, normalizes the world space,
and writes ``scene.json`` + ``points.npz`` (SfM points, colors, and the
per-camera point-visibility index used for sparse-depth supervision).

Only undistorted pinhole cameras are supported. For distorted captures run
``colmap image_undistorter`` first — for large scenes with known poses,
``colmap point_triangulator`` on the undistorted model is also much faster
than a from-scratch SfM.

Run: ``python -m citygs.scene.colmap --data-dir ... --result-dir ...``
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .manifest import CameraMeta, SceneManifest
from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


@dataclass
class IngestConfig:
    data_dir: str = "data/scene"
    # scene.json and points.npz are written here.
    result_dir: str = "results/scene"
    factor: int = 1
    test_every: int = 8
    normalize: bool = True
    # Generate <data_dir>/images_<factor> if it does not exist yet.
    write_downsampled: bool = True
    downsample_workers: int = 8


def _rel_paths(root: str) -> List[str]:
    out = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            out.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(out)


def _downsample_one(job) -> None:
    src, dst, factor = job
    if os.path.isfile(dst):
        return
    from PIL import Image

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        size = (round(im.width / factor), round(im.height / factor))
        im.resize(size, Image.BICUBIC).save(dst)


def downsample_images(data_dir: str, factor: int, workers: int = 8) -> str:
    """Create a cached ``images_<factor>`` folder (PNG) if missing."""
    src_dir = os.path.join(data_dir, "images")
    dst_dir = os.path.join(data_dir, f"images_{factor}")
    files = _rel_paths(src_dir)
    jobs = [
        (
            os.path.join(src_dir, f),
            os.path.join(dst_dir, os.path.splitext(f)[0] + ".png"),
            factor,
        )
        for f in files
    ]
    todo = [j for j in jobs if not os.path.isfile(j[1])]
    if todo:
        print(f"[ingest] downsampling {len(todo)} images by {factor}x -> {dst_dir}")
        from multiprocessing.pool import ThreadPool

        with ThreadPool(workers) as pool:
            list(pool.imap_unordered(_downsample_one, todo))
    return dst_dir


def ingest(cfg: IngestConfig) -> str:
    """Build scene.json + points.npz. Returns the manifest path."""
    try:
        import pycolmap
    except ImportError as e:
        raise ImportError(
            "pycolmap is required for ingest: pip install pycolmap"
        ) from e

    colmap_dir = os.path.join(cfg.data_dir, "sparse/0")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(cfg.data_dir, "sparse")
    assert os.path.exists(colmap_dir), f"COLMAP dir not found: {colmap_dir}"
    recon = pycolmap.Reconstruction(colmap_dir)

    cameras = {int(k): v for k, v in recon.cameras.items()}
    images = {int(k): v for k, v in recon.images.items()}
    image_ids = sorted(
        (int(i) for i in recon.reg_image_ids()), key=lambda i: images[i].name
    )
    if not image_ids:
        raise ValueError("No registered images in the COLMAP model.")

    # Intrinsics; require pinhole (no distortion).
    Ks: Dict[int, np.ndarray] = {}
    sizes: Dict[int, tuple] = {}
    for cam_id, cam in cameras.items():
        model = str(getattr(cam, "model_name", "") or getattr(cam, "model", ""))
        model = model.replace("CameraModelId.", "")
        if model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(
                f"Camera {cam_id} has model {model}; citygs only supports "
                "undistorted pinhole cameras. Run `colmap image_undistorter` first."
            )
        Ks[cam_id] = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        sizes[cam_id] = (int(cam.width), int(cam.height))

    # Extrinsics (c2w) in image-name order.
    w2c = []
    for i in image_ids:
        cam_from_world = images[i].cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        m = np.eye(4)
        m[:3, :4] = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        w2c.append(m)
    camtoworlds = np.linalg.inv(np.stack(w2c))
    names = [images[i].name for i in image_ids]
    cam_ids = [int(images[i].camera_id) for i in image_ids]

    # SfM points + per-image visibility index (for sparse-depth loss).
    points3D = {int(k): v for k, v in recon.points3D.items()}
    pids = sorted(points3D)
    xyz = np.array([points3D[p].xyz for p in pids], dtype=np.float32).reshape(-1, 3)
    rgb = np.array([points3D[p].color for p in pids], dtype=np.uint8).reshape(-1, 3)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    image_id_to_cam_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
    per_cam: List[List[int]] = [[] for _ in image_ids]
    for p in pids:
        for el in points3D[p].track.elements:
            cam_idx = image_id_to_cam_idx.get(int(el.image_id))
            if cam_idx is not None:
                per_cam[cam_idx].append(pid_to_idx[p])
    offsets = np.zeros(len(per_cam) + 1, dtype=np.int64)
    for i, lst in enumerate(per_cam):
        offsets[i + 1] = offsets[i] + len(lst)
    indices = (
        np.concatenate([np.asarray(lst, dtype=np.int32) for lst in per_cam])
        if offsets[-1]
        else np.empty(0, dtype=np.int32)
    )

    # Normalize the world space (rotation to z-up, recenter, unit-ish scale).
    if cfg.normalize:
        T1 = similarity_from_cameras(camtoworlds)
        camtoworlds = transform_cameras(T1, camtoworlds)
        xyz = transform_points(T1, xyz).astype(np.float32)
        T2 = align_principal_axes(xyz)
        camtoworlds = transform_cameras(T2, camtoworlds)
        xyz = transform_points(T2, xyz).astype(np.float32)
        transform = T2 @ T1
        if np.median(xyz[:, 2]) > np.mean(xyz[:, 2]):
            T3 = np.diag([1.0, -1.0, -1.0, 1.0])
            camtoworlds = transform_cameras(T3, camtoworlds)
            xyz = transform_points(T3, xyz).astype(np.float32)
            transform = T3 @ transform
    else:
        transform = np.eye(4)

    # Resolve the image directory (images_<factor> convention).
    image_dir_name = "images" if cfg.factor <= 1 else f"images_{cfg.factor}"
    image_dir = os.path.join(cfg.data_dir, image_dir_name)
    if cfg.factor > 1 and not os.path.isdir(image_dir):
        if cfg.write_downsampled:
            downsample_images(cfg.data_dir, cfg.factor, cfg.downsample_workers)
        else:
            raise ValueError(
                f"{image_dir} missing; pass --write-downsampled or create it."
            )
    # Map COLMAP names to files in the (possibly downsampled) image dir.
    colmap_files = _rel_paths(os.path.join(cfg.data_dir, "images"))
    image_files = _rel_paths(image_dir)
    if len(colmap_files) != len(image_files):
        raise ValueError(
            f"images/ has {len(colmap_files)} files but {image_dir_name}/ has "
            f"{len(image_files)}; regenerate the downsampled folder."
        )
    name_map = dict(zip(colmap_files, image_files))

    # COLMAP intrinsics may correspond to a different resolution than the
    # files on disk; rescale K to the actual image size.
    import imageio.v2 as imageio

    first = imageio.imread(os.path.join(image_dir, name_map[names[0]]))
    actual_h, actual_w = first.shape[:2]
    w0, h0 = sizes[cam_ids[0]]
    sx, sy = actual_w / w0, actual_h / h0

    metas = []
    for name, cid, c2w in zip(names, cam_ids, camtoworlds):
        K = Ks[cid].copy()
        w, h = sizes[cid]
        K[0, :] *= sx
        K[1, :] *= sy
        metas.append(
            CameraMeta(
                name=name,
                image=os.path.join(image_dir_name, name_map[name]),
                width=int(round(w * sx)),
                height=int(round(h * sy)),
                K=K.flatten().tolist(),
                camtoworld=c2w.flatten().tolist(),
                camera_id=cid,
            )
        )

    positions = camtoworlds[:, :3, 3]
    scene_scale = float(
        np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))
    )

    os.makedirs(cfg.result_dir, exist_ok=True)
    points_path = os.path.join(cfg.result_dir, "points.npz")
    np.savez(
        points_path,
        xyz=xyz,
        rgb=rgb,
        cam_point_indices=indices,
        cam_point_offsets=offsets,
    )

    manifest = SceneManifest(
        data_dir=os.path.abspath(cfg.data_dir),
        factor=cfg.factor,
        test_every=cfg.test_every,
        transform=transform.flatten().tolist(),
        scene_scale=scene_scale,
        cameras=metas,
        points_file="points.npz",
    )
    manifest_path = os.path.join(cfg.result_dir, "scene.json")
    manifest.save(manifest_path)
    print(
        f"[ingest] {len(metas)} cameras, {len(xyz)} points, "
        f"scene_scale={scene_scale:.3f} -> {manifest_path}"
    )
    return manifest_path


if __name__ == "__main__":
    import tyro

    ingest(tyro.cli(IngestConfig))
