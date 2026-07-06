"""Interactive block viewer built on viser (no custom renderer).

viser does the splat rendering; this app only adds block semantics:

  * one splat cloud per block with a visibility checkbox,
  * "color by block" mode,
  * block cell wireframes (contracted cells mapped back to world),
  * per-block camera frustums (which images train this block).

Run: ``python -m citygs.viz.viewer --manifest results/scene.json``
"""

import glob
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from ..ckpt import load_ckpt
from ..partition.contraction import uncontract
from ..scene.manifest import SceneManifest

_TAB20 = (
    np.array(
        [
            [31, 119, 180],
            [174, 199, 232],
            [255, 127, 14],
            [255, 187, 120],
            [44, 160, 44],
            [152, 223, 138],
            [214, 39, 40],
            [255, 152, 150],
            [148, 103, 189],
            [197, 176, 213],
            [140, 86, 75],
            [196, 156, 148],
            [227, 119, 194],
            [247, 182, 210],
            [127, 127, 127],
            [199, 199, 199],
            [188, 189, 34],
            [219, 219, 141],
            [23, 190, 207],
            [158, 218, 229],
        ],
        dtype=np.float32,
    )
    / 255.0
)


@dataclass
class ViewerConfig:
    manifest: str = "results/scene.json"
    blocks_dir: str = "results/blocks"
    port: int = 8080
    # Cap splats per block for interactivity (0 = no cap).
    max_splats_per_block: int = 1_000_000
    frustum_scale: float = 0.05


def _quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    """wxyz quaternions [N, 4] -> rotation matrices [N, 3, 3]."""
    q = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return R.reshape(-1, 3, 3)


def _rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def _splat_arrays(splats: Dict[str, torch.Tensor], max_n: int):
    """Convert a citygs checkpoint into viser's splat inputs."""
    means = splats["means"].numpy()
    n = len(means)
    if 0 < max_n < n:
        sel = np.random.default_rng(0).choice(n, max_n, replace=False)
        splats = {k: v[sel] for k, v in splats.items()}
        means = splats["means"].numpy()
    scales = splats["scales"].exp().numpy()
    opacities = torch.sigmoid(splats["opacities"]).numpy()[:, None]
    C0 = 0.28209479177387814
    rgbs = np.clip(splats["sh0"].numpy()[:, 0, :] * C0 + 0.5, 0.0, 1.0)
    R = _quat_to_rotmat(splats["quats"].numpy())
    S = scales[:, None, :] * np.eye(3)[None]  # [N, 3, 3] diag
    M = R @ S
    covariances = M @ M.transpose(0, 2, 1)
    return means, covariances, rgbs, opacities


def _cell_edges(block, contraction, clamp: float = 1.5) -> np.ndarray:
    """World-space wireframe of a block cell. [E, 2, 3]."""
    bmin = np.clip(np.asarray(block.bmin, dtype=np.float64), -clamp, clamp)
    bmax = np.clip(np.asarray(block.bmax, dtype=np.float64), -clamp, clamp)
    corners_c = np.array(
        [
            [x, y, z]
            for x in (bmin[0], bmax[0])
            for y in (bmin[1], bmax[1])
            for z in (bmin[2], bmax[2])
        ]
    )
    corners = uncontract(corners_c, contraction.center, contraction.half_extent)
    edge_idx = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return np.stack([[corners[a], corners[b]] for a, b in edge_idx])


def serve(cfg: ViewerConfig) -> None:
    import viser

    manifest = SceneManifest.load(cfg.manifest)
    assert manifest.blocks, "no blocks in manifest — run the partition stage first"
    assert manifest.contraction is not None

    ckpts = sorted(glob.glob(os.path.join(cfg.blocks_dir, "block_*.pt")))
    assert ckpts, f"no block checkpoints in {cfg.blocks_dir}"

    server = viser.ViserServer(port=cfg.port)
    server.scene.set_up_direction("+z")

    color_mode = server.gui.add_dropdown(
        "color mode", options=("rgb", "block"), initial_value="rgb"
    )
    show_cells = server.gui.add_checkbox("show block cells", initial_value=True)
    frustum_block = server.gui.add_dropdown(
        "camera frustums",
        options=("none",) + tuple(str(b.block_id) for b in manifest.blocks),
        initial_value="none",
    )

    splat_handles = {}
    cell_handles = {}
    block_data = {}
    checkboxes = {}

    with server.gui.add_folder("blocks"):
        for path in ckpts:
            splats, meta = load_ckpt(path)
            block_id = int(meta["block_id"])
            block_data[block_id] = _splat_arrays(splats, cfg.max_splats_per_block)
            checkboxes[block_id] = server.gui.add_checkbox(
                f"block {block_id} ({len(splats['means']):,})", initial_value=True
            )

    def add_splats():
        for block_id, (means, covs, rgbs, opac) in block_data.items():
            if color_mode.value == "block":
                rgbs = np.broadcast_to(_TAB20[block_id % 20], rgbs.shape).copy()
            splat_handles[block_id] = server.scene.add_gaussian_splats(
                f"/blocks/{block_id}",
                centers=means,
                covariances=covs.astype(np.float32),
                rgbs=rgbs,
                opacities=opac,
            )
            splat_handles[block_id].visible = checkboxes[block_id].value

    add_splats()

    for block in manifest.blocks:
        if block.block_id not in block_data:
            continue
        edges = _cell_edges(block, manifest.contraction)
        color = (_TAB20[block.block_id % 20] * 255).astype(np.uint8)
        cell_handles[block.block_id] = server.scene.add_line_segments(
            f"/cells/{block.block_id}",
            points=edges.astype(np.float32),
            colors=np.broadcast_to(color, (len(edges), 2, 3)).copy(),
            line_width=2.0,
        )

    frustum_handles: List = []

    def update_frustums(_=None):
        for h in frustum_handles:
            h.remove()
        frustum_handles.clear()
        if frustum_block.value == "none":
            return
        block = manifest.block(int(frustum_block.value))
        color = tuple((_TAB20[block.block_id % 20] * 255).astype(int).tolist())
        for cam_idx in block.cameras[:500]:
            cam = manifest.cameras[cam_idx]
            c2w = cam.camtoworld_np()
            K = cam.K_np()
            fov = 2 * np.arctan2(cam.height / 2, K[1, 1])
            frustum_handles.append(
                server.scene.add_camera_frustum(
                    f"/frustums/{cam_idx}",
                    fov=float(fov),
                    aspect=cam.width / cam.height,
                    scale=cfg.frustum_scale * manifest.scene_scale,
                    wxyz=_rotmat_to_wxyz(c2w[:3, :3]),
                    position=c2w[:3, 3],
                    color=color,
                )
            )

    def update_visibility(_=None):
        for block_id, handle in splat_handles.items():
            handle.visible = checkboxes[block_id].value

    def update_cells(_=None):
        for handle in cell_handles.values():
            handle.visible = show_cells.value

    for cb in checkboxes.values():
        cb.on_update(update_visibility)
    color_mode.on_update(lambda _: add_splats())
    show_cells.on_update(update_cells)
    frustum_block.on_update(update_frustums)

    print(f"[viewer] http://localhost:{cfg.port} — ctrl-c to quit")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    import tyro

    serve(tyro.cli(ViewerConfig))
