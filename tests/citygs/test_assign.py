import numpy as np
import torch

from citygs.partition.assign import (
    assign_cameras_to_blocks,
    block_camera_contribution,
    block_cost,
)
from citygs.partition.contraction import contract
from citygs.partition.partition import assign_gaussians_to_blocks, make_grid_blocks


def _camera_at(position, look_at):
    z = np.asarray(look_at, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, position
    return c2w


def test_contribution_prefers_visible_cluster():
    # A dense cluster at the origin; one camera looks at it, one looks away.
    torch.manual_seed(0)
    means = torch.randn(2000, 3) * 0.1
    scales = torch.full((2000, 3), 0.05)
    opacities = torch.full((2000,), 0.8)
    c2w_look = _camera_at([0, -2, 0.5], [0, 0, 0])
    c2w_away = _camera_at([0, -2, 0.5], [0, -10, 0.5])
    camtoworlds = torch.tensor(np.stack([c2w_look, c2w_away]), dtype=torch.float32)
    Ks = torch.tensor(
        [[[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]]] * 2, dtype=torch.float32
    )
    sizes = torch.tensor([[128.0, 96.0]] * 2)
    contrib = block_camera_contribution(
        means, scales, opacities, camtoworlds, Ks, sizes
    )
    assert contrib[0] > 0.05
    assert contrib[1] < 1e-6


def test_assign_cameras_guarantees(toy_manifest):
    manifest = toy_manifest
    # Gaussians: a cluster in each half-space (block 0: x<0, block 1: x>0).
    torch.manual_seed(0)
    left = torch.randn(1500, 3) * 0.15 + torch.tensor([-0.8, 0.0, 0.0])
    right = torch.randn(1500, 3) * 0.15 + torch.tensor([0.8, 0.0, 0.0])
    means = torch.cat([left, right])
    scales = torch.full((3000, 3), 0.05)
    opacities = torch.full((3000,), 0.9)

    ctr = manifest.contraction
    contracted = contract(means.double().numpy(), ctr.center, ctr.half_extent)
    blocks = manifest.blocks
    gids = assign_gaussians_to_blocks(contracted, blocks)

    train_idx = manifest.train_indices()
    cams = [manifest.cameras[i] for i in train_idx]
    # Point every camera straight down at the plane below it.
    camtoworlds = torch.tensor(
        np.stack(
            [
                _camera_at(
                    c.camtoworld_np()[:3, 3], c.camtoworld_np()[:3, 3] * [1, 1, 0]
                )
                for c in cams
            ]
        ),
        dtype=torch.float32,
    )
    Ks = torch.tensor(np.stack([c.K_np() for c in cams]), dtype=torch.float32)
    sizes = torch.tensor([[c.width, c.height] for c in cams], dtype=torch.float32)
    cam_contracted = contract(
        camtoworlds[:, :3, 3].double().numpy(), ctr.center, ctr.half_extent
    )

    contrib = assign_cameras_to_blocks(
        blocks,
        gids,
        means,
        scales,
        opacities,
        camtoworlds,
        Ks,
        sizes,
        camera_indices=train_idx,
        cam_contracted=cam_contracted,
        coverage_threshold=0.05,
        min_cameras=3,
    )
    assert contrib.shape == (2, len(train_idx))
    # Every block got at least min_cameras.
    for b in blocks:
        assert len(b.cameras) >= 3
        # Assigned camera indices are global manifest indices (train split).
        assert set(b.cameras) <= set(train_idx.tolist())
    # Every train camera is used by at least one block.
    used = set(blocks[0].cameras) | set(blocks[1].cameras)
    assert used == set(train_idx.tolist())
    # Cameras hovering over the left half must train the left block.
    left_cams = [
        i for i in train_idx if manifest.cameras[i].camtoworld_np()[0, 3] < -0.4
    ]
    assert set(left_cams) <= set(blocks[0].cameras)


def test_block_cost_monotonic(toy_manifest):
    b0, b1 = toy_manifest.blocks
    b0.num_gaussians, b0.cameras = 1000, list(range(10))
    b1.num_gaussians, b1.cameras = 2000, list(range(10))
    assert block_cost(b1) > block_cost(b0)
