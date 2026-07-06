import numpy as np
import pytest
import torch

from citygs.scene.manifest import BlockMeta, CameraMeta, Contraction, SceneManifest


def make_camera(index: int, position, look_dir=(0, 0, -1), width=64, height=48):
    """A simple pinhole camera at `position` looking along `look_dir`
    (OpenCV convention: +z forward, +y down)."""
    z = np.asarray(look_dir, dtype=np.float64)
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x, y, z
    c2w[:3, 3] = position
    K = np.array([[60.0, 0, width / 2], [0, 60.0, height / 2], [0, 0, 1]])
    return CameraMeta(
        name=f"cam{index:04d}.png",
        image=f"images/cam{index:04d}.png",
        width=width,
        height=height,
        K=K.flatten().tolist(),
        camtoworld=c2w.flatten().tolist(),
    )


@pytest.fixture
def toy_manifest(tmp_path):
    """A 2x1 partitioned scene with cameras on a grid, saved to disk."""
    rng = np.random.default_rng(0)
    cameras = []
    for i in range(24):
        # Cameras hover above the plane z=0, spread over [-1, 1]^2.
        pos = np.array(
            [(i % 6) / 2.5 - 1.0, (i // 6) / 1.5 - 1.0, 1.0 + 0.05 * rng.random()]
        )
        cameras.append(make_camera(i, pos))
    manifest = SceneManifest(
        data_dir=str(tmp_path / "data"),
        factor=1,
        test_every=8,
        cameras=cameras,
        scene_scale=1.5,
        contraction=Contraction(center=[0.0, 0.0, 0.5], half_extent=[1.2, 1.2, 1.0]),
    )
    manifest.blocks = [
        BlockMeta(0, (0, 0), bmin=[-2.0, -2.0, -2.0], bmax=[0.0, 2.0, 2.0]),
        BlockMeta(1, (1, 0), bmin=[0.0, -2.0, -2.0], bmax=[2.0, 2.0, 2.0]),
    ]
    path = tmp_path / "scene.json"
    manifest.save(str(path))
    return SceneManifest.load(str(path))


def make_splats(n: int, low=(-1, -1, -1), high=(1, 1, 1), seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * (
        torch.tensor(high, dtype=torch.float32) - torch.tensor(low, dtype=torch.float32)
    ) + torch.tensor(low, dtype=torch.float32)
    return {
        "means": means,
        "scales": torch.full((n, 3), -4.0),
        "quats": torch.tensor([1.0, 0, 0, 0]).repeat(n, 1),
        "opacities": torch.zeros(n),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 15, 3),
    }
