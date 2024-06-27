import math

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
def test_data_2():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(device=device)
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


def test_data():
    N = 2
    xs = torch.linspace(-1, 1, N, device=device)
    ys = torch.linspace(-1, 1, N, device=device)
    xys = torch.stack(torch.meshgrid(xs, ys), dim=-1).reshape(-1, 2)
    zs = torch.ones_like(xys[:, :1]) * 3
    means = torch.cat([xys, zs], dim=-1)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0]], device=device).repeat(len(means), 1)
    scales = torch.ones_like(means) * 0.1
    opacities = torch.ones(len(means), device=device) * 0.5
    colors = torch.rand(len(means), 3, device=device)
    viewmats = torch.eye(4, device=device).reshape(1, 4, 4)
    W, H = 640, 480
    fx, fy, cx, cy = W, W, W // 2, H // 2
    Ks = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device
    ).reshape(1, 3, 3)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": W,
        "height": H,
    }

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_projection(test_data):
    from gsplat.cuda._torch_impl import _fully_fused_projection_2dgs
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True
    
    _radii, _means2d, _depths, _ray_Ms = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    
    radii, means2d, depths, ray_transformations = fully_fused_projection_2dgs(
        
    )
    # print(f"{_radii.shape=} {_means2d.shape=} {_depths.shape=} {_ray_Ms.shape=}")
    # import ipdb

    # ipdb.set_trace()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_fully_fused_projection_packed(test_data):
    pass

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3, 32, 128])
def test_rasterize_to_pixels(test_data, channels: int):
    pass




if __name__ == "__main__":
    test_projection(test_data())
    print("All tests passed.")
