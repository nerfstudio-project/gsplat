import math

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")

# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
# @pytest.fixture
# def test_data_2():
#     (
#         means,
#         quats,
#         scales,
#         opacities,
#         colors,
#         viewmats,
#         Ks,
#         width,
#         height,
#     ) = load_test_data(device=device)
#     colors = colors[None].repeat(len(viewmats), 1, 1)
#     return {
#         "means": means,
#         "quats": quats,
#         "scales": scales,
#         "opacities": opacities,
#         "colors": colors,
#         "viewmats": viewmats,
#         "Ks": Ks,
#         "width": width,
#         "height": height,
#     }


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
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
    
    # forward
    _radii, _means2d, _depths, _ray_Ms = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    _ray_Ms = _ray_Ms.permute((0, 1, 3, 2)) # TODO(WZ): Figure out why do we need to permute here
    
    radii, means2d, depths, ray_Ms = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    
    # TODO (WZ): is the following true for 2dgs as while?
    # radii is integer so we allow for 1 unit difference
    valid = (radii > 0) & (_radii > 0)
    torch.testing.assert_close(radii, _radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ray_Ms[valid], _ray_Ms[valid], rtol=1e-4, atol=1e-4)
    
    # backward    
    v_means2d = torch.randn_like(means2d) * radii[..., None]
    v_depths = torch.randn_like(depths) * radii
    v_ray_Ms = torch.randn_like(ray_Ms) * radii[..., None, None]
    
    v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (ray_Ms * v_ray_Ms).sum()
        + 0,
        (quats, scales, means)
    )
    _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_ray_Ms * v_ray_Ms).sum()
        + 0,
        (quats, scales, means),
    )
    
    # torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=2e-1, atol=1e-2)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)
    
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
