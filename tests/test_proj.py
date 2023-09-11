import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_project_gaussians_forward():
    from diff_rast import _torch_impl, cuda_lib

    torch.manual_seed(42)

    num_points = 1_000_000
    means3d = torch.randn((num_points, 3), device=device, requires_grad=True)
    scales = torch.randn((num_points, 3), device=device)
    glob_scale = 0.3
    quats = torch.randn((num_points, 4), device=device)
    quats /= torch.linalg.norm(quats, dim=-1, keepdim=True)
    viewmat = torch.eye(4, device=device)
    projmat = torch.eye(4, device=device)
    fx, fy = 3.0, 3.0
    H, W = 512, 512

    BLOCK_X, BLOCK_Y = 16, 16
    tile_bounds = (W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1

    (
        cov3d,
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
    ) = cuda_lib.project_gaussians_forward(
        num_points,
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        projmat,
        fx,
        fy,
        (H, W),  # TODO(ruilong): Is this HW or WH?
        tile_bounds,
    )

    (
        _cov3d,
        _xys,
        _depths,
        _radii,
        _conics,
        _num_tiles_hit,
        _masks,
    ) = _torch_impl.project_gaussians_forward(
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        projmat,
        fx,
        fy,
        (H, W),
        tile_bounds,
    )

    assert torch.allclose(
        cov3d[_masks],
        _cov3d.view(-1, 9)[_masks][:, [0, 1, 2, 4, 5, 8]],
        atol=1e-5,
    )
    assert torch.allclose(xys[_masks], xys[_masks])
    assert torch.allclose(depths[_masks], depths[_masks])
    assert torch.allclose(radii[_masks], radii[_masks])
    assert torch.allclose(conics[_masks], conics[_masks])
    assert torch.allclose(num_tiles_hit[_masks], num_tiles_hit[_masks])


if __name__ == "__main__":
    test_project_gaussians_forward()
