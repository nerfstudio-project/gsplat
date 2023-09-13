import pytest
import torch

device = torch.device("cuda:0")


def _assert_tensors_close(t1: torch.Tensor, t2: torch.Tensor, **kwargs):
    assert torch.allclose(t1, t2, **kwargs), (
        f"Tensors not close: {t1} != {t2}"
        f"\nDiff: {t1 - t2}"
        f"\nmax atol: {torch.max(torch.abs(t1 - t2))}"
        f"\nmax rtol: {torch.max(torch.abs((t1 - t2) / t1))}"
    )

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

    _assert_tensors_close(
        cov3d[_masks],
        _cov3d.view(-1, 9)[_masks][:, [0, 1, 2, 4, 5, 8]],
        atol=1e-5,
    )
    _assert_tensors_close(xys[_masks], _xys[_masks])
    _assert_tensors_close(depths[_masks], _depths[_masks])
    _assert_tensors_close(radii[_masks], _radii[_masks])
    _assert_tensors_close(conics[_masks], _conics[_masks])
    _assert_tensors_close(num_tiles_hit[_masks], _num_tiles_hit[_masks])



if __name__ == "__main__":
    test_project_gaussians_forward()
