import pytest
import torch


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize_forward_kernel():
    from gsplat import _torch_impl
    from gsplat.rasterize_forward_kernel import RasterizeForwardKernel

    torch.manual_seed(42)

    num_points = 100

    means3d = torch.rand((num_points, 3), device=device)
    scales = torch.rand((num_points, 3), device=device)
    colors = torch.randn((num_points, 3), device=device)
    opacities = torch.randn((num_points, 1), device=device)
    background = torch.ones((3,), device=device)
    glob_scale = 0.3
    quats = torch.rand((num_points, 4), device=device)
    quats /= torch.linalg.norm(quats, dim=-1, keepdim=True)
    viewmat = torch.eye(4, device=device)
    projmat = torch.eye(4, device=device)
    fx, fy = 0.5, 0.5
    H, W = 32, 32
    cy, cx = H // 2, W // 2
    clip_thresh = 0.01

    BLOCK_X, BLOCK_Y = 16, 16
    block = BLOCK_X, BLOCK_Y, 1
    tile_bounds = (W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1
    img_size = W, H, 1

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
        cx,
        cy,
        (H, W),
        tile_bounds,
        clip_thresh,
    )

    _xys = _xys[_masks]
    _depths = _depths[_masks]
    _radii = _radii[_masks]
    _conics = _conics[_masks]
    _num_tiles_hit = _num_tiles_hit[_masks]

    num_points = num_points - torch.count_nonzero(~_masks).item()

    _cum_tiles_hit = torch.cumsum(_num_tiles_hit, dim=0, dtype=torch.int32)
    _num_intersects = _cum_tiles_hit[-1].item()
    _depths = _depths

    (
        _isect_ids_unsorted,
        _gaussian_ids_unsorted,
        _isect_ids_sorted,
        _gaussian_ids_sorted,
        _tile_bins,
    ) = _torch_impl.bin_and_sort_gaussians(
        num_points, _num_intersects, _xys, _depths, _radii, _cum_tiles_hit, tile_bounds
    )

    (
        out_img,
        final_Ts,
        final_idx,
    ) = RasterizeForwardKernel.apply(
        tile_bounds,
        block,
        img_size,
        _gaussian_ids_sorted.contiguous(),
        _tile_bins,
        _xys.contiguous(),
        _conics.contiguous(),
        colors,
        opacities,
        background,
    )

    (
        _out_img,
        _final_Ts,
        _final_idx,
    ) = _torch_impl.rasterize_forward_kernel(
        tile_bounds,
        block,
        img_size,
        _gaussian_ids_sorted,
        _tile_bins,
        _xys,
        _conics,
        colors,
        opacities,
        background,
    )

    torch.testing.assert_close(_out_img, out_img, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(_final_Ts, final_Ts)
    # torch.testing.assert_close(_final_idx, final_idx)


if __name__ == "__main__":
    test_rasterize_forward_kernel()
