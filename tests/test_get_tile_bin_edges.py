import pytest
import torch


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_get_tile_bin_edges():
    from gsplat import _torch_impl
    from gsplat.get_tile_bin_edges import get_tile_bin_edges

    torch.manual_seed(42)

    num_points = 100

    means3d = torch.randn((num_points, 3), device=device, requires_grad=True)
    scales = torch.randn((num_points, 3), device=device)
    glob_scale = 0.3
    quats = torch.randn((num_points, 4), device=device)
    quats /= torch.linalg.norm(quats, dim=-1, keepdim=True)
    viewmat = torch.eye(4, device=device)
    projmat = torch.eye(4, device=device)
    fx, fy = 3.0, 3.0
    H, W = 512, 512
    clip_thresh = 0.01

    BLOCK_X, BLOCK_Y = 16, 16
    tile_bounds = (W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1

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
    _depths = _depths.contiguous()

    (
        _isect_ids_unsorted,
        _gaussian_ids_unsorted,
    ) = _torch_impl.map_gaussian_to_intersects(
        num_points, _xys, _depths, _radii, _cum_tiles_hit, tile_bounds
    )

    # Sorting isect_ids_unsorted
    sorted_values, sorted_indices = torch.sort(_isect_ids_unsorted)

    _isect_ids_sorted = sorted_values
    _gaussian_ids_sorted = torch.gather(_gaussian_ids_unsorted, 0, sorted_indices)

    _tile_bins = _torch_impl.get_tile_bin_edges(_num_intersects, _isect_ids_sorted)
    tile_bins = get_tile_bin_edges(_num_intersects, _isect_ids_sorted)

    torch.testing.assert_close(_tile_bins, tile_bins)


if __name__ == "__main__":
    test_get_tile_bin_edges()
