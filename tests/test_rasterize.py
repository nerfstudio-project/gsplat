from typing import Optional

import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor

device = torch.device("cuda:0")


def _distortion_loss(
    weights: Tensor, t_mids: Tensor, ray_indices: Tensor, n_rays: int
) -> Tensor:
    from nerfacc import accumulate_along_rays, exclusive_sum

    loss_uni = 0.0
    loss_bi_0 = weights * t_mids * exclusive_sum(weights, indices=ray_indices)
    loss_bi_1 = weights * exclusive_sum(weights * t_mids, indices=ray_indices)
    loss_bi = 2 * (loss_bi_0 - loss_bi_1)
    loss = loss_uni + loss_bi
    loss = accumulate_along_rays(loss, None, ray_indices, n_rays)
    return loss


def _rasterize_gaussians(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
) -> Tensor:
    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        print("Please install the nerfacc package")
        exit()

    from gsplat import rasterize_indices

    H, W = img_height, img_width
    pixel_coords = torch.stack(
        torch.meshgrid(
            torch.arange(W, dtype=torch.long, device=device),
            torch.arange(H, dtype=torch.long, device=device),
            indexing="xy",
        ),
        dim=-1,
    )  # [H, W, 2]
    pixel_coords = pixel_coords.reshape(-1, 2)

    gs_ids, pixel_ids = rasterize_indices(
        xys, depths, radii, conics, num_tiles_hit, opacity, H, W
    )
    pixel_ids = pixel_ids.long()

    deltas = pixel_coords[pixel_ids] - xys[gs_ids]
    _conics = conics[gs_ids]
    sigmas = (
        0.5 * (_conics[:, 0] * deltas[:, 0] ** 2 + _conics[:, 2] * deltas[:, 1] ** 2)
        + _conics[:, 1] * deltas[:, 0] * deltas[:, 1]
    )
    alphas = opacity[gs_ids, 0] * torch.exp(-sigmas)
    alphas = torch.clamp_max(alphas, 0.999)
    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=pixel_ids, n_rays=H * W
    )
    render = accumulate_along_rays(weights, colors[gs_ids], pixel_ids, H * W)
    render = render.reshape(H, W, -1)

    # distloss = _distortion_loss(weights, depths[gs_ids], pixel_ids, H * W)
    # distloss = distloss.reshape(H, W, 1)

    if background is not None or return_alpha:
        render_alpha = accumulate_along_rays(weights, None, pixel_ids, H * W)
        render_alpha = render_alpha.reshape(H, W, 1)
        if background is not None:
            render = render + (1.0 - render_alpha) * background

    if return_alpha:
        return render, render_alpha
    else:
        return render


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize(N: int = 1000, D: int = 3, profile: bool = False):
    import tqdm

    from gsplat import project_gaussians, rasterize_gaussians

    torch.manual_seed(42)

    means3d = torch.rand((N, 3), device=device, requires_grad=False)
    scales = torch.rand((N, 3), device=device) * 5
    quats = torch.randn((N, 4), device=device)
    quats /= torch.linalg.norm(quats, dim=-1, keepdim=True)
    opacities = torch.ones((N, 1), device=device)
    rgbs = torch.rand((N, D), device=device)
    background = torch.zeros(D, device=device)

    viewmat = projmat = torch.eye(4, device=device)
    fx = fy = 3.0
    H, W = 256, 384
    BLOCK_X = BLOCK_Y = 16
    tile_bounds = (W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1

    xys, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(
        means3d,
        scales,
        1,
        quats,
        viewmat,
        projmat,
        fx,
        fy,
        W / 2,
        H / 2,
        H,
        W,
        tile_bounds,
    )
    # warmup
    if profile:
        rasterize_gaussians(
            xys, depths, radii, conics, num_tiles_hit, rgbs, opacities, H, W, background
        )
        _rasterize_gaussians(
            xys, depths, radii, conics, num_tiles_hit, rgbs, opacities, H, W, background
        )

    pbar = tqdm.trange(100) if profile else range(1)
    for _ in pbar:
        render = rasterize_gaussians(
            xys, depths, radii, conics, num_tiles_hit, rgbs, opacities, H, W, background
        )

    pbar = tqdm.trange(100) if profile else range(1)
    for _ in pbar:
        render2 = _rasterize_gaussians(
            xys, depths, radii, conics, num_tiles_hit, rgbs, opacities, H, W, background
        )
    torch.testing.assert_close(render, render2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    test_rasterize(args.N, args.D, args.profile)
