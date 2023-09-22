"""Test against ref bindings

Make sure you have the ref bindings installed:
    - install ref bindings: cd ref_rast && pip install -e .
"""


import torch
import imageio

from diff_rast.rasterize import RasterizeGaussians
from diff_rast.project_gaussians import ProjectGaussians

from ref_rast import GaussianRasterizationSettings, rasterize_gaussians

def test_bindings_forward(save_img=False):
    means3d, sh, opacities, colors, covs3d, scales, quats, viewmat, projmat = _init_gaussians()

    out_color = _run_ref(
        means3d.clone(),
        sh.clone(),
        opacities.clone(),
        colors.clone(),
        covs3d.clone(),
        scales.clone(),
        quats.clone(),
        viewmat.clone(),
        projmat.clone(),
    )
    ref_img = (255 * out_color.detach().cpu()).byte()
    imageio.imwrite("test_reference_forward.png", ref_img)

    color = _run_diff_rast(means3d, colors, opacities, scales, quats, viewmat, projmat)
    img = (255 * color.detach().cpu()).byte()
    imageio.imwrite("test_diff_forward.png", img)

    torch.testing.assert_close(color, out_color)


def _run_ref(means3D, sh, opacities, colors, covs3d, scales, quats, viewmat, projmat):
    settings = GaussianRasterizationSettings(
        img_height,
        img_width,
        tanfovx,
        tanfovy,
        bg,
        scale_modifier,
        viewmat,
        projmat,
        sh_degree=0,
        campos=viewmat[:3, 3],
        prefiltered=False,
        debug=True,
    )

    out_color, radii = rasterize_gaussians(
        means3D.clone(), None, sh, colors.clone(), opacities.clone(), scales.clone(), quats.clone(), covs3d, settings
    )

    return out_color.permute(1, 2, 0)


def _run_diff_rast(means, rgbs, opacities, scales, quats, viewmat, projmat):
    xys, depths, radii, conics, num_tiles_hit = ProjectGaussians.apply(
        means, scales, scale_modifier, quats, viewmat, projmat, fx, fy, img_height, img_width, TILE_BOUNDS
    )
    out_img = RasterizeGaussians.apply(xys, depths, radii, conics, num_tiles_hit, rgbs, opacities, img_height, img_width)

    return out_img


def _init_gaussians():
    means3d = torch.randn((num_points, 3), device=device)
    sh = torch.Tensor([]).to(device)
    opacities = torch.ones((num_points, 1), device=device) * 0.9
    colors = torch.rand((num_points, 3), device=device)
    covs3d = torch.Tensor([]).to(device)
    scales = torch.rand((num_points, 3), device=device)
    quats = torch.randn((num_points, 4), device=device)
    quats /= torch.linalg.norm(quats, dim=-1, keepdim=True)

    viewmat = torch.eye(4, device=device)

    projmat = viewmat

    means3d.requires_grad = True
    scales.requires_grad = True
    quats.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True

    return means3d, sh, opacities, colors, covs3d, scales, quats, viewmat, projmat


if __name__ == "__main__":
    device = torch.device("cuda:0")
    num_points = 100

    fx = 3.0
    fy = 3.0
    img_height = 256*2
    img_width = 256
    tanfovx = 0.5 * img_width / fx
    tanfovy = 0.5 * img_height / fy
    bg = torch.ones(3, device=device)
    scale_modifier = 1.0
    TILE_BOUNDS = (img_width + 16 - 1) // 16, (img_height + 16 - 1) // 16, 1

    test_bindings_forward(save_img=True)
