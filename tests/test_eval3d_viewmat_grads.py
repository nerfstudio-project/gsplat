# SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3DGUT pose gradients against PyTorch autograd and finite differences.

Poses use R0 Exp(xi), t0 + dt; off-manifold derivatives are unspecified.
"""

import math
import os

import pytest
import torch

import gsplat
from gsplat._helper import load_test_data
from gsplat.cuda._wrapper import RollingShutterType

device = torch.device("cuda:0")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device"),
    pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in"),
]


def _so3_exp(xi: torch.Tensor) -> torch.Tensor:
    """[..., 3] rotation vectors -> [..., 3, 3] rotations (differentiable at 0)."""
    x, y, z = xi.unbind(-1)
    o = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([o, -z, y], -1),
            torch.stack([z, o, -x], -1),
            torch.stack([-y, x, o], -1),
        ],
        -2,
    )
    return torch.linalg.matrix_exp(K)


def _perturb(
    viewmats: torch.Tensor, xi: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """World-to-camera [R | t] -> [R Exp(xi) | t + dt]."""
    R = viewmats[..., :3, :3] @ _so3_exp(xi)
    t = viewmats[..., :3, 3] + dt
    return torch.cat([torch.cat([R, t[..., None]], -1), viewmats[..., 3:, :]], -2)


def _pose_params(C: int, n_endpoints: int):
    return [
        torch.zeros(C, 3, device=device, requires_grad=True)
        for _ in range(2 * n_endpoints)
    ]


def _compare(a, b):
    a = torch.cat([x.flatten() for x in a]).double()
    b = torch.cat([x.flatten() for x in b]).double()
    rel = ((a - b).norm() / b.norm()).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return rel, cos


@pytest.fixture
def scene():
    means, quats, scales, opacities, _, viewmats, Ks, width, height = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    C = 2
    downscale = 8
    Ks = Ks[:C].clone()
    Ks[:, :2, :] /= downscale
    torch.manual_seed(0)
    N = means.shape[0]
    return {
        "means": means,
        "quats": quats,
        "scales": scales * 0.1,
        "opacities": opacities,
        "colors": torch.rand(C, N, 3, device=device),
        "viewmats": viewmats[:C].contiguous(),
        "Ks": Ks,
        "width": width // downscale,
        "height": height // downscale,
    }


def _rs_end_pose(viewmats: torch.Tensor, angle: float) -> torch.Tensor:
    """End-of-readout pose: small rotation (angle rad) and translation."""
    g = torch.Generator(device="cpu").manual_seed(1)
    axis = torch.randn(viewmats.shape[0], 3, generator=g)
    axis = (axis / axis.norm(dim=-1, keepdim=True) * angle).to(device)
    dt = (torch.randn(viewmats.shape[0], 3, generator=g) * 0.05).to(device)
    return _perturb(viewmats, axis, dt).detach().contiguous()


def _isects(scene, tile_size):
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
    )

    C = scene["viewmats"].shape[0]
    width, height = scene["width"], scene["height"]
    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        scene["means"],
        scene["quats"],
        scene["scales"],
        scene["opacities"],
        scene["viewmats"],
        scene["Ks"],
        width,
        height,
    )
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    return isect_offsets.reshape(C, tile_height, tile_width), flatten_ids


@pytest.mark.parametrize(
    "rs_type,rs_angle",
    [
        (RollingShutterType.GLOBAL, 0.0),
        (RollingShutterType.ROLLING_TOP_TO_BOTTOM, 0.02),
        (RollingShutterType.ROLLING_LEFT_TO_RIGHT, 0.02),
        (
            RollingShutterType.ROLLING_BOTTOM_TO_TOP,
            0.0,
        ),  # R0 == R1 (near-parallel slerp)
        (RollingShutterType.ROLLING_RIGHT_TO_LEFT, 0.3),
    ],
)
def test_viewmat_grads_match_torch_reference(scene, rs_type, rs_angle):
    """CUDA pose gradients vs autograd through the PyTorch eval3d reference."""
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
    from gsplat.cuda._wrapper import rasterize_to_pixels_eval3d_extra

    tile_size = 16
    C = scene["viewmats"].shape[0]
    rolling = rs_type != RollingShutterType.GLOBAL
    isect_offsets, flatten_ids = _isects(scene, tile_size)
    opacities = torch.broadcast_to(
        scene["opacities"][None], (C, scene["means"].shape[0])
    )
    vm1_base = _rs_end_pose(scene["viewmats"], rs_angle) if rolling else None

    torch.manual_seed(2)
    H, W = scene["height"], scene["width"]
    w_colors = torch.randn(C, H, W, 3, device=device)
    w_alphas = torch.randn(C, H, W, 1, device=device)

    def grads(render_fn):
        params = _pose_params(C, 2 if rolling else 1)
        vm0 = _perturb(scene["viewmats"], params[0], params[1])
        vm1 = _perturb(vm1_base, params[2], params[3]) if rolling else None
        colors, alphas = render_fn(vm0, vm1)
        loss = (colors * w_colors).sum() + (alphas * w_alphas).sum()
        return torch.autograd.grad(loss, params)

    def cuda_render(vm0, vm1):
        out = rasterize_to_pixels_eval3d_extra(
            scene["means"],
            scene["quats"],
            scene["scales"],
            scene["colors"],
            opacities,
            vm0,
            scene["Ks"],
            W,
            H,
            tile_size,
            isect_offsets,
            flatten_ids,
            rolling_shutter=rs_type,
            viewmats_rs=vm1,
        )
        return out[0], out[1]

    def ref_render(vm0, vm1):
        out = _rasterize_to_pixels_eval3d(
            scene["means"],
            scene["quats"],
            scene["scales"],
            scene["colors"],
            opacities,
            vm0,
            scene["Ks"],
            W,
            H,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            rs_type=rs_type,
            viewmats_rs=vm1,
        )
        return out[0], out[1]

    g_cuda = grads(cuda_render)
    g_ref = grads(ref_render)
    for g in g_cuda:
        assert torch.isfinite(g).all()
    rel, cos = _compare(g_cuda, g_ref)
    print(f"{rs_type.name} angle={rs_angle}: rel-L2 {rel:.3e}, cosine {cos:.8f}")
    assert rel < 1e-2 and cos > 0.9999, (rel, cos)


@pytest.fixture
def synthetic():
    """Sparse scene for finite differences, with 40 Gaussians and two cameras.

    Large Gaussians limit the effect of cutoff crossings on numerical derivatives.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 40
    means = torch.cat(
        [torch.rand(n, 2, generator=g) * 4 - 2, torch.rand(n, 1, generator=g) * 3 + 4],
        -1,
    )
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    scales = torch.rand(n, 3, generator=g) * 0.3 + 0.2
    opacities = torch.rand(n, generator=g) * 0.5 + 0.3
    colors = torch.rand(n, 3, generator=g)
    W, H = 96, 64
    Ks = torch.tensor([[80.0, 0.0, W / 2], [0.0, 80.0, H / 2], [0.0, 0.0, 1.0]])[
        None
    ].repeat(2, 1, 1)
    viewmats = torch.eye(4)[None].repeat(2, 1, 1)
    viewmats[1, :3, 3] = torch.tensor([0.3, -0.2, 0.1])
    out = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        Ks=Ks,
        viewmats=viewmats,
    )
    out = {k: v.to(device) for k, v in out.items()}
    out.update(width=W, height=H)
    return out


def _render_synthetic(s, viewmats, viewmats_rs=None, **camera):
    kwargs = dict(camera)
    Ks = kwargs.pop("Ks", s["Ks"])
    if viewmats_rs is not None:
        kwargs.update(
            rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
            viewmats_rs=viewmats_rs,
        )
    render, _, _ = gsplat.rasterization(
        s["means"],
        s["quats"],
        s["scales"],
        s["opacities"],
        s["colors"],
        viewmats,
        Ks,
        s["width"],
        s["height"],
        with_ut=True,
        with_eval3d=True,
        packed=False,
        **kwargs,
    )
    return render


def _check_finite_differences(loss_fn, params, label, h=3e-3, trials=4):
    """Check directional derivatives with central differences.

    Scale error by |grad| * |dir| to avoid near-zero directional denominators.
    """
    analytic = torch.autograd.grad(loss_fn(params), params)
    g_norm = torch.cat([a.flatten() for a in analytic]).norm().item()
    assert g_norm > 0
    torch.manual_seed(4)
    for trial in range(trials):
        dirs = [torch.randn_like(p) for p in params]
        d_norm = torch.cat([d.flatten() for d in dirs]).norm().item()
        expected = sum((a * d).sum() for a, d in zip(analytic, dirs)).item()
        with torch.no_grad():
            numeric = (
                loss_fn([h * d for d in dirs]).item()
                - loss_fn([-h * d for d in dirs]).item()
            ) / (2 * h)
        err = abs(numeric - expected) / (g_norm * d_norm)
        print(
            f"{label} trial {trial}: analytic {expected:.5e}, numeric {numeric:.5e}, err {err:.3e}"
        )
        assert err < 5e-2, (label, expected, numeric)


def _camera_kwargs(camera, s):
    """rasterization() keyword arguments for each camera model."""
    C = s["viewmats"].shape[0]
    if camera == "pinhole":
        return {}
    if camera == "opencv":
        return dict(
            radial_coeffs=torch.tensor(
                [[0.05, -0.01, 0.001, 0.0, 0.0, 0.0]] * C, device=device
            ),
            tangential_coeffs=torch.tensor([[0.001, -0.0005]] * C, device=device),
            thin_prism_coeffs=torch.tensor(
                [[0.0005, 0.0, -0.0003, 0.0]] * C, device=device
            ),
        )
    if camera == "fisheye":
        return dict(
            camera_model="fisheye",
            radial_coeffs=torch.tensor([[0.02, -0.005, 0.001, 0.0]] * C, device=device),
        )
    if camera == "ftheta":
        from gsplat.cuda._wrapper import (
            FThetaCameraDistortionParameters,
            FThetaPolynomialType,
        )

        focal = float(s["Ks"][0, 0, 0])
        return dict(
            camera_model="ftheta",
            ftheta_coeffs=FThetaCameraDistortionParameters(
                reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
                pixeldist_to_angle_poly=(0.0, 1.0 / focal, 0.0, 0.0, 0.0, 0.0),
                angle_to_pixeldist_poly=(0.0, focal, 0.0, 0.0, 0.0, 0.0),
                max_angle=math.radians(85.0),
                linear_cde=(1.0, 0.0, 0.0),
            ),
        )
    if camera == "ortho":
        # Orthographic: focal lengths in pixels per scene unit, rays start on the image plane.
        Ks = s["Ks"].clone()
        Ks[:, 0, 0] = Ks[:, 1, 1] = 20.0
        return dict(camera_model="ortho", Ks=Ks)
    raise ValueError(camera)


@pytest.mark.parametrize("rolling", [False, True])
@pytest.mark.parametrize("camera", ["pinhole", "opencv", "fisheye", "ftheta", "ortho"])
def test_viewmat_grads_finite_differences(synthetic, camera, rolling):
    """Directional derivatives of rasterization() vs central differences."""
    s = synthetic
    kwargs = _camera_kwargs(camera, s)
    vm1_base = _rs_end_pose(s["viewmats"], 0.02) if rolling else None
    with torch.no_grad():
        assert _render_synthetic(s, s["viewmats"], vm1_base, **kwargs).abs().sum() > 0
    torch.manual_seed(3)
    w = torch.randn(2, s["height"], s["width"], 3, device=device)

    def loss_fn(params):
        vm0 = _perturb(s["viewmats"], params[0], params[1])
        vm1 = _perturb(vm1_base, params[2], params[3]) if rolling else None
        return (_render_synthetic(s, vm0, vm1, **kwargs) * w).sum()

    _check_finite_differences(
        loss_fn, _pose_params(2, 2 if rolling else 1), f"{camera} rolling={rolling}"
    )


def test_viewmat_grads_batch_dims(synthetic):
    """A batch of scenes gives each scene's viewmats the gradient it gets on its own."""
    s = synthetic
    torch.manual_seed(8)
    w = torch.randn(2, 2, s["height"], s["width"], 3, device=device)
    shift = torch.tensor([0.1, -0.05, 0.2], device=device)
    scenes = [s["means"], s["means"] + shift]

    def render(means, viewmats, Ks):
        out, _, _ = gsplat.rasterization(
            means,
            *[
                torch.broadcast_to(s[k], means.shape[:-2] + s[k].shape)
                for k in ("quats", "scales", "opacities", "colors")
            ],
            viewmats,
            Ks,
            s["width"],
            s["height"],
            with_ut=True,
            with_eval3d=True,
            packed=False,
        )
        return out

    vm = torch.stack([s["viewmats"]] * 2).requires_grad_(True)  # [2, C, 4, 4]
    out = render(torch.stack(scenes), vm, torch.stack([s["Ks"]] * 2))
    (g_batched,) = torch.autograd.grad((out * w).sum(), (vm,))
    for b in range(2):
        vm_b = s["viewmats"].clone().requires_grad_(True)
        (g_b,) = torch.autograd.grad(
            (render(scenes[b], vm_b, s["Ks"]) * w[b]).sum(), (vm_b,)
        )
        torch.testing.assert_close(g_batched[b], g_b, rtol=1e-4, atol=1e-3)


def test_identical_endpoints_match_global(scene):
    """For identical endpoint poses, their gradients sum to the global gradient."""
    C = scene["viewmats"].shape[0]
    H, W = scene["height"], scene["width"]
    colors = scene["colors"][0]
    torch.manual_seed(5)
    w = torch.randn(C, H, W, 3, device=device)

    def run(rs_type, n_endpoints):
        params = _pose_params(C, n_endpoints)
        vm0 = _perturb(scene["viewmats"], params[0], params[1])
        vm1 = (
            _perturb(scene["viewmats"], params[2], params[3])
            if n_endpoints == 2
            else None
        )
        render, _, _ = gsplat.rasterization(
            scene["means"],
            scene["quats"],
            scene["scales"],
            scene["opacities"],
            colors,
            vm0,
            scene["Ks"],
            W,
            H,
            with_ut=True,
            with_eval3d=True,
            packed=False,
            rolling_shutter=rs_type,
            viewmats_rs=vm1,
        )
        return torch.autograd.grad((render * w).sum(), params)

    g_global = run(RollingShutterType.GLOBAL, 1)
    g_rs = run(RollingShutterType.ROLLING_TOP_TO_BOTTOM, 2)
    summed = (g_rs[0] + g_rs[2], g_rs[1] + g_rs[3])
    rel, cos = _compare(summed, g_global)
    print(f"identical endpoints: rel-L2 {rel:.3e}, cosine {cos:.8f}")
    assert rel < 1e-4 and cos > 0.999999, (rel, cos)


def test_viewmat_grads_only_when_requested(scene):
    """Enabling pose gradients preserves the Gaussian-position gradients."""
    C = scene["viewmats"].shape[0]
    H, W = scene["height"], scene["width"]
    means = scene["means"].clone().requires_grad_(True)
    colors = scene["colors"][0]
    torch.manual_seed(6)
    w = torch.randn(C, H, W, 3, device=device)

    def run(viewmats):
        render, _, _ = gsplat.rasterization(
            means,
            scene["quats"],
            scene["scales"],
            scene["opacities"],
            colors,
            viewmats,
            scene["Ks"],
            W,
            H,
            with_ut=True,
            with_eval3d=True,
            packed=False,
        )
        return (render * w).sum()

    viewmats = scene["viewmats"].clone()
    (g_means_plain,) = torch.autograd.grad(run(viewmats), (means,))

    viewmats_rg = scene["viewmats"].clone().requires_grad_(True)
    g_means, g_viewmats = torch.autograd.grad(run(viewmats_rg), (means, viewmats_rg))
    assert g_viewmats.abs().sum() > 0
    torch.testing.assert_close(g_means, g_means_plain, rtol=1e-5, atol=1e-6)


def test_viewmat_grads_zero_when_nothing_visible(synthetic):
    """A camera that sees no Gaussian gets an exactly zero (defined) gradient."""
    s = synthetic
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))
    viewmats = (flip @ s["viewmats"]).requires_grad_(
        True
    )  # looks away from the Gaussians
    render = _render_synthetic(s, viewmats)
    assert render.abs().max() == 0
    (g,) = torch.autograd.grad(render.sum(), (viewmats,))
    assert torch.equal(g, torch.zeros_like(g))


@pytest.mark.parametrize(
    "renderer_config",
    [gsplat.RendererConfig_MixedBatch(), gsplat.RendererConfig_ParallelBatch()],
    ids=["mixed_batch", "parallel_batch"],
)
def test_viewmat_grads_endpoint_flags(synthetic, renderer_config):
    """Freezing one endpoint preserves the other endpoint's gradient."""
    s = synthetic
    vm1_base = _rs_end_pose(s["viewmats"], 0.02)
    torch.manual_seed(9)
    w = torch.randn(2, s["height"], s["width"], 3, device=device)

    def grads(train0, train1):
        vm0 = s["viewmats"].clone().requires_grad_(train0)
        vm1 = vm1_base.clone().requires_grad_(train1)
        render, _, _ = gsplat.rasterization(
            s["means"],
            s["quats"],
            s["scales"],
            s["opacities"],
            s["colors"],
            vm0,
            s["Ks"],
            s["width"],
            s["height"],
            with_ut=True,
            with_eval3d=True,
            packed=False,
            rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
            viewmats_rs=vm1,
            renderer_config=renderer_config,
        )
        trained = [vm for vm, on in ((vm0, train0), (vm1, train1)) if on]
        return torch.autograd.grad((render * w).sum(), trained)

    g0_both, g1_both = grads(True, True)
    (g0_only,) = grads(True, False)
    (g1_only,) = grads(False, True)
    assert g0_both.abs().sum() > 0 and g1_both.abs().sum() > 0
    torch.testing.assert_close(g0_only, g0_both, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(g1_only, g1_both, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(
    not gsplat.has_camera_wrappers(), reason="Camera wrapper support isn't built in"
)
def test_explicit_rays_route_gradient_to_rays(synthetic):
    """Explicit rays receive gradients without a direct viewmat gradient."""
    from gsplat.cuda._torch_cameras import _BaseCameraModel
    from gsplat.cuda._torch_impl_eval3d import _generate_rays

    s = synthetic
    C, H, W = 2, s["height"], s["width"]
    Ks = s["Ks"]
    camera = _BaseCameraModel.create(
        width=W,
        height=H,
        camera_model="pinhole",
        focal_lengths=Ks[:, [0, 1], [0, 1]],
        principal_points=Ks[:, [0, 1], [2, 2]],
        rs_type=RollingShutterType.GLOBAL,
        lidar_coeffs=None,
    )
    rays = _generate_rays(camera, W, H, s["viewmats"]).reshape(C, H, W, 6)
    rays = rays.detach().requires_grad_(True)
    viewmats = s["viewmats"].clone().requires_grad_(True)
    render, _, _ = gsplat.rasterization(
        s["means"],
        s["quats"],
        s["scales"],
        s["opacities"],
        s["colors"],
        viewmats,
        Ks,
        W,
        H,
        with_ut=True,
        with_eval3d=True,
        packed=False,
        rays=rays,
    )
    g_rays, g_viewmats = torch.autograd.grad(
        render.square().sum(), (rays, viewmats), allow_unused=True
    )
    assert g_rays is not None and g_rays.abs().sum() > 0
    assert g_viewmats is None
