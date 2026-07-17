# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for EnvMapBackground and the BackgroundScene ABC.

Sampling is CUDA-only in the component (no runtime PyTorch fallback), so the
tests split into two groups:

* CPU tests that validate the sampling *semantics* run against the pure-PyTorch
  REFERENCE oracle (``equirect_sample_reference`` / ``cubemap_sample_reference``
  defined inline in this test module — they have no runtime callers and are not
  shipped in ``env_map_sample_ops.py``). These are the same math the CUDA kernels
  mirror; the component itself is never called for sampling on CPU.
* GPU-guarded parity tests compare the CUDA ``.apply`` forward/backward against
  the reference and are skipped when no CUDA device is present.

Non-sampling behavior (compositing formula, gradient tracking, inpaint mask,
state persistence, ...) is exercised on CPU by feeding a synthetic ``bg_rgb``
into the non-sampler helpers (``_activate`` / ``_blend_background``) or by
GPU-guarding the few tests that must call the sampler end-to-end.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.scene import (
    BackgroundScene,
    EnvMapBackground,
    EnvMapBackgroundConfig,
    EnvMapType,
    Scene,
)
from gsplat.scene.kernels.env_map_sample_ops import (
    CubemapEnvMapSampleFunction,
    EquirectEnvMapSampleFunction,
)


# ---------------------------------------------------------------------------
# Pure-PyTorch REFERENCE samplers (test oracle only; NOT used at runtime).
# Moved verbatim from gsplat.scene.kernels.env_map_sample_ops, which had no
# runtime callers for them — only these tests use them, so they live inline here.
# ---------------------------------------------------------------------------


def _dominant_axis_to_face_uv(rays_d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Route ray directions to cube faces and per-face UV coordinates.

    Uses the standard OpenGL cube-map convention. Faces are indexed
    ``0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z`` to match ``textures[0, face]``.

    Args:
        rays_d: World-space directions ``[N, 3]``.

    Returns:
        Tuple ``(face, u, v)`` with ``face`` a ``[N]`` long tensor and ``u``,
        ``v`` ``[N]`` float grid-sample coordinates in ``[-1, 1]``.
    """
    x, y, z = rays_d.unbind(-1)
    ax, ay, az = x.abs(), y.abs(), z.abs()
    n = rays_d.shape[0]
    device, dtype = rays_d.device, rays_d.dtype

    face = torch.zeros(n, dtype=torch.long, device=device)
    sc = torch.zeros(n, dtype=dtype, device=device)
    tc = torch.zeros(n, dtype=dtype, device=device)
    ma = torch.ones(n, dtype=dtype, device=device)

    # Partition into mutually exclusive dominant-axis masks (priority x > y > z).
    x_major = (ax >= ay) & (ax >= az)
    y_major = (~x_major) & (ay >= az)
    z_major = (~x_major) & (~y_major)

    xp = x_major & (x > 0)
    xn = x_major & (x <= 0)
    yp = y_major & (y > 0)
    yn = y_major & (y <= 0)
    zp = z_major & (z > 0)
    zn = z_major & (z <= 0)

    # +X
    face[xp] = 0
    sc[xp], tc[xp], ma[xp] = -z[xp], -y[xp], ax[xp]
    # -X
    face[xn] = 1
    sc[xn], tc[xn], ma[xn] = z[xn], -y[xn], ax[xn]
    # +Y
    face[yp] = 2
    sc[yp], tc[yp], ma[yp] = x[yp], z[yp], ay[yp]
    # -Y
    face[yn] = 3
    sc[yn], tc[yn], ma[yn] = x[yn], -z[yn], ay[yn]
    # +Z
    face[zp] = 4
    sc[zp], tc[zp], ma[zp] = x[zp], -y[zp], az[zp]
    # -Z
    face[zn] = 5
    sc[zn], tc[zn], ma[zn] = -x[zn], -y[zn], az[zn]

    # Grid-sample coordinates in [-1, 1] (face spans the whole texture).
    ma = ma.clamp(min=torch.finfo(dtype).eps)
    u = sc / ma
    v = tc / ma
    return face, u, v


def equirect_sample_reference(rays_d: Tensor, textures: Tensor) -> Tensor:
    """Reference equirectangular sampler → raw pre-activation radiance ``[N, 3]``.

    Differentiable w.r.t. both ``rays_d`` and ``textures`` (unit tests exercise
    the finite pole-direction gradient to ``rays_d``). ``rays_d`` is assumed
    already unit-normalized — normalization is the caller's responsibility and is
    deliberately NOT done here.

    Args:
        rays_d: ``[N, 3]`` directions, assumed unit-normalized.
        textures: ``[1, H, W, 3]`` texture.

    Returns:
        ``[N, 3]`` raw (pre-activation) radiance.
    """
    H, W = textures.shape[1], textures.shape[2]
    x, y, z = rays_d.unbind(-1)
    # Azimuth ∈ (-π, π]. At the exact poles (x = y = 0) the azimuth is
    # undefined and ``atan2(0, 0)`` has a NaN gradient; nudge x by a tiny
    # epsilon there so the gradient to ``rays_d`` stays finite (the value is
    # arbitrary at the pole regardless).
    pole = (x == 0) & (y == 0)
    phi = torch.atan2(y, torch.where(pole, x + 1e-6, x))
    # Clamp strictly inside [-1, 1]: acos' gradient blows up at the poles.
    theta = torch.acos(z.clamp(-1.0 + 1e-6, 1.0 - 1e-6))  # polar ∈ [0, π]

    # Azimuth fraction ∈ [0, 1) around the sphere.
    f = (phi + torch.pi) / (2.0 * torch.pi)
    # Wrap the texture by one column on each side, then map the azimuth so
    # sampling coordinates stay strictly inside the real columns (u never
    # reaches the padded border). This yields correct antimeridian
    # wrap-around; the vertical axis uses "border" for the poles.
    u = (2.0 * f * W + 2.0) / (W + 2.0) - 1.0
    v = theta / torch.pi * 2.0 - 1.0

    tex = textures.permute(0, 3, 1, 2)  # [1, 3, H, W]
    tex = torch.cat([tex[..., -1:], tex, tex[..., :1]], dim=-1)  # [1, 3, H, W+2]
    grid = torch.stack([u, v], dim=-1)[None, None].to(tex.dtype)  # [1, 1, N, 2]
    rgb = F.grid_sample(
        tex,
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return rgb.squeeze(0).squeeze(1).T  # [N, 3]


def cubemap_sample_reference(rays_d: Tensor, textures: Tensor) -> Tensor:
    """Reference cubemap sampler → raw pre-activation radiance ``[N, 3]``.

    Differentiable w.r.t. both ``rays_d`` and ``textures``. ``rays_d`` is assumed
    already unit-normalized — normalization is the caller's responsibility.

    Args:
        rays_d: ``[N, 3]`` directions, assumed unit-normalized.
        textures: ``[1, 6, H, W, 3]`` texture with ``H == W``.

    Returns:
        ``[N, 3]`` raw pre-activation radiance.
    """
    face, u, v = _dominant_axis_to_face_uv(rays_d)
    rgb = torch.zeros(rays_d.shape[0], 3, device=rays_d.device, dtype=textures.dtype)
    for face_idx in range(6):
        # No ``mask.any()`` early-out: it would force a GPU→CPU sync every
        # face. grid_sample tolerates a zero-row grid and the masked
        # assignment below is a no-op when the mask is empty.
        mask = face == face_idx
        grid = torch.stack([u[mask], v[mask]], dim=-1)[None, None].to(
            textures.dtype
        )  # [1, 1, M, 2]
        face_tex = textures[0, face_idx].permute(2, 0, 1)[None]  # [1, 3, H, W]
        sampled = (
            F.grid_sample(
                face_tex,
                grid,
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )
            .squeeze(0)
            .squeeze(1)
            .T
        )  # [M, 3]
        rgb[mask] = sampled
    return rgb


def _rand_dirs(n: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    d = torch.randn(n, 3)
    return d / d.norm(dim=-1, keepdim=True)


def _equirect(width: int = 16, height: int = 8, **kw) -> EnvMapBackground:
    return EnvMapBackground(
        EnvMapBackgroundConfig(
            envmap_type="equirectangular", width=width, height=height, **kw
        )
    )


def _cubemap(width: int = 8, **kw) -> EnvMapBackground:
    return EnvMapBackground(
        EnvMapBackgroundConfig(envmap_type="cubemap", width=width, height=width, **kw)
    )


def _reference_for(bg: EnvMapBackground):
    """Return the reference sampler matching ``bg``'s projection."""
    if bg.envmap_type == EnvMapType.EQUIRECTANGULAR:
        return equirect_sample_reference
    return cubemap_sample_reference


def _sample_reference(bg: EnvMapBackground, rays_d: torch.Tensor) -> torch.Tensor:
    """Sample ``bg``'s texture via the reference oracle (normalizing like the
    component's ``_sample_raw`` does), then apply the radiance activation.

    This mirrors what ``bg.sample`` would compute if it ran the reference — used
    to validate sampling semantics on CPU where the CUDA sampler cannot run.
    """
    normed = rays_d / rays_d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    raw = _reference_for(bg)(normed, bg.textures)
    return bg._activate(raw)


# ---------------------------------------------------------------------------
# Construction / typing
# ---------------------------------------------------------------------------


def test_is_scene_and_background_scene():
    bg = _cubemap()
    assert isinstance(bg, BackgroundScene)
    assert isinstance(bg, Scene)
    assert bg.id == "background"


def test_texture_shapes_and_init_value():
    eq = _equirect(width=16, height=8)
    assert eq.textures.shape == (1, 8, 16, 3)
    assert eq.texture_grads.shape == (8, 16)
    assert torch.allclose(eq.textures, torch.full_like(eq.textures, 0.5))

    cm = _cubemap(width=8)
    assert cm.textures.shape == (1, 6, 8, 8, 3)
    assert cm.texture_grads.shape == (6, 8, 8)
    assert torch.allclose(cm.textures, torch.full_like(cm.textures, 0.5))


def test_cubemap_requires_square():
    with pytest.raises(ValueError, match="width == height"):
        EnvMapBackground(
            EnvMapBackgroundConfig(envmap_type="cubemap", width=8, height=4)
        )


# ---------------------------------------------------------------------------
# No runtime fallback contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_sample_on_cpu_raises_no_fallback(factory):
    # Sampling is CUDA-only: CPU tensors must raise (no PyTorch fallback). This
    # locks the "NO runtime fallback" design decision.
    bg = factory()
    with pytest.raises((ValueError, TypeError)):
        bg.sample(_rand_dirs(8))


# ---------------------------------------------------------------------------
# Subclass inheritance seam
# ---------------------------------------------------------------------------


def test_per_projection_sampler_methods_exist():
    # NRE's SkyEnvMapBackground subclasses EnvMapBackground and calls the
    # inherited `_sample_cubemap` directly (and overrides `_sample_equirect`).
    # These overridable per-projection methods are a public inheritance seam;
    # removing them breaks that subclass, so lock their presence here (CPU-safe).
    for name in ("_sample_equirect", "_sample_cubemap"):
        assert callable(getattr(EnvMapBackground, name, None)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_subclass_overrides_one_sampler_inherits_other():
    # Reproduce NRE's pattern: override `_sample_equirect`, inherit `_sample_cubemap`.
    sentinel = {}

    class _Sub(EnvMapBackground):
        def _sample_equirect(self, rays_d):  # type: ignore[override]
            sentinel["equirect"] = True
            return torch.zeros(rays_d.shape[0], 3, device=rays_d.device)

    dirs = _rand_dirs(64).cuda()

    cube = _Sub(EnvMapBackgroundConfig(envmap_type="cubemap", width=8, height=8)).to(
        "cuda"
    )
    out_cube = cube.sample(dirs)  # inherited CUDA `_sample_cubemap` — must not raise
    assert out_cube.shape == (64, 3)

    eq = _Sub(
        EnvMapBackgroundConfig(envmap_type="equirectangular", width=16, height=8)
    ).to("cuda")
    _ = eq.sample(dirs)  # routes through the override
    assert sentinel.get("equirect") is True


# ---------------------------------------------------------------------------
# Sampling semantics (CPU, against the REFERENCE oracle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_reference_sample_shape_and_non_negativity(factory):
    bg = factory()
    dirs = _rand_dirs(128)
    out = _sample_reference(bg, dirs)
    assert out.shape == (128, 3)
    assert (out >= 0).all()


@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_reference_neutral_texture_returns_half(factory):
    bg = factory()
    out = _sample_reference(bg, _rand_dirs(50))
    # A constant 0.5 texture must sample to 0.5 everywhere (relu keeps it).
    torch.testing.assert_close(out, torch.full_like(out, 0.5))


def test_saturate_radiance_clamps():
    bg = _cubemap(saturate_radiance=True)
    with torch.no_grad():
        bg.textures.add_(5.0)  # push way above 1
    out = _sample_reference(bg, _rand_dirs(32))
    assert out.max() <= 1.0 + 1e-6
    assert out.min() >= 0.0


def test_relu_activation_allows_hdr():
    bg = _cubemap(saturate_radiance=False)
    with torch.no_grad():
        bg.textures.add_(5.0)
    out = _sample_reference(bg, _rand_dirs(32))
    assert out.max() > 1.0


def test_relu_activation_clips_negative():
    bg = _cubemap(saturate_radiance=False)
    with torch.no_grad():
        bg.textures.sub_(5.0)  # negative texture
    out = _sample_reference(bg, _rand_dirs(32))
    assert (out >= 0).all()


def test_cubemap_face_routing_maps_axes_to_expected_faces():
    # Paint each of the 6 faces a distinct constant color, then sample the 6
    # canonical axis directions. This locks in the dominant-axis -> face routing
    # of `_dominant_axis_to_face_uv`, whose documented face order is
    # 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
    bg = _cubemap(width=8)
    face_colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # face 0: +X
            [0.0, 1.0, 0.0],  # face 1: -X
            [0.0, 0.0, 1.0],  # face 2: +Y
            [1.0, 1.0, 0.0],  # face 3: -Y
            [1.0, 0.0, 1.0],  # face 4: +Z
            [0.0, 1.0, 1.0],  # face 5: -Z
        ]
    )
    with torch.no_grad():
        for f in range(6):
            bg.textures[0, f] = face_colors[f]

    # Directions ordered to match the face they should route to (+X..-Z).
    dirs = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # -> face 0 (+X)
            [-1.0, 0.0, 0.0],  # -> face 1 (-X)
            [0.0, 1.0, 0.0],  # -> face 2 (+Y)
            [0.0, -1.0, 0.0],  # -> face 3 (-Y)
            [0.0, 0.0, 1.0],  # -> face 4 (+Z)
            [0.0, 0.0, -1.0],  # -> face 5 (-Z)
        ]
    )
    out = _sample_reference(bg, dirs)
    torch.testing.assert_close(out, face_colors)


def test_dominant_axis_routes_axes_to_expected_faces():
    # Directly probe the routing helper's face indices for the canonical axes.
    dirs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    face, _u, _v = _dominant_axis_to_face_uv(dirs)
    assert face.tolist() == [0, 1, 2, 3, 4, 5]


def test_dominant_axis_off_axis_uv_matches_face_formulas():
    # On-axis probes leave u = v = 0, so a sign flip / mirror / axis-swap within
    # a face goes uncaught. Probe every face with ASYMMETRIC, opposite-signed
    # perturbations on its two non-dominant axes and assert exact (face, u, v).
    # Expected u, v are computed directly from the helper's own formulas
    # (u = sc / ma, v = tc / ma; ma = |dominant axis|):
    #   +X(0): sc=-z, tc=-y   -X(1): sc= z, tc=-y
    #   +Y(2): sc= x, tc= z   -Y(3): sc= x, tc=-z
    #   +Z(4): sc= x, tc=-y   -Z(5): sc=-x, tc=-y
    dirs = torch.tensor(
        [
            [4.0, 1.0, -2.0],  # +X(0): u=-z/|x|=0.5,   v=-y/|x|=-0.25
            [-4.0, 1.0, -2.0],  # -X(1): u= z/|x|=-0.5,  v=-y/|x|=-0.25
            [1.0, 4.0, -2.0],  # +Y(2): u= x/|y|=0.25,  v= z/|y|=-0.5
            [1.0, -4.0, -2.0],  # -Y(3): u= x/|y|=0.25,  v=-z/|y|=0.5
            [1.0, -2.0, 4.0],  # +Z(4): u= x/|z|=0.25,  v=-y/|z|=0.5
            [1.0, -2.0, -4.0],  # -Z(5): u=-x/|z|=-0.25, v=-y/|z|=0.5
            [3.0, 3.0, 1.0],  # tie |x|==|y|>|z| -> +X (x>y): u=-z/|x|, v=-y/|x|
            [1.0, 3.0, 3.0],  # tie |y|==|z|>|x| -> +Y (y>z): u= x/|y|, v= z/|y|
        ]
    )
    expected_face = [0, 1, 2, 3, 4, 5, 0, 2]
    expected_u = torch.tensor(
        [0.5, -0.5, 0.25, 0.25, 0.25, -0.25, -1.0 / 3.0, 1.0 / 3.0]
    )
    expected_v = torch.tensor([-0.25, -0.25, -0.5, 0.5, 0.5, 0.5, -1.0, 1.0])

    face, u, v = _dominant_axis_to_face_uv(dirs)
    assert face.tolist() == expected_face
    torch.testing.assert_close(u, expected_u)
    torch.testing.assert_close(v, expected_v)


def test_equirect_direction_maps_to_expected_texel():
    # Pin the equirectangular direction -> texel mapping (pixel-center + sign
    # semantics). Paint a per-column gradient in R (R = column index) and a
    # per-row gradient in a second texture, then probe canonical directions.
    # Mirrors test_cubemap_face_routing_maps_axes_to_expected_faces.
    H, W = 8, 16
    col_bg = _equirect(width=W, height=H)
    with torch.no_grad():
        for c in range(W):
            col_bg.textures[0, :, c, 0] = float(c)

    def probe_r(vec):
        return _sample_reference(col_bg, torch.tensor([vec]))[0, 0].item()

    # Azimuth: +x -> phi=0 (texture center, W/2 - 0.5); +y -> phi=pi/2 (3W/4);
    # -y -> phi=-pi/2 (W/4). Values land on bilinear midpoints between columns.
    assert probe_r([1.0, 0.0, 0.0]) == pytest.approx(W / 2 - 0.5, abs=1e-3)
    assert probe_r([0.0, 1.0, 0.0]) == pytest.approx(3 * W / 4 - 0.5, abs=1e-3)
    assert probe_r([0.0, -1.0, 0.0]) == pytest.approx(W / 4 - 0.5, abs=1e-3)

    row_bg = _equirect(width=W, height=H)
    with torch.no_grad():
        for r in range(H):
            row_bg.textures[0, r, :, 0] = float(r)

    # Polar: z=+1 -> top row (r=0); z=-1 -> bottom row (r=H-1).
    top = _sample_reference(row_bg, torch.tensor([[0.0, 0.0, 1.0]]))[0, 0].item()
    bottom = _sample_reference(row_bg, torch.tensor([[0.0, 0.0, -1.0]]))[0, 0].item()
    assert top == pytest.approx(0.0, abs=1e-2)
    assert bottom == pytest.approx(H - 1, abs=1e-2)


def test_equirect_sample_is_scale_invariant():
    # The component normalizes rays_d before sampling, so a direction and any
    # positive scaling of it must sample to the same color. The reference does
    # NOT normalize internally, so mirror the component's normalization here.
    torch.manual_seed(3)
    bg = _equirect(width=32, height=16)
    with torch.no_grad():
        bg.textures.copy_(torch.rand_like(bg.textures))
    dirs = _rand_dirs(50)
    torch.testing.assert_close(
        _sample_reference(bg, dirs), _sample_reference(bg, 2.0 * dirs)
    )


def test_equirect_antimeridian_wrap_is_continuous():
    # Two nearly identical directions straddling the +/- pi azimuth seam must
    # sample to nearly the same color for a smooth texture.
    torch.manual_seed(1)
    bg = _equirect(width=32, height=16)
    with torch.no_grad():
        bg.textures.copy_(torch.rand_like(bg.textures))
    eps = 1e-3
    # phi just above -pi and just below +pi point to almost the same place.
    left = torch.tensor(
        [
            [
                torch.cos(torch.tensor(-torch.pi + eps)),
                torch.sin(torch.tensor(-torch.pi + eps)),
                0.0,
            ]
        ]
    )
    right = torch.tensor(
        [
            [
                torch.cos(torch.tensor(torch.pi - eps)),
                torch.sin(torch.tensor(torch.pi - eps)),
                0.0,
            ]
        ]
    )
    a = _sample_reference(bg, left)
    b = _sample_reference(bg, right)
    assert torch.allclose(a, b, atol=5e-2)


def test_pole_direction_gradient_to_rays_d_is_finite():
    # acos gradient blows up at z=+/-1; the reference's interior clamp keeps it
    # finite. The reference is differentiable w.r.t. rays_d (unlike the CUDA
    # kernel, whose grad to rays_d is None), so this guarantee is exercised here.
    bg = _equirect(width=16, height=8)
    with torch.no_grad():
        bg.textures.copy_(torch.rand_like(bg.textures))
    dirs = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], requires_grad=True)
    normed = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    equirect_sample_reference(normed, bg.textures).sum().backward()
    assert dirs.grad is not None
    assert torch.isfinite(dirs.grad).all()


@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_reference_gradient_flows_to_textures(factory):
    bg = factory()
    out = _sample_reference(bg, _rand_dirs(64))
    out.sum().backward()
    assert bg.textures.grad is not None
    assert bg.textures.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# composite() — non-sampling formula, exercised via _blend_background with a
# synthetic bg_rgb so no CUDA sampler is required.
# ---------------------------------------------------------------------------


def test_composite_srgb_formula():
    bg = _cubemap()
    gaussian_rgb = torch.rand(40, 3)
    opacity = torch.rand(40)
    bg_rgb = torch.rand(40, 3)  # synthetic sampled background

    out = bg._blend_background(gaussian_rgb, bg_rgb, opacity)
    expected = gaussian_rgb + bg_rgb * (1.0 - opacity.unsqueeze(-1))
    torch.testing.assert_close(out, expected)


def test_opaque_pixels_ignore_background():
    bg = _cubemap()
    gaussian_rgb = torch.rand(10, 3)
    opacity = torch.ones(10)
    bg_rgb = torch.rand(10, 3)  # synthetic sampled background
    out = bg._blend_background(gaussian_rgb, bg_rgb, opacity)
    torch.testing.assert_close(out, gaussian_rgb)


def test_composite_accepts_opacity_column_vector():
    # Regression guard: opacity shaped [N, 1] (rasterizer alphas) must give the
    # same result as [N] and must NOT broadcast into an [N, N, 3] tensor.
    bg = _cubemap()
    gaussian_rgb = torch.rand(24, 3)
    opacity = torch.rand(24)
    bg_rgb = torch.rand(24, 3)  # synthetic sampled background

    out_flat = bg._blend_background(gaussian_rgb, bg_rgb, opacity)
    out_col = bg._blend_background(gaussian_rgb, bg_rgb, opacity.reshape(-1, 1))
    assert out_col.shape == (24, 3)
    torch.testing.assert_close(out_col, out_flat)


# ---------------------------------------------------------------------------
# activation helper (_activate)
# ---------------------------------------------------------------------------


def test_activate_relu_keeps_hdr_and_clips_negative():
    bg = _cubemap(saturate_radiance=False)
    raw = torch.tensor([[-1.0, 0.5, 3.0]])
    out = bg._activate(raw)
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.5, 3.0]]))


def test_activate_saturate_clamps_to_unit():
    bg = _cubemap(saturate_radiance=True)
    raw = torch.tensor([[-1.0, 0.5, 3.0]])
    out = bg._activate(raw)
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.5, 1.0]]))


# ---------------------------------------------------------------------------
# CUDA sampler parity (GPU-only; skipped without a device)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_cuda_forward_matches_reference(factory):
    torch.manual_seed(11)
    bg = factory().to("cuda")
    with torch.no_grad():
        bg.textures.copy_(torch.rand_like(bg.textures))
    rays = _rand_dirs(128).cuda()
    normed = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    cuda_raw = bg._sample_raw(rays)
    ref = _reference_for(bg)(normed, bg.textures)
    torch.testing.assert_close(cuda_raw, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_cuda_backward_matches_reference(factory):
    torch.manual_seed(13)
    rays = _rand_dirs(128).cuda()
    normed = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    grad_seed = torch.rand(128, 3, device="cuda")

    # CUDA path grad_textures (through the autograd Function).
    bg_cuda = factory().to("cuda")
    with torch.no_grad():
        bg_cuda.textures.copy_(torch.rand_like(bg_cuda.textures))
    raw_cuda = bg_cuda._sample_raw(rays)
    (raw_cuda * grad_seed).sum().backward()
    grad_cuda = bg_cuda.textures.grad

    # Reference grad_textures (same texture values, pure-PyTorch reference).
    bg_ref = factory().to("cuda")
    with torch.no_grad():
        bg_ref.textures.copy_(bg_cuda.textures.detach())
    raw_ref = _reference_for(bg_ref)(normed, bg_ref.textures)
    (raw_ref * grad_seed).sum().backward()
    grad_ref = bg_ref.textures.grad

    torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
# The kernel is float32-only, so gradcheck runs in float32 and PyTorch warns that
# the input is not double precision. That warning is expected here (sampling is
# exactly linear in ``textures``, so the float32 finite-difference Jacobian is
# exact); suppress it so the repo-wide ``filterwarnings = error`` does not turn it
# into a failure.
@pytest.mark.filterwarnings(
    "ignore:Input #.* requires gradient and is not a double precision"
)
@pytest.mark.parametrize(
    "fn,factory",
    [
        (EquirectEnvMapSampleFunction, _equirect),
        (CubemapEnvMapSampleFunction, _cubemap),
    ],
)
def test_cuda_gradcheck_textures(fn, factory):
    # gradcheck on the autograd.Function backward w.r.t. textures. Sampling is
    # exactly linear in ``textures`` (a fixed bilinear combination), so the
    # finite-difference Jacobian is exact and a float32 gradcheck is stable with
    # a coarse eps. The kernel is float32-only (double is rejected by the C++
    # layer), hence check_grad_dtypes=False; the backward atomicAdd scatter is
    # accumulation-order nondeterministic, hence a small nondet_tol.
    torch.manual_seed(7)
    bg = factory(width=4).to("cuda")
    rays = _rand_dirs(8).cuda()
    normed = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    textures = torch.rand_like(bg.textures).double().float().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda t: fn.apply(normed, t),
        (textures,),
        eps=1e-2,
        atol=1e-3,
        rtol=1e-3,
        check_grad_dtypes=False,
        nondet_tol=1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "fn,factory",
    [
        (EquirectEnvMapSampleFunction, _equirect),
        (CubemapEnvMapSampleFunction, _cubemap),
    ],
)
def test_cuda_function_grad_rays_d_is_none(fn, factory):
    # The CUDA kernel differentiates w.r.t. textures only; rays_d grad is None.
    bg = factory().to("cuda")
    rays = _rand_dirs(32).cuda().requires_grad_(True)
    out = fn.apply(rays, bg.textures)
    out.sum().backward()
    assert rays.grad is None
    assert bg.textures.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_cuda_matches_reference_on_boundary_dirs(factory):
    # The random-direction parity tests use `_rand_dirs`, which never lands on a
    # pole (x==y==0), the +/-pi azimuth seam, or a canonical axis, so a bug in the
    # kernel's seam wrap `((px-1)%W+W)%W` or its 1e-6 pole nudge would pass every
    # GPU test (the seam/pole/face-routing tests only exercise the pure-torch
    # oracle, never the kernel). Pin those boundaries kernel-vs-oracle here.
    # Looser tol than the random test: boundary atan2/acos amplify fp error, and
    # the kernel's pole nudge differs slightly from the oracle's interior clamp.
    torch.manual_seed(17)
    bg = factory().to("cuda")
    with torch.no_grad():
        bg.textures.copy_(torch.rand_like(bg.textures))
    dirs = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # +Z pole (equirect) / +Z face center (cubemap)
            [0.0, 0.0, -1.0],  # -Z pole / -Z face center
            [-1.0, 1e-7, 0.0],  # just above the -pi azimuth seam
            [-1.0, -1e-7, 0.0],  # just below the +pi azimuth seam
            [1.0, 0.0, 0.0],  # +X face center
            [-1.0, 0.0, 0.0],  # -X face center
            [0.0, 1.0, 0.0],  # +Y face center
            [0.0, -1.0, 0.0],  # -Y face center
        ],
        device="cuda",
    )
    normed = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cuda_raw = bg._sample_raw(normed)
    ref = _reference_for(bg)(normed, bg.textures)
    torch.testing.assert_close(cuda_raw, ref, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_cuda_empty_batch_returns_empty_and_zero_grad(factory):
    # A tile with no background rays yields n==0; all four launchers early-return.
    # No existing GPU test hits that path (smallest N is 8). Verify the empty
    # forward is well-shaped and that the pre-zeroed grad_textures stays zero (a
    # regressed early-return that left grad uninitialized would be caught here).
    bg = factory().to("cuda")
    bg.textures.requires_grad_(True)
    raw = bg._sample_raw(torch.empty(0, 3, device="cuda"))
    assert raw.shape == (0, 3)
    raw.sum().backward()
    assert bg.textures.grad is not None
    assert torch.count_nonzero(bg.textures.grad) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_cuda_single_ray_matches_reference(factory):
    # N==1 exercises the smallest non-empty launch (one block, one thread).
    torch.manual_seed(19)
    bg = factory().to("cuda")
    with torch.no_grad():
        bg.textures.copy_(torch.rand_like(bg.textures))
    rays = _rand_dirs(1).cuda()
    normed = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cuda_raw = bg._sample_raw(rays)
    ref = _reference_for(bg)(normed, bg.textures)
    torch.testing.assert_close(cuda_raw, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_cuda_backward_accepts_noncontiguous_grad(factory):
    # The ops wrapper `.contiguous()`-normalizes rays_d/textures/grad_out, but no
    # test feeds a strided input. Pass a transposed (non-contiguous) grad_seed so
    # the backward's grad_out `.contiguous()` is exercised: a dropped
    # `.contiguous()` would either read a strided raw pointer or trip the C++
    # CHECK_CONTIGUOUS, both caught by this parity check.
    torch.manual_seed(23)
    rays = _rand_dirs(128).cuda()
    normed = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    grad_seed = torch.rand(3, 128, device="cuda").T  # non-contiguous [128, 3]
    assert not grad_seed.is_contiguous()

    bg_cuda = factory().to("cuda")
    with torch.no_grad():
        bg_cuda.textures.copy_(torch.rand_like(bg_cuda.textures))
    raw_cuda = bg_cuda._sample_raw(rays)
    (raw_cuda * grad_seed).sum().backward()
    grad_cuda = bg_cuda.textures.grad

    bg_ref = factory().to("cuda")
    with torch.no_grad():
        bg_ref.textures.copy_(bg_cuda.textures.detach())
    raw_ref = _reference_for(bg_ref)(normed, bg_ref.textures)
    (raw_ref * grad_seed.contiguous()).sum().backward()
    grad_ref = bg_ref.textures.grad

    torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Gradients + gradient tracking (GPU-only: these call the CUDA sampler)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("factory", [_equirect, _cubemap])
def test_gradient_flows_through_sample(factory):
    bg = factory().to("cuda")
    out = bg.sample(_rand_dirs(64).cuda())
    out.sum().backward()
    assert bg.textures.grad is not None
    assert bg.textures.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gradient_flows_through_composite():
    bg = _cubemap().to("cuda")
    dirs = _rand_dirs(64).cuda()
    gaussian_rgb = torch.rand(64, 3, device="cuda")
    opacity = torch.rand(64, device="cuda")
    out = bg.composite(gaussian_rgb, opacity, dirs, is_training=False)
    out.sum().backward()
    assert bg.textures.grad is not None
    assert bg.textures.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gradient_tracking_updates_texture_grads():
    # With warm-up disabled (min_grad_updates = -1) and training on, composite
    # should populate texture_grads and advance the step counter.
    bg = _cubemap(min_grad_updates=-1).to("cuda")
    dirs = _rand_dirs(64).cuda()
    gaussian_rgb = torch.rand(64, 3, device="cuda")
    opacity = torch.rand(64, device="cuda")
    assert bg.n_grad_updates == 0
    before = bg.texture_grads.clone()
    bg.composite(gaussian_rgb, opacity, dirs, is_training=True)
    assert bg.n_grad_updates == 1
    assert bg.texture_grads.shape == (6, 8, 8)
    assert (bg.texture_grads >= before).all()
    assert bg.texture_grads.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backward_after_gradient_tracking():
    # With tracking active (warm-up disabled), composite runs an internal
    # autograd.grad(retain_graph=True). A subsequent real backward pass on the
    # composited output must still populate textures.grad (retain_graph kept the
    # graph alive).
    bg = _cubemap(min_grad_updates=-1).to("cuda")
    assert bg.textures.requires_grad
    dirs = _rand_dirs(48).cuda()
    gaussian_rgb = torch.rand(48, 3, device="cuda")
    opacity = torch.rand(48, device="cuda")

    out = bg.composite(gaussian_rgb, opacity, dirs, is_training=True)
    # Tracking happened on this step.
    assert bg.n_grad_updates == 1
    assert bg.texture_grads.abs().sum() > 0

    out.sum().backward()
    assert bg.textures.grad is not None
    assert torch.isfinite(bg.textures.grad).all()
    assert bg.textures.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gradient_tracking_respects_warmup():
    bg = _cubemap(min_grad_updates=1000).to("cuda")
    dirs = _rand_dirs(16).cuda()
    gaussian_rgb = torch.rand(16, 3, device="cuda")
    opacity = torch.rand(16, device="cuda")
    bg.composite(gaussian_rgb, opacity, dirs, is_training=True)
    # Counter advances but no tracking happens during warm-up.
    assert bg.n_grad_updates == 1
    assert bg.texture_grads.abs().sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gradient_tracking_skipped_when_not_training():
    bg = _cubemap(min_grad_updates=-1).to("cuda")
    dirs = _rand_dirs(16).cuda()
    gaussian_rgb = torch.rand(16, 3, device="cuda")
    opacity = torch.rand(16, device="cuda")
    bg.composite(gaussian_rgb, opacity, dirs, is_training=False)
    assert bg.n_grad_updates == 0
    assert bg.texture_grads.abs().sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_past_warmup_composite_under_no_grad_is_safe():
    # A past-warm-up instance must composite under torch.no_grad() without
    # raising (the guard requires an active graph) and without mutating state.
    bg = _cubemap(min_grad_updates=5).to("cuda")
    bg.n_grad_updates = 10  # well past warm-up
    dirs = _rand_dirs(16).cuda()
    gaussian_rgb = torch.rand(16, 3, device="cuda")
    opacity = torch.rand(16, device="cuda")
    grads_before = bg.texture_grads.clone()

    with torch.no_grad():
        # Default args (is_training defaults to False).
        out = bg.composite(gaussian_rgb, opacity, dirs)
        # Even an explicit is_training=True must not crash under no_grad.
        out_train = bg.composite(gaussian_rgb, opacity, dirs, is_training=True)

    assert out.shape == (16, 3)
    assert out_train.shape == (16, 3)
    assert bg.n_grad_updates == 10  # unchanged
    assert torch.equal(bg.texture_grads, grads_before)  # unmutated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gradient_tracking_fires_at_warmup_boundary():
    # Tracking fires exactly when n_grad_updates >= min_grad_updates. Probe the
    # boundary directly with a small warm-up.
    min_updates = 3
    dirs = _rand_dirs(16).cuda()
    gaussian_rgb = torch.rand(16, 3, device="cuda")
    opacity = torch.rand(16, device="cuda")

    # One step below the threshold: no tracking.
    bg = _cubemap(min_grad_updates=min_updates).to("cuda")
    bg.n_grad_updates = min_updates - 1
    bg.composite(gaussian_rgb, opacity, dirs, is_training=True)
    assert bg.n_grad_updates == min_updates
    assert bg.texture_grads.abs().sum() == 0

    # Exactly at the threshold: tracking fires.
    bg = _cubemap(min_grad_updates=min_updates).to("cuda")
    bg.n_grad_updates = min_updates
    bg.composite(gaussian_rgb, opacity, dirs, is_training=True)
    assert bg.n_grad_updates == min_updates + 1
    assert bg.texture_grads.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grad_track_interval_subsamples_tracking():
    # With interval=3 and warm-up disabled, tracking runs only on counts that
    # are multiples of the interval; the counter still advances every call so
    # the gate stays rank-uniform.
    dirs = _rand_dirs(16).cuda()
    gaussian_rgb = torch.rand(16, 3, device="cuda")
    opacity = torch.rand(16, device="cuda")
    bg = _cubemap(min_grad_updates=-1, grad_track_interval=3).to("cuda")

    fired = []
    for _ in range(6):
        counter = bg.n_grad_updates
        # Zero first: the running-max would otherwise hide a repeat firing that
        # produces the identical magnitude map for identical inputs.
        bg.texture_grads.zero_()
        bg.composite(gaussian_rgb, opacity, dirs, is_training=True)
        if bg.texture_grads.abs().sum() > 0:
            fired.append(counter)
    # Counter advanced every call; tracking only on multiples of 3 (0, 3).
    assert bg.n_grad_updates == 6
    assert fired == [0, 3]


# ---------------------------------------------------------------------------
# parameters() / put / get
# ---------------------------------------------------------------------------


def test_parameters_returns_texture():
    bg = _cubemap()
    params = bg.parameters()
    assert params == [bg.textures]
    assert isinstance(params[0], torch.nn.Parameter)


def test_put_get_named_attr():
    bg = _cubemap()
    t = torch.arange(3.0)
    bg.put("scratch", t)
    assert bg.get("scratch") is t


def test_put_rejects_ctx():
    bg = _cubemap()
    with pytest.raises(ValueError, match="transform context"):
        bg.put("x", torch.zeros(1), ctx={"poses": torch.zeros(1)})


# ---------------------------------------------------------------------------
# inpaint_mask()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,shape",
    [
        (lambda **kw: _equirect(16, 8, **kw), (8, 16)),
        (lambda **kw: _cubemap(8, **kw), (6, 8, 8)),
    ],
)
def test_inpaint_mask_shape(factory, shape):
    bg = factory()
    mask = bg.inpaint_mask()
    assert mask.shape == shape
    assert mask.dtype == torch.bool


def test_inpaint_mask_all_true_for_unobserved():
    # Fresh scene: texture_grads all zero < threshold -> everything unobserved.
    bg = _cubemap(8, inpaint_threshold=0.05)
    assert bg.inpaint_mask().all()


def test_inpaint_mask_dilation_marks_neighbours():
    bg = _equirect(16, 8, inpaint_threshold=0.05, inpaint_kernel_size=3)
    # Observe everything strongly, then knock a single texel below threshold.
    bg.texture_grads = torch.full((8, 16), 1.0)
    bg.texture_grads[4, 8] = 0.0
    mask = bg.inpaint_mask()
    # The low texel and its dilated neighbourhood are flagged.
    assert mask[4, 8]
    assert mask[3, 8] and mask[5, 8] and mask[4, 7] and mask[4, 9]
    # Far away stays observed.
    assert not mask[0, 0]


def test_inpaint_mask_even_kernel_dilation_is_symmetric():
    # The default kernel size (10) is even; it is rounded up to the next odd so
    # an interior single texel dilates equally on all sides (centered).
    bg = _equirect(41, 41, inpaint_threshold=0.05, inpaint_kernel_size=10)
    bg.texture_grads = torch.full((41, 41), 1.0)
    bg.texture_grads[20, 20] = 0.0  # single unobserved texel, well interior
    mask = bg.inpaint_mask()

    rows = torch.where(mask[:, 20])[0]
    cols = torch.where(mask[20, :])[0]
    # Rounded kernel is 11 -> half-width 5, symmetric about the hot texel.
    assert rows.min().item() == 20 - 5 and rows.max().item() == 20 + 5
    assert cols.min().item() == 20 - 5 and cols.max().item() == 20 + 5


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [lambda: _equirect(16, 8), lambda: _cubemap(8)])
def test_state_dict_round_trip(factory):
    bg = factory()
    with torch.no_grad():
        bg.textures.copy_(torch.rand_like(bg.textures))
        bg.texture_grads = torch.rand_like(bg.texture_grads)
    bg.n_grad_updates = 42
    bg.grad_track_interval = 4  # non-default, must round-trip

    buffer = io.BytesIO()
    torch.save(bg.state_dict(), buffer)
    buffer.seek(0)
    restored = EnvMapBackground.from_state_dict(torch.load(buffer, map_location="cpu"))

    assert restored.id == bg.id
    assert restored.envmap_type == bg.envmap_type
    assert restored.width == bg.width
    assert restored.height == bg.height
    assert restored.saturate_radiance == bg.saturate_radiance
    assert restored.min_grad_updates == bg.min_grad_updates
    assert restored.grad_track_interval == bg.grad_track_interval
    assert restored.should_inpaint == bg.should_inpaint
    assert restored.inpaint_threshold == bg.inpaint_threshold
    assert restored.inpaint_kernel_size == bg.inpaint_kernel_size
    assert restored.n_grad_updates == 42
    assert isinstance(restored.textures, torch.nn.Parameter)
    torch.testing.assert_close(restored.textures, bg.textures)
    torch.testing.assert_close(restored.texture_grads, bg.texture_grads)


def test_state_dict_preserves_requires_grad_false():
    bg = _cubemap()
    bg.textures.requires_grad_(False)
    restored = EnvMapBackground.from_state_dict(bg.state_dict())
    assert restored.textures.requires_grad is False


def test_from_state_dict_needs_no_constructor_args():
    bg = _equirect(16, 8)
    restored = EnvMapBackground.from_state_dict(bg.state_dict())
    # Restore succeeds from the checkpoint alone and reconstructs the texture.
    assert restored.textures.shape == bg.textures.shape
    torch.testing.assert_close(restored.textures, bg.textures)


def test_from_state_dict_tolerates_missing_n_grad_updates():
    # Older checkpoints may lack "n_grad_updates"; restore must default to 0
    # rather than raising KeyError.
    bg = _cubemap(8)
    state = bg.state_dict()
    del state["n_grad_updates"]
    restored = EnvMapBackground.from_state_dict(state)
    assert restored.n_grad_updates == 0


def test_from_state_dict_tolerates_missing_grad_track_interval():
    # Pre-existing checkpoints predate grad_track_interval; restore must default
    # to 1 (every-step tracking) rather than raising KeyError.
    bg = _cubemap(8)
    state = bg.state_dict()
    del state["grad_track_interval"]
    restored = EnvMapBackground.from_state_dict(state)
    assert restored.grad_track_interval == 1


def test_grad_track_interval_defaults_and_clamps():
    # Default is 1 (unchanged every-step behavior); non-positive values clamp to
    # 1 so the modulo gate in _track_gradients never divides by zero.
    assert _cubemap(8).grad_track_interval == 1
    assert _cubemap(8, grad_track_interval=4).grad_track_interval == 4
    assert _cubemap(8, grad_track_interval=0).grad_track_interval == 1
    assert _cubemap(8, grad_track_interval=-3).grad_track_interval == 1


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------


def test_constructor_device_places_tensors():
    # ``device`` is a runtime placement argument on the constructor, not a
    # serialized config field.
    bg = EnvMapBackground(
        EnvMapBackgroundConfig(envmap_type="cubemap", width=8, height=8),
        device="cpu",
    )
    assert bg.textures.device.type == "cpu"
    assert bg.texture_grads.device.type == "cpu"


def test_to_reassigns_parameter_and_keeps_grads_consistent():
    bg = _cubemap(8)
    old_requires_grad = bg.textures.requires_grad
    bg.to("cpu")
    # `to` reassigns the Parameter and preserves requires_grad.
    assert isinstance(bg.textures, torch.nn.Parameter)
    assert bg.textures.requires_grad == old_requires_grad
    assert bg.textures.device.type == "cpu"
    assert bg.texture_grads.device.type == "cpu"
    # textures and texture_grads stay co-located.
    assert bg.textures.device == bg.texture_grads.device


def test_to_preserves_parameter_identity_for_optimizer():
    # `to()` must mutate in place and preserve the Parameter object identity, so
    # an optimizer built earlier from parameters() keeps updating the live
    # texture across a (same-device) move.
    bg = _cubemap(8)
    opt = torch.optim.SGD(bg.parameters(), lr=0.1)
    bg.to("cpu")
    # The optimizer still references the exact same Parameter object.
    assert opt.param_groups[0]["params"][0] is bg.textures
    # A manual grad + step actually changes the live texture.
    before = bg.textures.detach().clone()
    bg.textures.grad = torch.ones_like(bg.textures)
    opt.step()
    assert not torch.equal(bg.textures.detach(), before)
    torch.testing.assert_close(bg.textures.detach(), before - 0.1)


def test_sync_texture_grad_noop_without_distributed():
    # With distributed not initialized, sync_texture_grad() must be a no-op:
    # return without error and leave textures.grad unchanged (None stays None).
    bg = _cubemap(8)
    assert bg.textures.grad is None
    bg.sync_texture_grad()
    assert bg.textures.grad is None


class _FakeAllReduce:
    """Records ``all_reduce`` calls; acts as identity (single-process stand-in).

    A real all-reduce would sum the (already /world_size) shard across ranks;
    with one process the identity is the correct single-rank stand-in and lets
    us assert the div-by-world-size + collective-participation arithmetic
    deterministically without spawning processes.
    """

    def __init__(self) -> None:
        self.ops: list = []

    def __call__(self, tensor, op=None):  # noqa: ANN001
        self.ops.append(op)


def _patch_distributed(monkeypatch, world_size: int, all_reduce) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)


def test_sync_texture_grad_averages_existing_grad(monkeypatch):
    # world_size=2: the in-place div_ halves the grad; the (identity) all_reduce
    # is entered exactly once with ReduceOp.SUM.
    bg = _cubemap(8)
    g = torch.rand_like(bg.textures)
    bg.textures.grad = g.clone()
    ar = _FakeAllReduce()
    _patch_distributed(monkeypatch, 2, ar)
    bg.sync_texture_grad()
    torch.testing.assert_close(bg.textures.grad, g / 2)
    assert ar.ops == [torch.distributed.ReduceOp.SUM]


def test_sync_texture_grad_zeros_participation_when_grad_none(monkeypatch):
    # An "empty-bg" rank (textures.grad is None) must still fabricate a zero
    # grad and enter the collective so ranks stay in lockstep.
    bg = _cubemap(8)
    assert bg.textures.grad is None
    ar = _FakeAllReduce()
    _patch_distributed(monkeypatch, 2, ar)
    bg.sync_texture_grad()
    assert bg.textures.grad is not None
    torch.testing.assert_close(bg.textures.grad, torch.zeros_like(bg.textures))
    assert ar.ops == [torch.distributed.ReduceOp.SUM]


def test_sync_texture_grad_single_rank_skips_collective(monkeypatch):
    # world_size=1: no div, no collective, grad untouched.
    bg = _cubemap(8)
    g = torch.rand_like(bg.textures)
    bg.textures.grad = g.clone()
    ar = _FakeAllReduce()
    _patch_distributed(monkeypatch, 1, ar)
    bg.sync_texture_grad()
    torch.testing.assert_close(bg.textures.grad, g)
    assert ar.ops == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_track_gradients_ddp_halves_tracked_magnitude(monkeypatch):
    # The DDP branch of _track_gradients divides the shard grad by world_size
    # before the (identity) all_reduce, so the tracked magnitude on a 2-rank
    # run is exactly half the single-rank magnitude for identical inputs.
    torch.manual_seed(3)
    dirs = _rand_dirs(64).cuda()
    gaussian_rgb = torch.rand(64, 3, device="cuda")
    opacity = torch.rand(64, device="cuda")
    tex = torch.rand(1, 6, 8, 8, 3, device="cuda")

    def tracked() -> torch.Tensor:
        bg = _cubemap(8).to("cuda")
        with torch.no_grad():
            bg.textures.copy_(tex)
        bg.n_grad_updates = bg.min_grad_updates
        bg.composite(gaussian_rgb, opacity, dirs, is_training=True)
        return bg.texture_grads.clone()

    single = tracked()  # distributed not initialized -> single-rank path
    ar = _FakeAllReduce()
    _patch_distributed(monkeypatch, 2, ar)
    ddp = tracked()
    torch.testing.assert_close(ddp, single / 2, atol=1e-6, rtol=1e-5)
    assert ar.ops == [torch.distributed.ReduceOp.SUM]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_cuda_moves_textures_and_grads():
    bg = _cubemap(8)
    bg.to("cuda")
    assert bg.textures.is_cuda
    assert bg.texture_grads.is_cuda
    # Sampling and gradient tracking run end-to-end on-device without a mismatch.
    dirs = _rand_dirs(16).cuda()
    gaussian_rgb = torch.rand(16, 3, device="cuda")
    opacity = torch.rand(16, device="cuda")
    bg.n_grad_updates = bg.min_grad_updates
    out = bg.composite(gaussian_rgb, opacity, dirs, is_training=True)
    assert out.is_cuda
    assert bg.texture_grads.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_from_state_dict_honors_device():
    bg = _cubemap(8)
    restored = EnvMapBackground.from_state_dict(bg.state_dict(), device="cuda")
    assert restored.textures.is_cuda
    assert restored.texture_grads.is_cuda
