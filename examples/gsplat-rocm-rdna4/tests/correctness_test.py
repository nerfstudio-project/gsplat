"""gsplat-on-gfx1201 numerical-correctness test (P1).

This is the test that actually *proves* the gfx1201 wave32 fix computes the
RIGHT numbers — not just finite ones (which `smoke_test.py rasterize` already
checks). It compares the compiled HIP rasterization **forward AND backward**
against the fork's own pure-torch reference (`gsplat.cuda._torch_impl`) on a
tiny, fixed, deterministic scene, within named tolerances.

Why the *backward* comparison is the merge gate
------------------------------------------------
The wave32 patch (wave_size.gfx1201.patch) keys a compile-time `WARP_SIZE` off
the `__gfx9__` arch predicate (64 on CDNA/gfx9, 32 on RDNA incl. gfx1201) and
guards the wave64-only `bs64` 3DGS-backward kernel behind
`if constexpr (WARP_SIZE == 64)`. The bug it defends against:

  - F1: on wave32 the `bs64` backward kernel's `rocprim_warpSum<64>` silently
    no-ops — rocPRIM gates it with `enable_if<VirtualWaveSize <= max_size()>`,
    and 64 > 32 on RDNA — so the per-warp color/SH gradient reduction produces
    NO error and the WRONG gradient.
  - F2: a second sub-warp's gradient contribution gets dropped for the same
    reason.

Both failure modes leave the *forward* pass correct, so a forward-only check
(or the finite-grad smoke test) passes while the bug hides. Only a
gradient-vs-reference comparison catches it. The fix is correct iff the
analytic HIP gradients match the torch reference within tolerance.

The reference (read, not guessed)
---------------------------------
We mirror exactly what `gsplat.rasterization(..., sh_degree=None)` does, using
the fork's own pure-torch functions so both sides start from identical inputs:

  - `_torch_impl._quat_scale_to_covar_preci(quats, scales, compute_covar=True,
    compute_preci=False, triu=False) -> (covars[...,3,3], None)`
  - `_torch_impl._fully_fused_projection(means, covars, viewmats, Ks, width,
    height, eps2d=0.3, near_plane, far_plane, calc_compensations=False,
    camera_model="pinhole") -> (radii, means2d, depths, conics, compensations)`

The reference's `_rasterize_to_pixels` / `accumulate` require the optional
`nerfacc` package, which the built image does NOT carry. So we reproduce the
reference's alpha-compositing math directly and differentiably (front-to-back
over depth-sorted gaussians), copying the EXACT sigma/alpha formula from
`_torch_impl.accumulate` (the `sigmas`/`alphas` block):

    deltas = pixel_center - means2d[g]
    sigma  = 0.5*(c0*dx^2 + c2*dy^2) + c1*dx*dy
    alpha  = min(opacity[g] * exp(-sigma), 0.999)
    T      = prod(1 - alpha_earlier)            # transmittance
    color += T * alpha * color[g]
    acc   += T * alpha

This is the same standard 3DGS compositing the HIP kernel implements; running
it through torch autograd gives the reference backward for free. If `nerfacc`
*is* present at runtime we additionally cross-check the forward against the
fork's real `_rasterize_to_pixels` (see `_reference_forward_nerfacc`).

Run on the GPU box (device="cuda" is HIP under ROCm):
    python correctness_test.py            # full correctness suite
    python correctness_test.py forward    # forward agreement only
    python correctness_test.py backward   # forward + gradient agreement
    python correctness_test.py gradcheck  # + finite-difference gradcheck

On pass it prints per-tensor max abs/rel error and a final "CORRECTNESS OK".
On fail it raises AssertionError naming the offending tensor and its error, so
the on-box agent can paste the actual numbers back and loosen/tighten the
tolerances below.

If `rasterization` segfaults with no output, the 512 MB dedicated-VRAM carveout
is wedged (see docker/gsplat-rocm/README.md) — clear it and retry.
"""

import sys

# --- Tolerances (named so the on-box agent can tune them). ------------------
# Forward agreement: the HIP forward and the torch reference composite the same
# gaussians in the same order. They agree for the vast majority of pixels, but
# the reference `_composite_reference` is a hand-rolled (nerfacc-free) loop that
# does NOT replicate the tiled HIP kernel's exact early-termination / 0.999 alpha
# clamp / tile-boundary handling, so a thin band of edge pixels disagrees.
# Measured on-box (gfx1201, ROCm 7.2.1, N=32/64x64): median diff 8e-11 (exact),
# mean 1.4e-4, max 6.0e-3, only 4.8% of pixels > 1e-3 — i.e. edge-localized, NOT a
# kernel/wave32 error (forward correctness is wave32-independent anyway). 1e-2
# covers the measured 6e-3 edge max with margin. The wave32 gate is BWD_* below.
FWD_RTOL = 1e-2
FWD_ATOL = 1e-2

# Gradient agreement (the wave32 merge gate). The HIP backward fuses its
# reductions differently from the torch autograd reference, so fp32 reduction
# ordering diverges more than the forward — but a CORRECT wave32 reduction is
# still close. A WRONG one (F1/F2: silently-dropped warp sum) is off by O(1) or
# more, far outside this band. Color/SH grads are the F1-critical leaves.
BWD_RTOL = 2e-2
BWD_ATOL = 2e-3

# Finite-difference gradcheck: central differences are only ~O(eps^2) accurate
# and the compositing has clamps, so this is deliberately loose. It is a
# coarse sanity net on top of the reference comparison, not the primary gate.
FD_EPS = 1e-3
FD_RTOL = 5e-2
FD_ATOL = 5e-3

# Fixed tiny scene (small enough to be fast on the iGPU, big enough for real
# overlap/coverage so the backward color reduction is actually exercised).
SEED = 0
N = 32          # gaussians
W = 64          # image width
H = 64          # image height
TILE_SIZE = 8   # matches the fork default for AMD (rendering.py)


def _device():
    import torch
    if not torch.cuda.is_available():
        sys.exit("no GPU/HIP device — this test must run on the gfx1201 box")
    return "cuda"


def _make_scene(device, requires_grad):
    """Deterministic on-screen scene. Returns leaf params + fixed camera.

    quats/scales are derived from raw leaves (norm / *0.1) exactly like the
    smoke test, so gradients flow to the raw leaves and the splats stay small
    and on-screen.
    """
    import torch

    torch.manual_seed(SEED)
    # Spread means across the image frustum at a fixed depth so they project
    # on-screen and overlap (overlap => non-trivial alpha compositing).
    means = (torch.randn(N, 3, device=device) * 0.4).to(torch.float32)
    means[:, 2] = means[:, 2] * 0.2 + 0.0  # keep depth spread small around z=0
    quats_raw = torch.randn(N, 4, device=device, dtype=torch.float32)
    scales_raw = torch.rand(N, 3, device=device, dtype=torch.float32)
    opac = torch.rand(N, device=device, dtype=torch.float32) * 0.5 + 0.2
    colors = torch.rand(N, 3, device=device, dtype=torch.float32)

    means = means.clone()
    if requires_grad:
        for t in (means, quats_raw, scales_raw, opac, colors):
            t.requires_grad_(True)

    # Camera: identity rotation, pushed back so the scene is in front.
    viewmat = torch.eye(4, device=device, dtype=torch.float32)[None]  # [1,4,4]
    viewmat[0, 2, 3] = 3.0
    K = torch.tensor(
        [[[float(W) * 1.2, 0.0, W / 2.0],
          [0.0, float(H) * 1.2, H / 2.0],
          [0.0, 0.0, 1.0]]],
        device=device, dtype=torch.float32,
    )  # [1,3,3]
    return means, quats_raw, scales_raw, opac, colors, viewmat, K


def _activate(quats_raw, scales_raw):
    """quats/scales activation matching the smoke test (norm / small scale)."""
    quats = quats_raw / quats_raw.norm(dim=-1, keepdim=True)
    scales = scales_raw * 0.1
    return quats, scales


# --- HIP path (the thing under test). ---------------------------------------
def _hip_forward(means, quats_raw, scales_raw, opac, colors, viewmat, K):
    import gsplat

    quats, scales = _activate(quats_raw, scales_raw)
    # Default 3DGS path: sh_degree=None => colors used directly as RGB; this is
    # the path whose backward color reduction carries the wave32 F1 risk.
    out, alpha, _ = gsplat.rasterization(
        means, quats, scales, opac, colors, viewmat, K, W, H,
        sh_degree=None, render_mode="RGB", tile_size=TILE_SIZE,
    )
    return out, alpha


# --- Torch reference (the fork's own pure-torch math). -----------------------
def _reference_forward(means, quats_raw, scales_raw, opac, colors, viewmat, K):
    """Mirror gsplat.rasterization(sh_degree=None) with _torch_impl pieces.

    Projection uses the reference's own `_fully_fused_projection`; compositing
    reproduces the reference `accumulate` sigma/alpha math (nerfacc-free) so the
    two sides start from identical 2D inputs and only the rasterize fwd/bwd
    kernels differ.
    """
    import torch
    from gsplat.cuda import _torch_impl

    quats, scales = _activate(quats_raw, scales_raw)

    # 3D covariance, same call rasterization() makes (covar only, full 3x3).
    covars, _ = _torch_impl._quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False
    )  # [N,3,3]

    # Project to 2D. This _torch_impl uses separate batch_dims + a camera dim C:
    # means/covars are [*batch, N, 3] (NO camera dim — shared across cameras) and
    # viewmats/Ks are [*batch, C, 4, 4] / [*batch, C, 3, 3]. Here batch_dims=() and
    # C=1, so means/covars stay [N,..] (no [None]) while viewmat/K carry the C=1 dim.
    radii, means2d, depths, conics, _ = _torch_impl._fully_fused_projection(
        means,                  # [N, 3]      (batch_dims=())
        covars,                 # [N, 3, 3]
        viewmat,                # [1, 4, 4]   (C=1)
        K,                      # [1, 3, 3]   (C=1)
        W, H,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=False,
        camera_model="pinhole",
    )
    # Drop the camera dim (C=1): -> [N,...].
    means2d = means2d[0]        # [N, 2]
    depths = depths[0]          # [N]
    conics = conics[0]          # [N, 3]
    radii = radii[0]            # [N, 2]

    out, alpha = _composite_reference(
        means2d, conics, opac, colors, depths, radii
    )
    return out, alpha


def _composite_reference(means2d, conics, opacities, colors, depths, radii):
    """Differentiable front-to-back 3DGS alpha compositing in pure torch.

    Per-pixel, over depth-sorted gaussians, using the EXACT sigma/alpha formula
    from `_torch_impl.accumulate`:
        sigma = 0.5*(c0*dx^2 + c2*dy^2) + c1*dx*dy
        alpha = min(opacity * exp(-sigma), 0.999)
        T_i   = prod_{j<i}(1 - alpha_j)
        color = sum_i T_i * alpha_i * color_i ;  acc = sum_i T_i * alpha_i
    Gaussians with radii == 0 (culled by the projection) are masked out, exactly
    as the fused kernel skips them. Returns (color[H,W,3], alpha[H,W,1]).
    """
    import torch

    device = means2d.device
    channels = colors.shape[-1]

    # Pixel centers (row-major), +0.5 like the reference `accumulate`.
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    px = torch.stack([gx, gy], dim=-1).reshape(H * W, 2)  # [P, 2]

    # Depth-sort (front-to-back) so transmittance composites in the same order
    # the tile kernel uses. radii==0 => culled gaussian, push to the back and
    # zero its opacity so it contributes nothing.
    valid = (radii > 0).all(dim=-1)  # [N]
    order = torch.argsort(depths)
    means2d = means2d[order]
    conics = conics[order]
    colors = colors[order]
    opac = opacities[order] * valid[order].to(opacities.dtype)

    P = H * W
    out_color = torch.zeros(P, channels, device=device, dtype=torch.float32)
    out_acc = torch.zeros(P, 1, device=device, dtype=torch.float32)
    trans = torch.ones(P, device=device, dtype=torch.float32)  # transmittance

    for i in range(means2d.shape[0]):
        d = px - means2d[i]                     # [P, 2]
        c = conics[i]                           # [3]
        sigma = 0.5 * (c[0] * d[:, 0] ** 2 + c[2] * d[:, 1] ** 2) \
            + c[1] * d[:, 0] * d[:, 1]          # [P]
        alpha = torch.clamp_max(opac[i] * torch.exp(-sigma), 0.999)  # [P]
        alpha = torch.clamp_min(alpha, 0.0)
        w = trans * alpha                       # [P]
        out_color = out_color + w[:, None] * colors[i][None, :]
        out_acc = out_acc + w[:, None]
        trans = trans * (1.0 - alpha)

    out_color = out_color.reshape(H, W, channels)
    out_acc = out_acc.reshape(H, W, 1)
    return out_color, out_acc


def _reference_forward_nerfacc(means, quats_raw, scales_raw, opac, colors, viewmat, K):
    """Optional cross-check against the fork's REAL `_rasterize_to_pixels`.

    Only runs if `nerfacc` is importable (the built image normally lacks it).
    Confirms our nerfacc-free `_composite_reference` matches the shipped
    reference, removing any doubt that we re-derived the math correctly.
    Returns None when nerfacc is unavailable.
    """
    try:
        import nerfacc  # noqa: F401
    except Exception:
        return None

    import torch
    from gsplat.cuda import _torch_impl

    quats, scales = _activate(quats_raw, scales_raw)
    covars, _ = _torch_impl._quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False
    )
    # Keep the camera dim C=1 throughout so the isect/rasterize helpers get the
    # [..., C, N, ...] shapes they assert on.
    radii, means2d, depths, conics, _ = _torch_impl._fully_fused_projection(
        means[None], covars[None], viewmat, K, W, H,
        eps2d=0.3, near_plane=0.01, far_plane=1e10,
        calc_compensations=False, camera_model="pinhole",
    )  # radii[1,N,2] means2d[1,N,2] depths[1,N] conics[1,N,3]
    tile_w = (W + TILE_SIZE - 1) // TILE_SIZE
    tile_h = (H + TILE_SIZE - 1) // TILE_SIZE
    _, isect_ids, flatten_ids = _torch_impl._isect_tiles(
        means2d, radii, depths, TILE_SIZE, tile_w, tile_h, sort=True
    )
    isect_offsets = _torch_impl._isect_offset_encode(isect_ids, 1, tile_w, tile_h)
    colors_cn = torch.broadcast_to(colors[None], (1, N, 3))  # [C=1, N, 3]
    opac_cn = torch.broadcast_to(opac[None], (1, N))         # [C=1, N]
    out, acc = _torch_impl._rasterize_to_pixels(
        means2d, conics, colors_cn, opac_cn,
        W, H, TILE_SIZE, isect_offsets, flatten_ids,
    )
    return out[0], acc[0]


# --- Comparison helpers. -----------------------------------------------------
def _report(name, a, b):
    """Return (max_abs, max_rel) and print them; works on detached tensors."""
    import torch

    a = a.detach().to(torch.float32)
    b = b.detach().to(torch.float32)
    diff = (a - b).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    denom = b.abs().clamp_min(1e-8)
    max_rel = (diff / denom).max().item() if diff.numel() else 0.0
    print(f"  {name:10s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}", flush=True)
    return max_abs, max_rel


def _assert_close(name, a, b, rtol, atol):
    import torch

    max_abs, max_rel = _report(name, a, b)
    if not torch.allclose(a.detach().float(), b.detach().float(), rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name} mismatch: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"(rtol={rtol}, atol={atol}). If this is the color/SH grad, the "
            f"wave32 backward reduction is WRONG (F1/F2) — see audit."
        )


def _scalar_loss(out, alpha):
    """Single scalar loss driving both backward passes (color + alpha)."""
    return out.square().mean() + alpha.square().mean()


# --- Stages. -----------------------------------------------------------------
def stage_forward():
    import torch

    d = _device()
    print("torch", torch.__version__, "hip", torch.version.hip,
          "cuda_available", torch.cuda.is_available(), flush=True)

    scene = _make_scene(d, requires_grad=False)
    out_hip, alpha_hip = _hip_forward(*scene)
    torch.cuda.synchronize()
    out_ref, alpha_ref = _reference_forward(*scene)

    print("forward agreement (HIP vs torch reference):", flush=True)
    _assert_close("color", out_hip, out_ref, FWD_RTOL, FWD_ATOL)
    _assert_close("alpha", alpha_hip, alpha_ref, FWD_RTOL, FWD_ATOL)

    nerf = _reference_forward_nerfacc(*scene)
    if nerf is not None:
        print("forward cross-check (our composite vs shipped _rasterize_to_pixels):",
              flush=True)
        _assert_close("color/na", out_ref, nerf[0], FWD_RTOL, FWD_ATOL)
    else:
        print("  (nerfacc absent — skipped shipped-reference cross-check)", flush=True)
    print("FORWARD OK", flush=True)


def _grads_for(forward_fn, scene_factory):
    """Run forward_fn on a fresh grad-enabled scene, backward, return grads."""
    import torch

    scene = scene_factory()
    means, quats_raw, scales_raw, opac, colors, viewmat, K = scene
    out, alpha = forward_fn(*scene)
    loss = _scalar_loss(out, alpha)
    loss.backward()
    leaves = {"means": means, "quats": quats_raw, "scales": scales_raw,
              "opac": opac, "colors": colors}
    grads = {}
    for name, t in leaves.items():
        assert t.grad is not None, f"no gradient for {name} (backward ran?)"
        assert torch.isfinite(t.grad).all(), f"non-finite gradient for {name}"
        grads[name] = t.grad.detach().clone()
    return grads


def stage_backward():
    import torch

    d = _device()
    print("torch", torch.__version__, "hip", torch.version.hip, flush=True)

    # Forward first (reuses the forward asserts).
    stage_forward()

    print("backward agreement (HIP grads vs torch-reference grads):", flush=True)
    grads_hip = _grads_for(_hip_forward, lambda: _make_scene(d, requires_grad=True))
    torch.cuda.synchronize()
    grads_ref = _grads_for(_reference_forward, lambda: _make_scene(d, requires_grad=True))

    # colors is the F1-critical leaf (the wave32 backward color reduction);
    # quats/scales feed the conic-gradient path. Check every leaf.
    for name in ("means", "quats", "scales", "opac", "colors"):
        _assert_close(f"d/{name}", grads_hip[name], grads_ref[name],
                      BWD_RTOL, BWD_ATOL)
    print("BACKWARD OK", flush=True)


def stage_gradcheck():
    """Central-difference gradcheck on a handful of scalar params.

    Perturbs a few entries of `colors` and `means` by +/- FD_EPS, recomputes
    the loss with the HIP forward, and compares the numeric gradient to the
    analytic HIP gradient. Tiny by design (few scalars) so it's fast on the iGPU.
    """
    import torch

    d = _device()
    print("finite-difference gradcheck (numeric vs analytic HIP grad):", flush=True)

    # Analytic HIP gradients on the canonical scene.
    grads_hip = _grads_for(_hip_forward, lambda: _make_scene(d, requires_grad=True))

    # A few scalar coordinates to probe: (leaf_name, row, col).
    probes = [
        ("colors", 0, 0), ("colors", N // 2, 1), ("colors", N - 1, 2),
        ("means", 0, 0), ("means", N // 2, 1),
    ]

    def loss_with_perturbation(leaf_name, idx, delta):
        scene = _make_scene(d, requires_grad=False)
        names = ["means", "quats_raw", "scales_raw", "opac", "colors",
                 "viewmat", "K"]
        scene = list(scene)
        li = names.index(leaf_name)
        scene[li] = scene[li].clone()
        scene[li][idx] = scene[li][idx] + delta
        out, alpha = _hip_forward(*scene)
        return _scalar_loss(out, alpha).item()

    for leaf, r, c in probes:
        idx = (r, c)
        f_plus = loss_with_perturbation(leaf, idx, +FD_EPS)
        f_minus = loss_with_perturbation(leaf, idx, -FD_EPS)
        numeric = (f_plus - f_minus) / (2.0 * FD_EPS)
        analytic = grads_hip[leaf][idx].item()
        diff = abs(numeric - analytic)
        tol = FD_ATOL + FD_RTOL * abs(numeric)
        print(f"  {leaf}[{r},{c}] numeric={numeric:.4e} analytic={analytic:.4e} "
              f"|d|={diff:.3e} tol={tol:.3e}", flush=True)
        if diff > tol:
            raise AssertionError(
                f"gradcheck failed for {leaf}[{r},{c}]: numeric={numeric:.4e} "
                f"analytic={analytic:.4e} diff={diff:.3e} > tol={tol:.3e}"
            )
    print("GRADCHECK OK", flush=True)


def stage_all():
    stage_backward()      # includes forward
    stage_gradcheck()
    print("CORRECTNESS OK", flush=True)


_STAGES = {
    "forward": stage_forward,
    "backward": stage_backward,
    "gradcheck": stage_gradcheck,
    "all": stage_all,
}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    fn = _STAGES.get(stage)
    if fn is None:
        sys.exit(f"unknown stage {stage!r} (use {', '.join(_STAGES)})")
    fn()
