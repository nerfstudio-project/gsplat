"""gsplat-on-gfx1151 smoke test.

Two stages:
  1. import     — dlopen gsplat/csrc.so + link against torch (no GPU op).
  2. rasterize  — a tiny rasterization on the iGPU, forward AND backward
                  (needs a healthy GPU).

The backward pass matters: the gfx1151 wave32 fixes (wave_size.gfx1151.patch)
live in the *backward* rasterize kernels (RasterizeToPixels*Bwd.cu + the
Utils.cuh warp reductions) and the fused projection kernels. A forward-only check
would compile-and-run while never touching the patched code, so this stage runs
forward + .backward() and asserts the grads are present and finite. NOTE: a
finite-grad check does NOT prove numerical correctness of the wave32 reductions
(F1/F2 in the audit) — that needs the P1 gradcheck-vs-torch test on the GPU.

Run `python smoke_test.py import` at build time (GPU-independent) and
`python smoke_test.py rasterize` (or no arg) at run time on the GPU.

If the rasterize stage segfaults with no output, the 512 MB dedicated-VRAM
carveout is wedged (see docker/gsplat-rocm/README.md) — clear it and retry.
"""

import sys


def stage_import() -> None:
    import gsplat
    print("IMPORT OK gsplat", getattr(gsplat, "__version__", "?"))


def stage_rasterize() -> None:
    import torch
    import gsplat

    d = "cuda"
    N = 100
    print("torch", torch.__version__, "hip", torch.version.hip,
          "cuda_available", torch.cuda.is_available(), flush=True)

    # Leaf inputs with grads so .backward() drives the patched backward kernels.
    means = torch.randn(N, 3, device=d, requires_grad=True)
    quats_raw = torch.randn(N, 4, device=d, requires_grad=True)
    scales_raw = torch.rand(N, 3, device=d, requires_grad=True)
    opac = torch.rand(N, device=d, requires_grad=True)
    colors = torch.rand(N, 3, device=d, requires_grad=True)
    quats = quats_raw / quats_raw.norm(dim=-1, keepdim=True)
    scales = scales_raw * 0.1  # keep splats small; grad flows to scales_raw
    viewmat = torch.eye(4, device=d)[None]
    viewmat[0, 2, 3] = 3.0
    K = torch.tensor([[[300.0, 0, 128.0], [0, 300.0, 128.0], [0, 0, 1.0]]], device=d)

    # Forward (default = 3DGS path).
    out, alpha, _ = gsplat.rasterization(
        means, quats, scales, opac, colors, viewmat, K, 256, 256
    )
    torch.cuda.synchronize()

    # Backward — exercises RasterizeToPixels3DGSBwd + the wave32 warp reductions.
    loss = out.mean() + alpha.mean()
    loss.backward()
    torch.cuda.synchronize()

    leaves = {"means": means, "quats": quats_raw, "scales": scales_raw,
              "opac": opac, "colors": colors}
    grad_norm = 0.0
    for name, t in leaves.items():
        assert t.grad is not None, f"no gradient for {name} (backward kernel ran?)"
        assert torch.isfinite(t.grad).all(), f"non-finite gradient for {name}"
        grad_norm += t.grad.norm().item()
    assert grad_norm > 0.0, "all-zero gradients — backward did not propagate"
    print("RASTERIZE+BACKWARD OK", tuple(out.shape), out.device,
          f"grad_norm_sum={grad_norm:.4f}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "rasterize"
    if stage == "import":
        stage_import()
    elif stage == "rasterize":
        stage_rasterize()
    else:
        sys.exit(f"unknown stage {stage!r} (use 'import' or 'rasterize')")
