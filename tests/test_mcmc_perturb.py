# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused CUDA MCMC perturbation kernel.

Verifies that the CUDA kernel in MCMCPerturbCUDA.cu produces results
numerically consistent with the PyTorch reference implementation in
gsplat/strategy/ops.py::inject_noise_to_position.

Usage:
    pytest tests/test_mcmc_perturb.py -s
"""

import pytest
import torch

import gsplat

device = torch.device("cuda:0")


def _pytorch_mcmc_perturb(positions, quats, scales_log, opacities_logit, noise, scaler):
    """Pure-PyTorch reference matching the CUDA kernel logic."""
    from gsplat import quat_scale_to_covar_preci

    scales = torch.exp(scales_log)
    covars, _ = quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False
    )

    density = torch.sigmoid(opacities_logit)

    def op_sigmoid(x, k=100, x0=0.995):
        return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

    w = op_sigmoid(1.0 - density) * scaler
    weighted_noise = noise * w.unsqueeze(-1)
    delta = torch.einsum("bij,bj->bi", covars, weighted_noise)
    return positions + delta


def _make_inputs(N=1024, seed=42):
    torch.manual_seed(seed)
    positions = torch.randn(N, 3, device=device, dtype=torch.float32)
    quats = torch.randn(N, 4, device=device, dtype=torch.float32)
    scales_log = torch.randn(N, 3, device=device, dtype=torch.float32)
    opacities_logit = torch.randn(N, device=device, dtype=torch.float32)
    noise = torch.randn(N, 3, device=device, dtype=torch.float32)
    return positions, quats, scales_log, opacities_logit, noise


def _cuda_mcmc_perturb(positions, quats, scales_log, opacities_logit, noise, scaler):
    """Call the fused CUDA kernel via torch.ops.gsplat."""
    pos = positions.clone()
    torch.ops.gsplat.mcmc_perturb_positions(
        pos,
        quats.contiguous(),
        scales_log.contiguous(),
        opacities_logit.flatten().contiguous(),
        noise.contiguous(),
        float(scaler),
    )
    return pos


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
class TestMCMCPerturbCUDA:
    def test_cuda_matches_pytorch(self):
        """CUDA kernel output should closely match the PyTorch reference."""
        positions, quats, scales_log, opacities_logit, noise = _make_inputs()
        scaler = 0.01

        expected = _pytorch_mcmc_perturb(
            positions, quats, scales_log, opacities_logit, noise, scaler
        )
        actual = _cuda_mcmc_perturb(
            positions, quats, scales_log, opacities_logit, noise, scaler
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_cuda_matches_pytorch_large_scaler(self):
        """Consistency with a larger scaler value."""
        positions, quats, scales_log, opacities_logit, noise = _make_inputs()
        scaler = 1.0

        expected = _pytorch_mcmc_perturb(
            positions, quats, scales_log, opacities_logit, noise, scaler
        )
        actual = _cuda_mcmc_perturb(
            positions, quats, scales_log, opacities_logit, noise, scaler
        )

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)

    def test_in_place_modification(self):
        """Kernel should modify positions in-place."""
        positions, quats, scales_log, opacities_logit, noise = _make_inputs()
        original = positions.clone()

        torch.ops.gsplat.mcmc_perturb_positions(
            positions,
            quats.contiguous(),
            scales_log.contiguous(),
            opacities_logit.flatten().contiguous(),
            noise.contiguous(),
            0.01,
        )

        assert not torch.allclose(
            positions, original
        ), "Positions should be modified in-place"

    def test_zero_scaler_no_change(self):
        """With scaler=0, positions should remain unchanged."""
        positions, quats, scales_log, opacities_logit, noise = _make_inputs()
        original = positions.clone()

        torch.ops.gsplat.mcmc_perturb_positions(
            positions,
            quats.contiguous(),
            scales_log.contiguous(),
            opacities_logit.flatten().contiguous(),
            noise.contiguous(),
            0.0,
        )

        torch.testing.assert_close(positions, original)

    def test_empty_tensor(self):
        """Kernel should handle N=0 gracefully."""
        positions = torch.empty(0, 3, device=device, dtype=torch.float32)
        quats = torch.empty(0, 4, device=device, dtype=torch.float32)
        scales = torch.empty(0, 3, device=device, dtype=torch.float32)
        opacities = torch.empty(0, device=device, dtype=torch.float32)
        noise = torch.empty(0, 3, device=device, dtype=torch.float32)

        torch.ops.gsplat.mcmc_perturb_positions(
            positions, quats, scales, opacities, noise, 0.01
        )
        assert positions.shape == (0, 3)

    def test_single_gaussian(self):
        """Verify correctness for a single Gaussian."""
        positions, quats, scales_log, opacities_logit, noise = _make_inputs(N=1)
        scaler = 0.05

        expected = _pytorch_mcmc_perturb(
            positions, quats, scales_log, opacities_logit, noise, scaler
        )
        actual = _cuda_mcmc_perturb(
            positions, quats, scales_log, opacities_logit, noise, scaler
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_high_opacity_minimal_noise(self):
        """High-opacity Gaussians should receive near-zero perturbation."""
        N = 256
        positions, quats, scales_log, _, noise = _make_inputs(N=N)
        opacities_logit = torch.full((N,), 10.0, device=device, dtype=torch.float32)

        original = positions.clone()
        torch.ops.gsplat.mcmc_perturb_positions(
            positions,
            quats.contiguous(),
            scales_log.contiguous(),
            opacities_logit.contiguous(),
            noise.contiguous(),
            0.01,
        )

        max_delta = (positions - original).abs().max().item()
        assert (
            max_delta < 1e-3
        ), f"High-opacity Gaussians should barely move, but max_delta={max_delta}"

    def test_various_sizes(self):
        """Test with different N to catch alignment/boundary issues."""
        for N in [1, 7, 128, 255, 256, 257, 1000, 4096]:
            positions, quats, scales_log, opacities_logit, noise = _make_inputs(N=N)
            expected = _pytorch_mcmc_perturb(
                positions, quats, scales_log, opacities_logit, noise, 0.01
            )
            actual = _cuda_mcmc_perturb(
                positions, quats, scales_log, opacities_logit, noise, 0.01
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_deterministic_with_same_noise(self):
        """Same noise input should produce identical output across calls."""
        positions, quats, scales_log, opacities_logit, noise = _make_inputs()
        result1 = _cuda_mcmc_perturb(
            positions, quats, scales_log, opacities_logit, noise, 0.01
        )
        result2 = _cuda_mcmc_perturb(
            positions, quats, scales_log, opacities_logit, noise, 0.01
        )
        torch.testing.assert_close(result1, result2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
class TestMCMCPerturbFallback:
    def test_fallback_path(self):
        """_cuda_fused_mcmc_perturb returns False for CPU tensors."""
        from gsplat.strategy.ops import _cuda_fused_mcmc_perturb

        positions = torch.randn(10, 3, dtype=torch.float32)
        quats = torch.randn(10, 4, dtype=torch.float32)
        scales = torch.randn(10, 3, dtype=torch.float32)
        opacities = torch.randn(10, dtype=torch.float32)

        assert not _cuda_fused_mcmc_perturb(positions, quats, scales, opacities, 0.01)

    def test_fallback_wrong_dtype(self):
        """_cuda_fused_mcmc_perturb returns False for float64 tensors."""
        from gsplat.strategy.ops import _cuda_fused_mcmc_perturb

        positions = torch.randn(10, 3, device=device, dtype=torch.float64)
        quats = torch.randn(10, 4, device=device, dtype=torch.float64)
        scales = torch.randn(10, 3, device=device, dtype=torch.float64)
        opacities = torch.randn(10, device=device, dtype=torch.float64)

        assert not _cuda_fused_mcmc_perturb(positions, quats, scales, opacities, 0.01)

    def test_fallback_non_contiguous_positions(self):
        """_cuda_fused_mcmc_perturb returns False when positions are non-contiguous."""
        from gsplat.strategy.ops import _cuda_fused_mcmc_perturb

        N = 10
        positions = torch.randn(N, 6, device=device, dtype=torch.float32)[:, :3]
        assert not positions.is_contiguous()
        quats = torch.randn(N, 4, device=device, dtype=torch.float32)
        scales = torch.randn(N, 3, device=device, dtype=torch.float32)
        opacities = torch.randn(N, device=device, dtype=torch.float32)

        assert not _cuda_fused_mcmc_perturb(positions, quats, scales, opacities, 0.01)

    def test_cuda_path_succeeds(self):
        """_cuda_fused_mcmc_perturb returns True for valid CUDA float32 tensors."""
        from gsplat.strategy.ops import _cuda_fused_mcmc_perturb

        positions = torch.randn(10, 3, device=device, dtype=torch.float32)
        quats = torch.randn(10, 4, device=device, dtype=torch.float32)
        scales = torch.randn(10, 3, device=device, dtype=torch.float32)
        opacities = torch.randn(10, device=device, dtype=torch.float32)

        assert _cuda_fused_mcmc_perturb(positions, quats, scales, opacities, 0.01)

    def test_inject_noise_drives_fallback_when_cuda_path_bails(self, monkeypatch):
        """When _cuda_fused_mcmc_perturb returns False, inject_noise_to_position
        executes the PyTorch fallback and matches the reference numerically."""
        from gsplat.strategy import ops as gsplat_ops

        monkeypatch.setattr(
            gsplat_ops, "_cuda_fused_mcmc_perturb", lambda *a, **kw: False
        )

        N = 64
        scaler = 0.05
        torch.manual_seed(0)
        means = torch.nn.Parameter(
            torch.randn(N, 3, device=device, dtype=torch.float32)
        )
        params = {
            "means": means,
            "quats": torch.nn.Parameter(
                torch.randn(N, 4, device=device, dtype=torch.float32)
            ),
            "scales": torch.nn.Parameter(
                torch.randn(N, 3, device=device, dtype=torch.float32)
            ),
            "opacities": torch.nn.Parameter(
                torch.randn(N, device=device, dtype=torch.float32)
            ),
        }
        original = means.detach().clone()

        rng_state = torch.cuda.get_rng_state(device)
        gsplat_ops.inject_noise_to_position(
            params, optimizers={}, state={}, scaler=scaler
        )

        torch.cuda.set_rng_state(rng_state, device)
        noise = torch.randn_like(original)
        expected = _pytorch_mcmc_perturb(
            original,
            params["quats"].detach(),
            params["scales"].detach(),
            params["opacities"].detach(),
            noise,
            scaler,
        )
        torch.testing.assert_close(means.detach(), expected, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
