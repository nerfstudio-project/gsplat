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

"""Tests for BivariateWindshieldModel external distortion.

Tests cover:
- compute_order mapping from coefficient count to polynomial order (pure Python)
- BivariateWindshieldModelParameters construction and constant validation (pure Python)
- CUDA eval_bivariate_poly kernel (cross-validated against Python reference)
- CUDA distort_camera_rays kernel (cross-validated against Python reference)
- Camera model integration via BaseCameraModel.create() with external distortion
- Integration testing with full rendering pipeline (3DGUT)
"""

import math
import os

import pytest
import torch

import gsplat
from gsplat.cuda._wrapper import (
    BivariateWindshieldModelParameters,
    _make_lazy_cuda_func,
    ExternalDistortionReferencePolynomial,
)
from gsplat.cuda._torch_external_distortion import (  # PyTorch reference
    ref_compute_order,
    ref_eval_bivariate_poly,
    ref_distort_camera_ray,
    num_coeffs_for_order,
    make_identity_horizontal_poly,
    make_identity_vertical_poly,
    make_zero_poly,
    make_params,
)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# ===========================================================================
# Helper functions
# ===========================================================================


def distort_camera_rays_cuda(
    rays: torch.Tensor,
    params: "BivariateWindshieldModelParameters",
    inverse: bool = False,
) -> torch.Tensor:
    """Distort/undistort camera rays using the CUDA bivariate windshield model.

    Requires GSPLAT_BUILD_CAMERA_WRAPPERS=1.

    Args:
        rays: Input rays [N, 3] (float32, CUDA)
        params: Bivariate windshield model parameters
        inverse: If True, apply inverse (undistort). Default False (forward distort).

    Returns:
        Distorted rays [N, 3] (float32, CUDA)
    """
    return _make_lazy_cuda_func("distort_camera_rays")(
        rays,
        params.horizontal_poly,
        params.vertical_poly,
        params.horizontal_poly_inverse,
        params.vertical_poly_inverse,
        int(params.reference_poly),
        inverse,
    )


def eval_bivariate_poly_cuda(
    x: torch.Tensor, y: torch.Tensor, poly_coeffs: torch.Tensor, order: int
) -> torch.Tensor:
    """Evaluate a 2D bivariate polynomial at (x, y) points using CUDA.

    Requires GSPLAT_BUILD_CAMERA_WRAPPERS=1.

    Args:
        x: Input x values [N] (float32, CUDA)
        y: Input y values [N] (float32, CUDA)
        poly_coeffs: Polynomial coefficients [num_coeffs] (float32, CUDA)
        order: Polynomial order

    Returns:
        Evaluated values [N] (float32, CUDA)
    """
    return _make_lazy_cuda_func("eval_bivariate_poly")(x, y, poly_coeffs, order)


# ===========================================================================
# 1. compute_order tests (pure Python)
# ===========================================================================


class TestComputeOrder:
    """Verify compute_order maps coefficient count -> polynomial order correctly."""

    @pytest.mark.parametrize(
        "order, expected_coeffs",
        [
            (0, 1),
            (1, 3),
            (2, 6),
            (3, 10),
            (4, 15),
            (5, 21),  # MAX_ORDER
        ],
    )
    def test_num_coeffs_formula(self, order, expected_coeffs):
        assert num_coeffs_for_order(order) == expected_coeffs

    @pytest.mark.parametrize("order", list(range(6)))
    def test_round_trip(self, order):
        n = num_coeffs_for_order(order)
        assert ref_compute_order(n) == order


# ===========================================================================
# 2. BivariateWindshieldModelParameters construction tests (pure Python)
# ===========================================================================


class TestParameterConstruction:
    """Test Python dataclass construction and constant values."""

    def test_max_order_constant(self):
        assert BivariateWindshieldModelParameters.MAX_ORDER == 5

    def test_max_coeffs_constant(self):
        assert BivariateWindshieldModelParameters.MAX_COEFFS == 21

    def test_max_coeffs_matches_max_order(self):
        expected = num_coeffs_for_order(BivariateWindshieldModelParameters.MAX_ORDER)
        assert BivariateWindshieldModelParameters.MAX_COEFFS == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_construct_order1(self):
        params = make_params(
            h_poly=make_identity_horizontal_poly(),
            v_poly=make_identity_vertical_poly(),
        )
        assert params.horizontal_poly.shape == (3,)
        assert params.vertical_poly.shape == (3,)
        assert params.reference_poly == ExternalDistortionReferencePolynomial.FORWARD

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_construct_order5(self):
        """Construct with maximum order (5) polynomials."""
        n = num_coeffs_for_order(5)
        h = [0.0] * n
        v = [0.0] * n
        h[1] = 1.0  # identity-like for horizontal
        v[2] = 1.0  # identity-like for vertical
        params = make_params(h_poly=h, v_poly=v, h_inv=h, v_inv=v)
        assert params.horizontal_poly.shape == (21,)
        assert params.vertical_poly.shape == (21,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_construct_all_orders(self):
        """Ensure we can construct parameters for every valid order 0..5."""
        for order in range(BivariateWindshieldModelParameters.MAX_ORDER + 1):
            n = num_coeffs_for_order(order)
            coeffs = [0.0] * n
            params = make_params(
                h_poly=coeffs, v_poly=coeffs, h_inv=coeffs, v_inv=coeffs
            )
            assert params.horizontal_poly.shape == (n,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_reference_poly_forward(self):
        params = make_params(
            h_poly=make_identity_horizontal_poly(),
            v_poly=make_identity_vertical_poly(),
            ref_poly=ExternalDistortionReferencePolynomial.FORWARD,
        )
        assert params.reference_poly == ExternalDistortionReferencePolynomial.FORWARD

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_reference_poly_backward(self):
        params = make_params(
            h_poly=make_identity_horizontal_poly(),
            v_poly=make_identity_vertical_poly(),
            ref_poly=ExternalDistortionReferencePolynomial.BACKWARD,
        )
        assert params.reference_poly == ExternalDistortionReferencePolynomial.BACKWARD

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_params_tensor_shapes(self):
        """Verify constructed params have correct tensor shapes."""
        if not gsplat.has_3dgut():
            pytest.skip("CUDA extension not available")
        if not gsplat.has_camera_wrappers():
            pytest.skip("Camera wrappers not built (need BUILD_CAMERA_WRAPPERS=1)")
        params = make_params(
            h_poly=make_identity_horizontal_poly(),
            v_poly=make_identity_vertical_poly(),
        )
        assert params is not None
        assert params.horizontal_poly.shape == (3,)
        assert params.vertical_poly.shape == (3,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_params_max_order(self):
        """Verify params construction works with maximum-order polynomials."""
        if not gsplat.has_3dgut():
            pytest.skip("CUDA extension not available")
        if not gsplat.has_camera_wrappers():
            pytest.skip("Camera wrappers not built (need BUILD_CAMERA_WRAPPERS=1)")
        n = num_coeffs_for_order(5)
        coeffs = [0.0] * n
        params = make_params(h_poly=coeffs, v_poly=coeffs, h_inv=coeffs, v_inv=coeffs)
        assert params.horizontal_poly.shape == (n,)


# ===========================================================================
# 3. CUDA eval_bivariate_poly tests
#    Calls the actual CUDA kernel and cross-validates against Python reference.
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestBivariatePolyEvaluationCUDA:
    """Test the CUDA eval_bivariate_poly kernel against Python reference."""

    @pytest.fixture(autouse=True)
    def _require_camera_wrappers(self):
        if not gsplat.has_camera_wrappers():
            pytest.skip("Camera wrappers not built (need BUILD_CAMERA_WRAPPERS=1)")

    @staticmethod
    def _eval_cuda(
        poly_coeffs: list, order: int, x_vals: list, y_vals: list
    ) -> torch.Tensor:
        x = torch.tensor(x_vals, dtype=torch.float32, device="cuda")
        y = torch.tensor(y_vals, dtype=torch.float32, device="cuda")
        coeffs = torch.tensor(poly_coeffs, dtype=torch.float32, device="cuda")
        return eval_bivariate_poly_cuda(x, y, coeffs, order)

    def test_constant_poly(self):
        """Order 0: single constant coefficient."""
        result = self._eval_cuda([3.5], 0, [0.0, 1.0], [0.0, 2.0])
        assert result[0].item() == pytest.approx(3.5, abs=1e-6)
        assert result[1].item() == pytest.approx(3.5, abs=1e-6)

    def test_linear_identity_horizontal(self):
        """Order 1: f(x,y) = x."""
        poly = [0.0, 1.0, 0.0]
        result = self._eval_cuda(poly, 1, [0.5], [0.7])
        assert result[0].item() == pytest.approx(0.5, abs=1e-6)

    def test_linear_identity_vertical(self):
        """Order 1: f(x,y) = y."""
        poly = [0.0, 0.0, 1.0]
        result = self._eval_cuda(poly, 1, [0.5], [0.7])
        assert result[0].item() == pytest.approx(0.7, abs=1e-6)

    def test_linear_general(self):
        """Order 1: f(x,y) = 2 + 3x + 5y."""
        poly = [2.0, 3.0, 5.0]
        result = self._eval_cuda(poly, 1, [1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 0.0, 3.0])
        expected = [10.0, 2.0, 8.0, 17.0]
        for i, exp in enumerate(expected):
            assert result[i].item() == pytest.approx(exp, abs=1e-5)

    def test_quadratic_poly(self):
        """Order 2: various quadratic polynomials."""
        # f(x,y) = 1 (constant)
        result = self._eval_cuda([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, [3.0], [4.0])
        assert result[0].item() == pytest.approx(1.0, abs=1e-5)

        # f(x,y) = x^2 + y^2
        result = self._eval_cuda([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 2, [3.0], [4.0])
        assert result[0].item() == pytest.approx(25.0, abs=1e-4)

        # f(x,y) = 2*x*y
        result = self._eval_cuda([0.0, 0.0, 0.0, 0.0, 2.0, 0.0], 2, [3.0], [4.0])
        assert result[0].item() == pytest.approx(24.0, abs=1e-4)

    def test_eval_at_zero(self):
        """All polynomials evaluated at (0, 0) should return c0."""
        for order in range(6):
            n = num_coeffs_for_order(order)
            coeffs = [float(i + 1) for i in range(n)]
            result = self._eval_cuda(coeffs, order, [0.0], [0.0])
            assert result[0].item() == pytest.approx(coeffs[0], abs=1e-5)

    def test_cross_validate_against_python_reference(self):
        """Compare CUDA results to Python reference for many random inputs."""
        torch.manual_seed(42)
        for order in range(5):
            n = num_coeffs_for_order(order)
            coeffs = torch.randn(n).tolist()
            x_vals = (torch.rand(50) * 2 - 1).tolist()  # [-1, 1]
            y_vals = (torch.rand(50) * 2 - 1).tolist()
            cuda_result = self._eval_cuda(coeffs, order, x_vals, y_vals)
            for i in range(len(x_vals)):
                ref = ref_eval_bivariate_poly(coeffs, order, x_vals[i], y_vals[i])
                assert cuda_result[i].item() == pytest.approx(
                    ref, abs=1e-4
                ), f"Mismatch at order={order}, i={i}: CUDA={cuda_result[i].item()}, ref={ref}"


# ===========================================================================
# 4. CUDA distort_camera_rays tests
#    Calls the actual CUDA kernel and cross-validates against Python reference.
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestDistortCameraRaysCUDA:
    """Test the CUDA distort_camera_rays kernel against Python reference."""

    @pytest.fixture(autouse=True)
    def _require_camera_wrappers(self):
        if not gsplat.has_camera_wrappers():
            pytest.skip("Camera wrappers not built (need BUILD_CAMERA_WRAPPERS=1)")

    @staticmethod
    def _distort_cuda(rays_list, h_poly, v_poly, h_inv=None, v_inv=None, inverse=False):
        """Distort rays using the CUDA kernel."""
        rays = torch.tensor(rays_list, dtype=torch.float32, device="cuda")
        if h_inv is None:
            h_inv = make_identity_horizontal_poly()
        if v_inv is None:
            v_inv = make_identity_vertical_poly()
        params = make_params(h_poly=h_poly, v_poly=v_poly, h_inv=h_inv, v_inv=v_inv)
        return distort_camera_rays_cuda(rays, params, inverse=inverse)

    def test_identity_on_axis_ray(self):
        """Identity poly on z-axis ray (0,0,1) should return (0,0,1)."""
        h = make_identity_horizontal_poly()
        v = make_identity_vertical_poly()
        result = self._distort_cuda([[0.0, 0.0, 1.0]], h, v)
        assert result[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert result[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert result[0, 2].item() == pytest.approx(1.0, abs=1e-6)

    def test_identity_on_negative_z_ray(self):
        """Identity poly on negative z-axis ray (0,0,-1)."""
        h = make_identity_horizontal_poly()
        v = make_identity_vertical_poly()
        result = self._distort_cuda([[0.0, 0.0, -1.0]], h, v)
        assert result[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert result[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert result[0, 2].item() == pytest.approx(-1.0, abs=1e-6)

    def test_identity_preserves_direction(self):
        """Identity distortion should preserve the normalized ray direction."""
        h = make_identity_horizontal_poly()
        v = make_identity_vertical_poly()
        rays = [
            [0.3, 0.4, 0.8],
            [0.1, 0.0, 1.0],
            [0.0, 0.2, 0.9],
            [-0.3, 0.2, 0.7],
            [0.5, -0.3, 0.6],
        ]
        result = self._distort_cuda(rays, h, v)
        for i, ray in enumerate(rays):
            length = math.sqrt(sum(c**2 for c in ray))
            expected = [c / length for c in ray]
            for j in range(3):
                assert result[i, j].item() == pytest.approx(
                    expected[j], abs=1e-5
                ), f"Component {j} mismatch for ray {ray}"

    def test_zero_poly_maps_to_z_axis(self):
        """All-zero polynomials map any ray to the z-axis (since sin(0)=0)."""
        h = make_zero_poly(1)
        v = make_zero_poly(1)
        result = self._distort_cuda([[0.5, 0.3, 0.8]], h, v)
        assert result[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert result[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert result[0, 2].item() == pytest.approx(1.0, abs=1e-6)

    def test_zero_poly_negative_z(self):
        """All-zero polynomials with negative z preserve z sign."""
        h = make_zero_poly(1)
        v = make_zero_poly(1)
        result = self._distort_cuda([[0.5, 0.3, -0.8]], h, v)
        assert result[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert result[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert result[0, 2].item() == pytest.approx(-1.0, abs=1e-6)

    def test_output_is_unit_length(self):
        """Distorted ray should be unit length when x^2+y^2 <= 1."""
        h = make_identity_horizontal_poly()
        v = make_identity_vertical_poly()
        rays = [
            [0.3, 0.4, 0.8],
            [0.1, 0.0, 1.0],
            [0.0, 0.2, 0.9],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.7],
        ]
        result = self._distort_cuda(rays, h, v)
        for i in range(len(rays)):
            length = torch.norm(result[i]).item()
            assert length == pytest.approx(
                1.0, abs=1e-5
            ), f"Non-unit length {length} for ray {rays[i]}"

    def test_clamp_prevents_nan(self):
        """When x^2 + y^2 > 1, the clamp should prevent NaN in sqrt."""
        # Order 0: constant ~pi/2 -> sin(pi/2)=1 for both -> x^2+y^2=2>1 -> z clamped to 0
        h = [math.pi / 2]
        v = [math.pi / 2]
        result = self._distort_cuda([[0.3, 0.3, 0.8]], h, v, h_inv=h, v_inv=v)
        assert not torch.isnan(result).any()
        assert result[0, 2].item() == pytest.approx(0.0, abs=1e-5)

    def test_known_constant_offset(self):
        """h_poly = [0.1, 1, 0] on z-axis -> x = sin(0.1), y = 0."""
        h = [0.1, 1.0, 0.0]
        v = [0.0, 0.0, 1.0]
        result = self._distort_cuda([[0.0, 0.0, 1.0]], h, v)
        assert result[0, 0].item() == pytest.approx(math.sin(0.1), abs=1e-5)
        assert result[0, 1].item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetric_horizontal_vertical_swap(self):
        """Swapping h/v polynomials should swap x/y in output for on-axis rays."""
        h = [0.1, 0.9, 0.05]
        v = [0.0, 0.0, 1.0]
        ray = [[0.0, 0.0, 1.0]]

        result_normal = self._distort_cuda(ray, h, v)
        result_swapped = self._distort_cuda(ray, v, h, h_inv=v, v_inv=h)

        assert result_normal[0, 0].item() == pytest.approx(
            result_swapped[0, 1].item(), abs=1e-5
        )
        assert result_normal[0, 1].item() == pytest.approx(
            result_swapped[0, 0].item(), abs=1e-5
        )

    def test_inverse_flag_uses_inverse_polynomials(self):
        """When inverse=True, the kernel should use the inverse polynomials."""
        h_fwd = [0.1, 1.0, 0.0]
        v_fwd = [0.0, 0.0, 1.0]
        h_inv = [-0.1, 1.0, 0.0]
        v_inv = [0.0, 0.0, 1.0]
        ray = [[0.0, 0.0, 1.0]]

        result_fwd = self._distort_cuda(
            ray, h_fwd, v_fwd, h_inv=h_inv, v_inv=v_inv, inverse=False
        )
        result_inv = self._distort_cuda(
            ray, h_fwd, v_fwd, h_inv=h_inv, v_inv=v_inv, inverse=True
        )

        # Forward: x = sin(0.1) ≈ 0.0998
        assert result_fwd[0, 0].item() == pytest.approx(math.sin(0.1), abs=1e-5)
        # Inverse: x = sin(-0.1) ≈ -0.0998
        assert result_inv[0, 0].item() == pytest.approx(math.sin(-0.1), abs=1e-5)

    def test_cross_validate_against_python_reference(self):
        """Compare CUDA distort_camera_rays to Python reference for random rays."""
        h = make_identity_horizontal_poly()
        v = make_identity_vertical_poly()
        torch.manual_seed(42)
        # Generate forward-looking rays (positive z)
        rays_list = []
        for _ in range(20):
            r = torch.randn(3)
            r[2] = abs(r[2]) + 0.5  # ensure positive z
            rays_list.append(r.tolist())

        result = self._distort_cuda(rays_list, h, v)
        for i, ray in enumerate(rays_list):
            ref = ref_distort_camera_ray(tuple(ray), h, v, 1, 1)
            for j in range(3):
                assert result[i, j].item() == pytest.approx(
                    ref[j], abs=1e-5
                ), f"Mismatch at ray {i}, component {j}"

    def test_cross_validate_nonidentity_poly(self):
        """Cross-validate CUDA vs Python for non-identity polynomials."""
        h = [0.05, 0.95, 0.02]
        v = [-0.02, 0.01, 0.98]
        torch.manual_seed(123)
        rays_list = []
        for _ in range(20):
            r = torch.randn(3)
            r[2] = abs(r[2]) + 0.5
            rays_list.append(r.tolist())

        result = self._distort_cuda(rays_list, h, v)
        h_order = ref_compute_order(len(h))
        v_order = ref_compute_order(len(v))
        for i, ray in enumerate(rays_list):
            ref = ref_distort_camera_ray(tuple(ray), h, v, h_order, v_order)
            for j in range(3):
                assert result[i, j].item() == pytest.approx(
                    ref[j], abs=1e-4
                ), f"Mismatch at ray {i}, component {j}: CUDA={result[i,j].item()}, ref={ref[j]}"

    def test_batch_consistency(self):
        """CUDA kernel should produce same result regardless of batch size."""
        h = [0.05, 1.0, 0.0]
        v = [0.0, 0.0, 1.0]
        ray = [0.3, 0.2, 0.9]

        # Single ray
        result_single = self._distort_cuda([ray], h, v)
        # Batch of 10 identical rays
        result_batch = self._distort_cuda([ray] * 10, h, v)

        for i in range(10):
            torch.testing.assert_close(
                result_single[0], result_batch[i], atol=1e-7, rtol=1e-7
            )

    def test_boundary_rays_asin_clamp(self):
        """Boundary rays where ray[i]/ray_length == 1.0 must not produce NaN.

        Under --use_fast_math, the ratio can exceed 1.0 by an ULP, making
        asin return NaN.  The clamp in ExternalDistortion.cuh and the Python
        ref prevents this.  Cross-validates CUDA against Python reference.
        """
        h = make_identity_horizontal_poly()
        v = make_identity_vertical_poly()

        boundary_rays = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1e-30],
            [0.0, 1.0, 1e-30],
            [-1.0, 0.0, 1e-30],
            [0.0, -1.0, 1e-30],
        ]

        cuda_result = self._distort_cuda(boundary_rays, h, v)
        assert torch.isfinite(
            cuda_result
        ).all(), "NaN/Inf in CUDA distort_camera_rays for boundary rays"

        for i, ray in enumerate(boundary_rays):
            ref = ref_distort_camera_ray(tuple(ray), h, v, 1, 1)
            for j in range(3):
                assert math.isfinite(
                    ref[j]
                ), f"NaN/Inf in Python ref for ray {ray}, component {j}"
                assert cuda_result[i, j].item() == pytest.approx(
                    ref[j], abs=1e-5
                ), f"Mismatch at ray {i}, component {j}"

    def test_asin_clamp_with_imprecise_sqrt(self):
        """Python ref asin clamp must prevent ValueError when sqrt is imprecise.

        Under --use_fast_math, CUDA's sqrt can return a value slightly smaller
        than the true result, making ray[i]/ray_length exceed 1.0.  We
        monkey-patch math.sqrt to simulate this (can't patch GPU code).
        """
        import unittest.mock

        h = make_identity_horizontal_poly()
        v = make_identity_vertical_poly()

        boundary_rays = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 1e-30),
            (0.0, 1.0, 1e-30),
            (-1.0, 0.0, 1e-30),
            (0.0, -1.0, 1e-30),
        ]

        original_sqrt = math.sqrt

        def fast_math_sqrt(x):
            """Simulate --use_fast_math: sqrt returns value slightly too small."""
            return original_sqrt(x) * (1.0 - 1e-7)

        # Precondition: patched sqrt makes ratio > 1.0 for boundary rays
        ray_length = fast_math_sqrt(sum(c**2 for c in boundary_rays[0]))
        ratio = boundary_rays[0][0] / ray_length
        assert (
            ratio > 1.0
        ), f"Precondition failed: patched sqrt should make ratio > 1.0, got {ratio}"
        with pytest.raises(ValueError):
            math.asin(ratio)

        # All boundary rays must produce finite output despite ratio > 1.0
        with unittest.mock.patch(
            "gsplat.cuda._torch_external_distortion.math.sqrt", fast_math_sqrt
        ):
            for ray in boundary_rays:
                result = ref_distort_camera_ray(ray, h, v, 1, 1)
                for j, val in enumerate(result):
                    assert math.isfinite(
                        val
                    ), f"NaN/Inf in component {j} for ray {ray}: got {val}"


# ===========================================================================
# 5. Camera model integration tests via BaseCameraModel.create()
#    Tests that external distortion is correctly threaded through the
#    camera wrapper infrastructure (project/unproject roundtrips).
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestCameraWithExternalDistortion:
    """Test external distortion integrated into camera models via BaseCameraModel.create()."""

    @pytest.fixture(autouse=True)
    def _require_camera_wrappers(self):
        if not gsplat.has_camera_wrappers():
            pytest.skip("Camera wrappers not built (need BUILD_CAMERA_WRAPPERS=1)")

    @staticmethod
    def _create_pinhole_camera(width=640, height=480, external_distortion_coeffs=None):
        from gsplat.cuda._wrapper import create_camera_model

        focal_lengths = torch.tensor(
            [[320.0, 320.0]], dtype=torch.float32, device="cuda"
        )
        principal_points = torch.tensor(
            [[320.0, 240.0]], dtype=torch.float32, device="cuda"
        )
        return create_camera_model(
            width=width,
            height=height,
            camera_model="pinhole",
            principal_points=principal_points,
            focal_lengths=focal_lengths,
            external_distortion_coeffs=external_distortion_coeffs,
        )

    def test_no_distortion_roundtrip(self):
        """Project -> unproject roundtrip without distortion should recover direction."""
        cam = self._create_pinhole_camera()
        # Forward-looking rays
        rays = torch.tensor(
            [[[0.1, 0.05, 1.0], [0.0, 0.0, 1.0], [-0.1, 0.1, 1.0]]],
            dtype=torch.float32,
            device="cuda",
        )
        img_pts, valid_proj = cam.camera_ray_to_image_point(rays)
        assert valid_proj.all()

        rays_back, valid_unproj = cam.image_point_to_camera_ray(img_pts)
        assert valid_unproj.all()

        # Normalize both for comparison
        rays_norm = rays / rays.norm(dim=-1, keepdim=True)
        rays_back_norm = rays_back / rays_back.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(rays_norm, rays_back_norm, atol=1e-4, rtol=1e-4)

    def test_identity_distortion_matches_no_distortion(self):
        """Identity external distortion should produce same results as no distortion."""
        cam_none = self._create_pinhole_camera()
        identity_params = make_params(
            h_poly=make_identity_horizontal_poly(),
            v_poly=make_identity_vertical_poly(),
            h_inv=make_identity_horizontal_poly(),
            v_inv=make_identity_vertical_poly(),
        )
        cam_identity = self._create_pinhole_camera(
            external_distortion_coeffs=identity_params
        )

        rays = torch.tensor(
            [[[0.1, 0.05, 1.0], [0.0, 0.0, 1.0], [-0.05, 0.03, 1.0]]],
            dtype=torch.float32,
            device="cuda",
        )

        img_none, _ = cam_none.camera_ray_to_image_point(rays)
        img_identity, _ = cam_identity.camera_ray_to_image_point(rays)
        torch.testing.assert_close(img_none, img_identity, atol=1e-4, rtol=1e-4)

    def test_nonzero_distortion_changes_projection(self):
        """A non-trivial distortion should change the projected image points."""
        cam_none = self._create_pinhole_camera()
        perturbed_params = make_params(
            h_poly=[0.05, 1.0, 0.0],
            v_poly=[0.0, 0.0, 1.0],
            h_inv=[-0.05, 1.0, 0.0],
            v_inv=[0.0, 0.0, 1.0],
        )
        cam_perturbed = self._create_pinhole_camera(
            external_distortion_coeffs=perturbed_params
        )

        rays = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]], dtype=torch.float32, device="cuda"
        )

        img_none, _ = cam_none.camera_ray_to_image_point(rays)
        img_perturbed, _ = cam_perturbed.camera_ray_to_image_point(rays)

        diff = (img_perturbed - img_none).abs().max().item()
        assert (
            diff > 1.0
        ), f"Expected visible pixel shift from 0.05 rad offset, got max diff {diff}"

    def test_distortion_undistortion_roundtrip(self):
        """Project with distortion -> unproject should approximately recover direction.

        This tests that the forward distortion (in projection) and inverse distortion
        (in unprojection) are consistently applied through the camera model.
        """
        params = make_params(
            h_poly=[0.02, 1.0, 0.0],
            v_poly=[0.0, 0.0, 1.0],
            h_inv=[-0.02, 1.0, 0.0],
            v_inv=[0.0, 0.0, 1.0],
        )
        cam = self._create_pinhole_camera(external_distortion_coeffs=params)

        rays = torch.tensor(
            [[[0.05, 0.03, 1.0], [0.0, 0.0, 1.0], [-0.05, 0.05, 1.0]]],
            dtype=torch.float32,
            device="cuda",
        )

        img_pts, valid_proj = cam.camera_ray_to_image_point(rays)
        assert valid_proj.all()

        rays_back, valid_unproj = cam.image_point_to_camera_ray(img_pts)
        assert valid_unproj.all()

        rays_norm = rays / rays.norm(dim=-1, keepdim=True)
        rays_back_norm = rays_back / rays_back.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(rays_norm, rays_back_norm, atol=1e-5, rtol=1e-5)

    def test_zero_poly_projects_to_center(self):
        """All-zero polynomials map every ray to the z-axis -> principal point."""
        zero_params = make_params(
            h_poly=make_zero_poly(1),
            v_poly=make_zero_poly(1),
            h_inv=make_zero_poly(1),
            v_inv=make_zero_poly(1),
        )
        cam = self._create_pinhole_camera(external_distortion_coeffs=zero_params)

        rays = torch.tensor(
            [[[0.3, 0.2, 0.8], [-0.1, 0.5, 0.6], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device="cuda",
        )

        img_pts, valid = cam.camera_ray_to_image_point(rays)
        # All rays distorted to z-axis -> all project to principal point (320, 240)
        for i in range(3):
            assert img_pts[0, i, 0].item() == pytest.approx(320.0, abs=1.0)
            assert img_pts[0, i, 1].item() == pytest.approx(240.0, abs=1.0)

    def test_higher_order_poly(self):
        """Order-2 identity-like polynomial should work through camera model."""
        # Order-2 identity: f(phi,theta) = phi for h, f(phi,theta) = theta for v
        h = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        params = make_params(h_poly=h, v_poly=v, h_inv=h, v_inv=v)
        cam = self._create_pinhole_camera(external_distortion_coeffs=params)

        rays = torch.tensor(
            [[[0.1, 0.05, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32, device="cuda"
        )
        img_pts, valid = cam.camera_ray_to_image_point(rays)
        assert valid.all()
        assert not torch.isnan(img_pts).any()


# ===========================================================================
# 6. Integration tests through the full rendering pipeline (3DGUT)
# ===========================================================================


@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = gsplat._helper.load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "viewmats": viewmats,
        "Ks": Ks,
        "colors": colors,
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestRenderingWithExternalDistortion:
    """Integration tests: external distortion through the full rendering pipeline."""

    @pytest.fixture(autouse=True)
    def _require_3dgut(self):
        if not gsplat.has_3dgut():
            pytest.skip("3DGUT support isn't built in")

    @staticmethod
    def _render(test_data, external_distortion_coeffs=None):
        from gsplat.rendering import (
            FThetaCameraDistortionParameters,
            FThetaPolynomialType,
            rasterization,
        )

        C = test_data["Ks"].shape[0]
        colors = test_data["colors"].repeat(C, 1, 1)

        ftheta_coeffs = FThetaCameraDistortionParameters(
            reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=(
                0.0,
                8.4335003e-03,
                2.3174282e-06,
                -5.0478608e-08,
                6.1392608e-10,
                -1.7447865e-12,
            ),
            angle_to_pixeldist_poly=(
                0.0,
                118.43232,
                -2.562147,
                6.317949,
                -10.41861,
                3.6694396,
            ),
            max_angle=1000,
            linear_cde=(9.9968284e-01, 1.8735906e-05, 1.7659619e-05),
        )

        renders, alphas, meta = rasterization(
            means=test_data["means"],
            quats=test_data["quats"],
            scales=test_data["scales"],
            opacities=test_data["opacities"],
            colors=colors,
            viewmats=test_data["viewmats"],
            Ks=test_data["Ks"],
            width=test_data["width"],
            height=test_data["height"],
            render_mode="RGB",
            camera_model="ftheta",
            packed=False,
            ftheta_coeffs=ftheta_coeffs,
            external_distortion_coeffs=external_distortion_coeffs,
            with_ut=True,
            with_eval3d=True,
        )
        return renders, alphas

    def test_no_external_distortion(self, test_data):
        """Rendering with external_distortion_coeffs=None should succeed."""
        renders, alphas = self._render(test_data, external_distortion_coeffs=None)
        C = test_data["Ks"].shape[0]
        assert renders.shape == (C, test_data["height"], test_data["width"], 3)
        assert not torch.isnan(renders).any()
        assert not torch.isinf(renders).any()

    def test_identity_distortion_matches_no_distortion(self, test_data):
        """Identity external distortion should produce output close to no distortion."""
        renders_none, _ = self._render(test_data, external_distortion_coeffs=None)

        identity_params = make_params(
            h_poly=make_identity_horizontal_poly(),
            v_poly=make_identity_vertical_poly(),
            h_inv=make_identity_horizontal_poly(),
            v_inv=make_identity_vertical_poly(),
        )
        renders_identity, _ = self._render(
            test_data, external_distortion_coeffs=identity_params
        )

        assert not torch.isnan(renders_identity).any()
        diff = (renders_identity - renders_none).abs()
        assert diff.mean() < 0.05, f"Mean pixel difference too large: {diff.mean():.4f}"

    def test_nonzero_distortion_changes_output(self, test_data):
        """A non-trivial distortion should produce different output than no distortion."""
        renders_none, _ = self._render(test_data, external_distortion_coeffs=None)

        perturbed_params = make_params(
            h_poly=[0.05, 1.0, 0.0],
            v_poly=[0.0, 0.0, 1.0],
            h_inv=[-0.05, 1.0, 0.0],
            v_inv=[0.0, 0.0, 1.0],
        )
        renders_perturbed, _ = self._render(
            test_data, external_distortion_coeffs=perturbed_params
        )

        assert not torch.isnan(renders_perturbed).any()
        diff = (renders_perturbed - renders_none).abs()
        assert (
            diff.mean() > 1e-4
        ), f"Expected visible difference from 0.05 rad offset, got mean diff {diff.mean():.6f}"

    def test_zero_poly_distortion(self, test_data):
        """All-zero polynomials should not crash (maps everything to z-axis)."""
        zero_params = make_params(
            h_poly=make_zero_poly(1),
            v_poly=make_zero_poly(1),
            h_inv=make_zero_poly(1),
            v_inv=make_zero_poly(1),
        )
        renders, alphas = self._render(
            test_data, external_distortion_coeffs=zero_params
        )
        assert not torch.isnan(renders).any()
        assert not torch.isinf(renders).any()

    def test_higher_order_poly(self, test_data):
        """Order-2 polynomial should render without errors."""
        h = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        params = make_params(h_poly=h, v_poly=v, h_inv=h, v_inv=v)
        renders, alphas = self._render(test_data, external_distortion_coeffs=params)
        assert renders.shape == (
            test_data["Ks"].shape[0],
            test_data["height"],
            test_data["width"],
            3,
        )
        assert not torch.isnan(renders).any()

    def test_backward_reference_poly(self, test_data):
        """Rendering with BACKWARD reference polynomial should succeed."""
        params = make_params(
            h_poly=make_identity_horizontal_poly(),
            v_poly=make_identity_vertical_poly(),
            ref_poly=ExternalDistortionReferencePolynomial.BACKWARD,
        )
        renders, alphas = self._render(test_data, external_distortion_coeffs=params)
        assert not torch.isnan(renders).any()
