/**
 * @file ExternalDistortionWrappers.h
 * @brief Host-callable wrappers for external distortion CUDA functions.
 *
 * These wrappers expose the device-only functions in ExternalDistortion.cuh
 * to Python for testing. Guarded by GSPLAT_BUILD_CAMERA_WRAPPERS.
 */

#pragma once

#if GSPLAT_BUILD_CAMERA_WRAPPERS

#include <torch/extension.h>
#include "ExternalDistortion.h"

namespace gsplat::extdist {

/**
 * @brief Distort or undistort camera rays using the bivariate windshield model.
 *
 * Launches a CUDA kernel that calls BivariateWindshieldModel::distort_camera_ray()
 * on each input ray.
 *
 * @param rays Input rays [N, 3] (float32, CUDA)
 * @param params Bivariate windshield model parameters (polynomial tensors must be CUDA)
 * @param inverse If false, apply forward distortion. If true, apply inverse (undistort).
 * @return Distorted rays [N, 3] (float32, CUDA)
 */
torch::Tensor distort_camera_rays(
    const torch::Tensor& rays,
    const BivariateWindshieldModelParameters& params,
    bool inverse);

/**
 * @brief Evaluate a 2D bivariate polynomial at (x, y) points.
 *
 * Launches a CUDA kernel that calls eval_bivariate_poly() on each input pair.
 *
 * @param x Input x values [N] (float32, CUDA)
 * @param y Input y values [N] (float32, CUDA)
 * @param poly_coeffs Polynomial coefficients [num_coeffs] (float32, CUDA)
 * @param order Polynomial order
 * @return Evaluated values [N] (float32, CUDA)
 */
torch::Tensor eval_bivariate_poly_wrapper(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& poly_coeffs,
    int64_t order);

/**
 * @brief Torch-op-compatible wrapper for distort_camera_rays.
 *
 * Accepts individual polynomial tensors instead of the BivariateWindshieldModelParameters
 * struct, allowing registration as a torch custom op via TORCH_LIBRARY.
 *
 * @param rays Input rays [N, 3] (float32, CUDA)
 * @param h_poly Horizontal forward polynomial coefficients (float32, CUDA)
 * @param v_poly Vertical forward polynomial coefficients (float32, CUDA)
 * @param h_inv_poly Horizontal inverse polynomial coefficients (float32, CUDA)
 * @param v_inv_poly Vertical inverse polynomial coefficients (float32, CUDA)
 * @param reference_poly Reference polynomial type (int cast of ReferencePolynomialType)
 * @param inverse If false, apply forward distortion. If true, apply inverse (undistort).
 * @return Distorted rays [N, 3] (float32, CUDA)
 */
torch::Tensor distort_camera_rays_torch_op(
    const torch::Tensor& rays,
    const torch::Tensor& h_poly,
    const torch::Tensor& v_poly,
    const torch::Tensor& h_inv_poly,
    const torch::Tensor& v_inv_poly,
    int64_t reference_poly,
    bool inverse);

} // namespace gsplat::extdist

#endif // GSPLAT_BUILD_CAMERA_WRAPPERS
