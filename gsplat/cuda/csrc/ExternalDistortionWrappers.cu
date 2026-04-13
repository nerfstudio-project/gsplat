/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file ExternalDistortionWrappers.cu
 * @brief CUDA kernels wrapping ExternalDistortion.cuh device functions for Python testing.
 *
 * These wrappers allow Python tests to call the actual CUDA implementations of
 * eval_bivariate_poly() and distort_camera_ray(), ensuring regression coverage
 * of the device code.
 */

#if GSPLAT_BUILD_CAMERA_WRAPPERS

#include "ExternalDistortionWrappers.h"
#include "ExternalDistortion.cuh"
#include <c10/cuda/CUDAStream.h>

namespace gsplat::extdist {

// ---------------------------------------------------------------------------
// eval_bivariate_poly kernel
// ---------------------------------------------------------------------------

__global__ void eval_bivariate_poly_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ poly_coeffs,
    int32_t order,
    float* __restrict__ result,
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    result[idx] = gsplat::extdist::eval_bivariate_poly(poly_coeffs, order, x[idx], y[idx]);
}

torch::Tensor eval_bivariate_poly_wrapper(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& poly_coeffs,
    int64_t order)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(poly_coeffs.is_cuda(), "poly_coeffs must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(poly_coeffs.dtype() == torch::kFloat32, "poly_coeffs must be float32");
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have same number of elements");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(poly_coeffs.is_contiguous(), "poly_coeffs must be contiguous");

    int64_t N = x.numel();
    auto result = torch::empty_like(x);

    if (N == 0) return result;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());

    int threads = 256;
    int64_t blocks = (N + threads - 1) / threads;

    eval_bivariate_poly_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        poly_coeffs.data_ptr<float>(),
        order,
        result.data_ptr<float>(),
        N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "eval_bivariate_poly_kernel failed: ", cudaGetErrorString(err));

    return result;
}

// ---------------------------------------------------------------------------
// distort_camera_rays kernel
// ---------------------------------------------------------------------------

__global__ void distort_camera_rays_kernel(
    const float* __restrict__ rays,        // [N, 3]
    const float* __restrict__ h_poly,
    const float* __restrict__ v_poly,
    int32_t h_order,
    int32_t v_order,
    float* __restrict__ result,            // [N, 3]
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    glm::fvec3 ray(rays[idx * 3 + 0], rays[idx * 3 + 1], rays[idx * 3 + 2]);

    glm::fvec3 distorted = BivariateWindshieldModel::distort_camera_ray(
        ray, h_poly, v_poly, h_order, v_order);

    result[idx * 3 + 0] = distorted.x;
    result[idx * 3 + 1] = distorted.y;
    result[idx * 3 + 2] = distorted.z;
}

torch::Tensor distort_camera_rays(
    const torch::Tensor& rays,
    const BivariateWindshieldModelParameters& params,
    bool inverse)
{
    TORCH_CHECK(rays.is_cuda(), "rays must be a CUDA tensor");
    TORCH_CHECK(rays.dtype() == torch::kFloat32, "rays must be float32");
    TORCH_CHECK(rays.dim() >= 1 && rays.size(-1) == 3, "rays must have shape [..., 3]");

    auto rays_contig = rays.contiguous();
    int64_t N = rays_contig.numel() / 3;

    auto result = torch::empty_like(rays_contig);

    if (N == 0) return result;

    // Select forward or inverse polynomials
    const torch::Tensor& h_poly = inverse ? params.horizontal_poly_inverse : params.horizontal_poly;
    const torch::Tensor& v_poly = inverse ? params.vertical_poly_inverse : params.vertical_poly;

    TORCH_CHECK(h_poly.is_cuda(), "horizontal polynomial must be a CUDA tensor");
    TORCH_CHECK(v_poly.is_cuda(), "vertical polynomial must be a CUDA tensor");

    auto h_poly_contig = h_poly.contiguous();
    auto v_poly_contig = v_poly.contiguous();

    int32_t h_order = compute_order(h_poly_contig.size(-1));
    int32_t v_order = compute_order(v_poly_contig.size(-1));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(rays.device().index());

    int threads = 256;
    int64_t blocks = (N + threads - 1) / threads;

    distort_camera_rays_kernel<<<blocks, threads, 0, stream>>>(
        rays_contig.data_ptr<float>(),
        h_poly_contig.data_ptr<float>(),
        v_poly_contig.data_ptr<float>(),
        h_order,
        v_order,
        result.data_ptr<float>(),
        N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "distort_camera_rays_kernel failed: ", cudaGetErrorString(err));

    return result;
}

torch::Tensor distort_camera_rays_torch_op(
    const torch::Tensor& rays,
    const torch::Tensor& h_poly,
    const torch::Tensor& v_poly,
    const torch::Tensor& h_inv_poly,
    const torch::Tensor& v_inv_poly,
    int64_t reference_poly,
    bool inverse)
{
    BivariateWindshieldModelParameters params;
    params.horizontal_poly = h_poly;
    params.vertical_poly = v_poly;
    params.horizontal_poly_inverse = h_inv_poly;
    params.vertical_poly_inverse = v_inv_poly;
    params.reference_poly = static_cast<ReferencePolynomialType>(reference_poly);
    return distort_camera_rays(rays, params, inverse);
}

} // namespace gsplat::extdist

#endif // GSPLAT_BUILD_CAMERA_WRAPPERS
