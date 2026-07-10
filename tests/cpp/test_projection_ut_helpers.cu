/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "Config.h"

#if GSPLAT_BUILD_3DGUT
#    include "Cameras.cuh"
#    include "Lidars.cuh"
#    include "Utils.cuh"

namespace
{
constexpr int kRotationColumnCases = 4096;
constexpr int kComponentsPerCase   = 9; // 3 columns x 3 rows
constexpr int kMaxGlmUlpDrift      = 2;
// A rotation-matrix entry is |x| <= 1, so a single FMA-contraction difference vs
// glm is bounded by the rounding of its unit-magnitude intermediates: at most a
// couple of ULP of 1.0, independent of the result's own magnitude. Measured over
// the full sweep on both JIT and offline SASS (sm_89) the worst gap is exactly
// 2^-23 (~1.19e-7, one ULP of 1.0); we cap at 2^-22 (~2.38e-7, two ULP of 1.0).
// This absolute cap is what stays meaningful once the near-zero diagonal
// 1 - 2(a^2 + b^2) collapses the ULP scale (the rolled runtime-column form drifts
// up to ~2048 ULP there while staying inside this bound). It is deliberately ~4x
// tighter than the old 1e-6, which was loose enough to accept mid-magnitude
// entries many ULP off (e.g. 0.25 vs 0x3e800010: 16 ULP, 4.77e-7).
constexpr float kMaxGlmAbsDrift    = 0x1p-22f; // two ULP of 1.0 ~= 2.38e-7
constexpr float kHelperTolerance   = 1e-6f;

float f32_from_bits(uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t lcg_next(uint32_t state)
{
    return state * 1'664'525u + 1'013'904'223u;
}

std::vector<float> make_quaternion_cases()
{
    std::vector<float> quats(kRotationColumnCases * 4);
    auto store = [&](int index, float w, float x, float y, float z)
    {
        quats[index * 4 + 0] = w;
        quats[index * 4 + 1] = x;
        quats[index * 4 + 2] = y;
        quats[index * 4 + 3] = z;
    };

    store(0, 1.f, 0.f, 0.f, 0.f);
    store(1, 0.125f, -0.75f, 0.5f, -0.375f);
    store(2, f32_from_bits(0x3F3504F3u), f32_from_bits(0xBF22E9D9u), 0.3125f, -0.6875f);
    store(3, f32_from_bits(0x3EAAAAABu), f32_from_bits(0x3F000001u), f32_from_bits(0xBF4CCCCDu), 0.0625f);

    uint32_t state = 0x4D595DF4u;
    for(int i = 4; i < kRotationColumnCases; ++i)
    {
        float values[4];
        for(float &value: values)
        {
            state                         = lcg_next(state);
            const int32_t signed_mantissa = static_cast<int32_t>((state >> 8) & 0xFFFFu) - 32768;
            value                         = static_cast<float>(signed_mantissa) / 32768.f;
        }
        if(values[0] == 0.f && values[1] == 0.f && values[2] == 0.f && values[3] == 0.f)
        {
            values[0] = 1.f;
        }
        store(i, values[0], values[1], values[2], values[3]);
    }
    return quats;
}

// Map float bits onto a monotonic unsigned axis so adjacent representable values
// are one step apart regardless of sign. Negative values are order-reversed via
// two's complement (~bits + 1); non-negative values get the sign bit set so they
// sort above every negative. +0 and -0 both map to 0x80000000u, so they are zero
// ULP apart. Keeping both the keys and their distance unsigned avoids the signed
// overflow that made the previous maximum unreliable across signs.
__device__ uint32_t ordered_float_bits(float value)
{
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits + 1u : bits | 0x80000000u;
}

__device__ uint32_t ulp_distance(float lhs, float rhs)
{
    const uint32_t lhs_ordered = ordered_float_bits(lhs);
    const uint32_t rhs_ordered = ordered_float_bits(rhs);
    return lhs_ordered > rhs_ordered ? lhs_ordered - rhs_ordered : rhs_ordered - lhs_ordered;
}

// Documented contract for a compact rotation column vs glm::mat3_cast: within
// kMaxGlmUlpDrift ULP of the value (holds away from cancellation), OR within
// kMaxGlmAbsDrift absolute -- the physical one-FMA-of-a-unit-entry bound, which is
// what remains meaningful once the near-zero diagonal collapses the ULP scale.
__device__ bool within_glm_contract(float value, float reference)
{
    return ulp_distance(value, reference) <= static_cast<uint32_t>(kMaxGlmUlpDrift)
        || fabsf(value - reference) <= kMaxGlmAbsDrift;
}

__global__ void rotation_matrix_column_oracle_kernel(
    const float *__restrict__ raw_quats, int num_quats, int *__restrict__ glm_contract_violations
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_quats * kComponentsPerCase)
    {
        return;
    }

    const int quat_index = idx / kComponentsPerCase;
    const int rem        = idx - quat_index * kComponentsPerCase;
    const unsigned col   = static_cast<unsigned>(rem / 3);
    const int row        = rem % 3;

    glm::fquat q{
        raw_quats[quat_index * 4 + 0],
        raw_quats[quat_index * 4 + 1],
        raw_quats[quat_index * 4 + 2],
        raw_quats[quat_index * 4 + 3]
    };
    q *= rsqrtf(glm::dot(q, q));

    // Match the production instantiations: the fully-unrolled couple loop only
    // ever calls the unrolled form with compile-time columns, so its switch
    // folds to a single case; the rolled path uses a runtime column.
    glm::fvec3 unrolled_column;
    switch(col)
    {
    case 0u: unrolled_column = rotation_matrix_column<false>(q, 0u); break;
    case 1u: unrolled_column = rotation_matrix_column<false>(q, 1u); break;
    default: unrolled_column = rotation_matrix_column<false>(q, 2u); break;
    }
    const glm::fvec3 rolled_column = rotation_matrix_column<true>(q, col);
    const glm::fvec3 glm_column    = glm::mat3_cast(q)[col];

    // Both shipped lowerings -- the unrolled constant column (cameras) and the
    // rolled runtime column (LiDAR) -- must match glm::mat3_cast within the
    // documented tolerance. They are NOT bit-identical to each other: JIT and
    // offline ptxas contract these FMAs differently, so each lowering is held to
    // the glm contract instead of to the other.
    if(!within_glm_contract(unrolled_column[row], glm_column[row]))
    {
        atomicAdd(glm_contract_violations, 1);
    }
    if(!within_glm_contract(rolled_column[row], glm_column[row]))
    {
        atomicAdd(glm_contract_violations, 1);
    }
}

// Pin the ULP metric on adversarial inputs so a future edit to ordered_float_bits
// cannot silently reintroduce the signed overflow. Distances are checked against
// hand-computed unsigned values that span the full axis (opposite signs, +/-inf).
__global__ void ulp_distance_special_values_kernel(int *__restrict__ mismatch_count)
{
    auto check = [&](uint32_t lhs_bits, uint32_t rhs_bits, uint32_t expected)
    {
        if(ulp_distance(__uint_as_float(lhs_bits), __uint_as_float(rhs_bits)) != expected)
        {
            atomicAdd(mismatch_count, 1);
        }
    };

    constexpr uint32_t kPosZero   = 0x00000000u;
    constexpr uint32_t kNegZero   = 0x80000000u;
    constexpr uint32_t kPosDenorm = 0x00000001u; // smallest positive subnormal
    constexpr uint32_t kNegDenorm = 0x80000001u; // smallest negative subnormal
    constexpr uint32_t kOne       = 0x3F800000u; // 1.0f
    constexpr uint32_t kOneNext   = 0x3F800001u; // nextafter(1.0f, +inf)
    constexpr uint32_t kNegOne    = 0xBF800000u; // -1.0f
    constexpr uint32_t kPosInf    = 0x7F800000u;
    constexpr uint32_t kNegInf    = 0xFF800000u;
    constexpr uint32_t kQuietNaN  = 0x7FC00000u;

    check(kPosZero, kNegZero, 0u);          // signed zeros coincide
    check(kPosZero, kPosDenorm, 1u);        // zero to smallest subnormal
    check(kPosDenorm, kNegDenorm, 2u);      // smallest subnormals straddle zero
    check(kOne, kOneNext, 1u);              // adjacent same-sign values
    check(kOne, kNegOne, 0x7F000000u);      // opposite signs, no signed overflow
    check(kPosInf, kOne, 0x40000000u);      // finite value to +inf
    check(kPosInf, kNegInf, 0xFF000000u);   // +inf to -inf spans the whole axis
    check(kQuietNaN, kQuietNaN, 0u);        // identical NaN payloads coincide
    check(kQuietNaN, kPosInf, 0x00400000u); // quiet NaN sits just above +inf
}

__device__ bool nearly_equal(float lhs, float rhs, float tolerance = kHelperTolerance)
{
    return fabsf(lhs - rhs) <= tolerance;
}

__device__ void expect_near(float lhs, float rhs, int *mismatch_count)
{
    if(!nearly_equal(lhs, rhs))
    {
        atomicAdd(mismatch_count, 1);
    }
}

__device__ void expect_vec3_near(const glm::fvec3 &lhs, const glm::fvec3 &rhs, int *mismatch_count)
{
    expect_near(lhs.x, rhs.x, mismatch_count);
    expect_near(lhs.y, rhs.y, mismatch_count);
    expect_near(lhs.z, rhs.z, mismatch_count);
}

__global__ void sigma_points_couple_oracle_kernel(int *__restrict__ mismatch_count)
{
    const glm::fvec3 mean{10.f, 20.f, 30.f};
    const glm::fvec3 scale{2.f, 3.f, 5.f};
    const float sigma_scale = 0.25f;
    const SigmaPointsProvider provider{
        glm::fquat{1.f, 0.f, 0.f, 0.f}
    };

    const SigmaPointsCouple center = provider.sigma_points_couple<false>(0u, mean, scale, sigma_scale);
    expect_vec3_near(center.points[0], mean, mismatch_count);
    expect_vec3_near(center.points[1], mean, mismatch_count);

    for(unsigned couple = 1u; couple <= 3u; ++couple)
    {
        const unsigned dim = couple - 1u;
        glm::fvec3 delta{0.f, 0.f, 0.f};
        delta[dim] = sigma_scale * scale[dim];

        const SigmaPointsCouple compact = provider.sigma_points_couple<false>(couple, mean, scale, sigma_scale);
        const SigmaPointsCouple rolled  = provider.sigma_points_couple<true>(couple, mean, scale, sigma_scale);
        expect_vec3_near(compact.points[0], mean + delta, mismatch_count);
        expect_vec3_near(compact.points[1], mean - delta, mismatch_count);
        expect_vec3_near(rolled.points[0], compact.points[0], mismatch_count);
        expect_vec3_near(rolled.points[1], compact.points[1], mismatch_count);
    }
}

__global__ void packed_covariance_helpers_oracle_kernel(int *__restrict__ mismatch_count)
{
    glm::fvec3 covar{4.f, 1.f, 9.f}; // symmetric 2x2: [[4, 1], [1, 9]]
    expect_near(gsplat::symmetric_cov2d_det(covar), 35.f, mismatch_count);

    float compensation       = 0.f;
    const float det_blur     = gsplat::add_blur(0.5f, covar, compensation);
    const float inv_det_blur = 1.f / det_blur;
    expect_near(covar.x, 4.5f, mismatch_count);
    expect_near(covar.y, 1.f, mismatch_count);
    expect_near(covar.z, 9.5f, mismatch_count);
    expect_near(det_blur, 41.75f, mismatch_count);
    expect_near(inv_det_blur, 1.f / 41.75f, mismatch_count);

    const glm::fvec3 inv = gsplat::inverse_symmetric_cov2d(covar, inv_det_blur);
    expect_near(inv.x, 9.5f / 41.75f, mismatch_count);
    expect_near(inv.y, -1.f / 41.75f, mismatch_count);
    expect_near(inv.z, 4.5f / 41.75f, mismatch_count);
    expect_near(compensation, sqrtf(35.f / 41.75f), mismatch_count);

    glm::fvec3 recovered_covar{-0.1f, 0.f, 0.1f};
    float recovered_compensation  = 0.f;
    const float recovered_det     = gsplat::add_blur(0.3f, recovered_covar, recovered_compensation);
    const float recovered_inv_det = 1.f / recovered_det;
    expect_near(recovered_covar.x, 0.2f, mismatch_count);
    expect_near(recovered_covar.z, 0.4f, mismatch_count);
    expect_near(recovered_det, 0.08f, mismatch_count);
    const glm::fvec3 recovered_inv = gsplat::inverse_symmetric_cov2d(recovered_covar, recovered_inv_det);
    expect_near(recovered_inv.x, 5.f, mismatch_count);
    expect_near(recovered_inv.y, 0.f, mismatch_count);
    expect_near(recovered_inv.z, 2.5f, mismatch_count);
}

__global__ void lidar_angle_scale_oracle_kernel(int *__restrict__ mismatch_count)
{
    constexpr int kScale = RowOffsetStructuredSpinningLidarModelParametersExtDevice::ANGLE_TO_PIXEL_SCALING_FACTOR;
    static_assert(kScale > 0, "LiDAR angle scale must reject zero");
    static_assert((kScale & (kScale - 1)) == 0, "LiDAR angle scale must be a power of two");

    const float angles[]     = {0.f, 0.125f, -0.25f, 1.5f, -3.0f};
    constexpr float kToAngle = 1.f / kScale;
    for(float angle: angles)
    {
        const float recovered = (angle * kScale) * kToAngle;
        if(__float_as_uint(recovered) != __float_as_uint(angle))
        {
            atomicAdd(mismatch_count, 1);
        }
    }
}
} // namespace

// Shared fixture so every case skips cleanly on a machine that compiled the CUDA
// toolkit but has no usable device, matching the other native CUDA tests. The
// launches below all touch the device, so gating them individually would leave
// the non-first cases failing instead of skipping.
class ProjectionUTHelpersTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(!torch::cuda::is_available())
        {
            GTEST_SKIP() << "CUDA runtime is not available";
        }
    }
};

TEST_F(ProjectionUTHelpersTest, RotationMatrixColumnLoweringsStayWithinGlmTolerance)
{
    const std::vector<float> quats = make_quaternion_cases();
    at::Tensor quats_cpu           = at::empty({kRotationColumnCases, 4}, at::TensorOptions().dtype(at::kFloat));
    std::memcpy(quats_cpu.data_ptr<float>(), quats.data(), quats.size() * sizeof(float));
    at::Tensor quats_cuda = quats_cpu.to(at::kCUDA);

    at::Tensor glm_contract_violations = at::zeros({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));

    constexpr int threads = 256;
    const int elements    = kRotationColumnCases * kComponentsPerCase;
    const int blocks      = (elements + threads - 1) / threads;
    rotation_matrix_column_oracle_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        quats_cuda.const_data_ptr<float>(), kRotationColumnCases, glm_contract_violations.data_ptr<int>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));

    const int violations = glm_contract_violations.cpu().item<int>();
    EXPECT_EQ(
        violations, 0
    ) << "A shipped rotation_matrix_column lowering exceeded both the ULP and absolute tolerance vs glm::mat3_cast";
}

TEST_F(ProjectionUTHelpersTest, UlpDistanceHandlesSpecialValues)
{
    at::Tensor mismatch_count = at::zeros({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
    ulp_distance_special_values_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(mismatch_count.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    EXPECT_EQ(mismatch_count.cpu().item<int>(), 0)
        << "ULP distance metric disagreed with hand-computed special-value cases";
}

TEST_F(ProjectionUTHelpersTest, SigmaPointCouplesMatchHandComputableAxes)
{
    at::Tensor mismatch_count = at::zeros({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
    sigma_points_couple_oracle_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(mismatch_count.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    EXPECT_EQ(mismatch_count.cpu().item<int>(), 0);
}

TEST_F(ProjectionUTHelpersTest, PackedCovarianceHelpersMatchHandComputableInputs)
{
    at::Tensor mismatch_count = at::zeros({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
    packed_covariance_helpers_oracle_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        mismatch_count.data_ptr<int>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    EXPECT_EQ(mismatch_count.cpu().item<int>(), 0);
}

TEST_F(ProjectionUTHelpersTest, LidarAngleScaleRoundTripsExactly)
{
    at::Tensor mismatch_count = at::zeros({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
    lidar_angle_scale_oracle_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(mismatch_count.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    EXPECT_EQ(mismatch_count.cpu().item<int>(), 0);
}
#endif // GSPLAT_BUILD_3DGUT
