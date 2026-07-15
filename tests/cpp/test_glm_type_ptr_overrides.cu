/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <torch/cuda.h>

#include "Common.h"

// Device-side regression coverage for the gsplat GLM pointer overloads in
// GlmTypePtrOverrides.cuh. The aligned vector-load branches are easy to break
// with an index shuffle typo, and only the aligned branch runs on typical tensor
// layouts. Run both aligned and deliberately offset pointers so every overload
// and branch is exercised on device.

namespace
{
constexpr int kAlignedOffset    = 0;
constexpr int kMisalignedOffset = 1;
constexpr int kFloatCount       = 64;

__device__ bool vec2_matches_ptr(const glm::vec2 &value, const float *ptr)
{
    return value.x == ptr[0] && value.y == ptr[1];
}

__device__ bool vec3_matches_ptr(const glm::vec3 &value, const float *ptr)
{
    return value.x == ptr[0] && value.y == ptr[1] && value.z == ptr[2];
}

__device__ bool vec4_matches_ptr(const glm::vec4 &value, const float *ptr)
{
    return value.x == ptr[0] && value.y == ptr[1] && value.z == ptr[2] && value.w == ptr[3];
}

template<int Cols, int Rows>
__device__ bool matrix_matches_ptr(const glm::mat<Cols, Rows, float, glm::defaultp> &value, const float *ptr)
{
    for(int col = 0; col < Cols; ++col)
    {
        for(int row = 0; row < Rows; ++row)
        {
            if(value[col][row] != ptr[col * Rows + row])
            {
                return false;
            }
        }
    }
    return true;
}

__device__ bool run_all_overload_checks(const float *ptr, int *results, int base_index)
{
    bool ok                 = true;
    ok                      = ok && vec2_matches_ptr(glm::make_vec2(ptr), ptr);
    results[base_index + 0] = ok ? 1 : 0;

    ok                      = vec3_matches_ptr(glm::make_vec3(ptr), ptr);
    results[base_index + 1] = ok ? 1 : 0;

    ok                      = vec4_matches_ptr(glm::make_vec4(ptr), ptr);
    results[base_index + 2] = ok ? 1 : 0;

    ok                      = matrix_matches_ptr<2, 2>(glm::make_mat2(ptr), ptr);
    results[base_index + 3] = ok ? 1 : 0;

    ok                      = matrix_matches_ptr<3, 3>(glm::make_mat3(ptr), ptr);
    results[base_index + 4] = ok ? 1 : 0;

    ok                      = matrix_matches_ptr<3, 4>(glm::make_mat3x4(ptr), ptr);
    results[base_index + 5] = ok ? 1 : 0;

    ok                      = matrix_matches_ptr<4, 3>(glm::make_mat4x3(ptr), ptr);
    results[base_index + 6] = ok ? 1 : 0;

    ok                      = matrix_matches_ptr<4, 4>(glm::make_mat4x4(ptr), ptr);
    results[base_index + 7] = ok ? 1 : 0;

    ok                      = matrix_matches_ptr<4, 4>(glm::make_mat4(ptr), ptr);
    results[base_index + 8] = ok ? 1 : 0;

    return ok;
}

__global__ void glm_type_ptr_override_kernel(const float *data, int *results)
{
    if(blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    run_all_overload_checks(data + kAlignedOffset, results, 0);
    run_all_overload_checks(data + kMisalignedOffset, results, 9);
}

bool launch_glm_type_ptr_override_test(const std::vector<float> &host_data, std::vector<int> &host_results)
{
    float *device_data  = nullptr;
    int *device_results = nullptr;
    if(cudaMalloc(&device_data, host_data.size() * sizeof(float)) != cudaSuccess)
    {
        return false;
    }
    if(cudaMalloc(&device_results, host_results.size() * sizeof(int)) != cudaSuccess)
    {
        cudaFree(device_data);
        return false;
    }
    if(cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice)
       != cudaSuccess)
    {
        cudaFree(device_data);
        cudaFree(device_results);
        return false;
    }

    glm_type_ptr_override_kernel<<<1, 1>>>(device_data, device_results);
    if(cudaGetLastError() != cudaSuccess)
    {
        cudaFree(device_data);
        cudaFree(device_results);
        return false;
    }
    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        cudaFree(device_data);
        cudaFree(device_results);
        return false;
    }
    if(cudaMemcpy(host_results.data(), device_results, host_results.size() * sizeof(int), cudaMemcpyDeviceToHost)
       != cudaSuccess)
    {
        cudaFree(device_data);
        cudaFree(device_results);
        return false;
    }

    cudaFree(device_data);
    cudaFree(device_results);
    return true;
}

std::vector<float> make_test_buffer()
{
    std::vector<float> data(kFloatCount);
    for(int i = 0; i < kFloatCount; ++i)
    {
        data[static_cast<size_t>(i)] = 100.0f + static_cast<float>(i);
    }
    return data;
}
} // namespace

class GlmTypePtrOverridesTest : public ::testing::Test
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

TEST_F(GlmTypePtrOverridesTest, DeviceOverloadMatchesScalarLayout)
{
    static constexpr std::array<const char *, 9> kCaseNames = {
        "make_vec2",
        "make_vec3",
        "make_vec4",
        "make_mat2",
        "make_mat3",
        "make_mat3x4",
        "make_mat4x3",
        "make_mat4x4",
        "make_mat4",
    };

    const std::vector<float> data = make_test_buffer();
    std::vector<int> results(18, 0);
    ASSERT_TRUE(launch_glm_type_ptr_override_test(data, results));

    for(int alignment = 0; alignment < 2; ++alignment)
    {
        const char *alignment_label = alignment == 0 ? "aligned" : "misaligned";
        const int base_index        = alignment * 9;
        for(int case_index = 0; case_index < 9; ++case_index)
        {
            EXPECT_EQ(results[static_cast<size_t>(base_index + case_index)], 1)
                << kCaseNames[static_cast<size_t>(case_index)]
                << " failed for "
                << alignment_label
                << " pointer";
        }
    }
}
