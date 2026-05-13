/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include <torch/cuda.h>
#include <torch/torch.h>

#include "Config.h"
#include "Ops.h"

namespace {

using torch::indexing::Slice;

// Native regression coverage for fwd_chunk_state early-exit padding.
//
// When every pixel in a tile reaches done=true before the forward kernel has
// walked all batches, the kernel breaks out of the compositing loop and a
// per-thread padding pass fills the remaining chunk-boundary slots with the
// frozen terminal state. Without that padding the affected slots hold
// uninitialised memory from at::empty, and the backward pass silently corrupts
// gradients from anything that reads them.
//
// Pytest still owns discovery and execution through tests/test_cpp.py; the
// assertions below call the gsplat C++ entry points directly.

#if GSPLAT_BUILD_3DGUT

c10::intrusive_ptr<FThetaCameraDistortionParameters> default_ftheta_coeffs()
{
    // The pinhole path does not consume f-theta coefficients, but the 3DGUT
    // operator signature requires a non-null custom-class holder.
    return c10::make_intrusive<FThetaCameraDistortionParameters>(
        FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE,
        std::array<float, FThetaCameraDistortionParameters::PolynomialDegree>{},
        std::array<float, FThetaCameraDistortionParameters::PolynomialDegree>{}, 0.0f,
        std::array<float, 3>{});
}

float max_abs_diff(const at::Tensor &actual, const at::Tensor &expected)
{
    // Pull only the scalar maximum back to CPU. The comparison tensors can stay
    // on CUDA, matching the kernel output path under test.
    return (actual - expected).abs().max().cpu().item<float>();
}

#endif // GSPLAT_BUILD_3DGUT

} // namespace

class FwdChunkStateTest
    : public ::testing::TestWithParam<int>
{
protected:
    void SetUp() override
    {
        // Skipping from SetUp keeps the actual test focused on the invariant.
        // GoogleTest records this as a skipped parameter instance.
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA runtime is not available";
        }

#if !GSPLAT_BUILD_3DGUT
        GTEST_SKIP() << "3DGUT support is not built in";
#endif
    }
};

INSTANTIATE_TEST_SUITE_P(TileSize, FwdChunkStateTest, ::testing::Values(8, 16),
    [](const ::testing::TestParamInfo<int> &info) {
        return "Tile" + std::to_string(info.param);
    });

TEST_P(FwdChunkStateTest, C0MatchesTerminalAfterEarlyExit)
{
#if !GSPLAT_BUILD_3DGUT
    // SetUp() has already reported the skip. This return keeps builds without
    // 3DGUT from compiling/linking calls to 3DGUT-only operators below.
    return;
#else
    const int tile_size = GetParam();
    SCOPED_TRACE("tile_size=" + std::to_string(tile_size));

    // Synthetic scene chosen to trigger multi-chunk early-exit padding:
    // - image is exactly one tile so num_chunks tracks the gaussian count.
    // - many fully covering, fully opaque gaussians make every pixel reach
    //   done=true within the first batch.
    // - subsequent chunk boundaries can therefore only be populated by the
    //   forward kernel's early-exit padding path.
    const int width = tile_size;
    const int height = tile_size;
    const int pixels_per_tile = tile_size * tile_size;
    // Five full tile-sized batches plus one extra Gaussian produce six raster
    // batches. With the production CHUNK_BATCHES=4, that means at least two
    // chunk slots for this single tile.
    const int num_gaussians = pixels_per_tile * 5 + 1;
    const int channels = 3;
    const int num_images = 1;
    const int tile_width = static_cast<int>(std::ceil(width / static_cast<float>(tile_size)));
    const int tile_height = static_cast<int>(std::ceil(height / static_cast<float>(tile_size)));

    const torch::TensorOptions options = torch::TensorOptions()
                                             .device(torch::kCUDA)
                                             .dtype(torch::kFloat32);

    at::Tensor means = torch::zeros({num_gaussians, 3}, options);
    means.index_put_(
        {Slice(), 2},
        1.0f + torch::arange(num_gaussians, options) * 1.0e-4f);

    // All Gaussians sit in front of one pinhole camera. A large scale makes
    // every Gaussian cover the whole tile, and opacity 1.0 forces early exit.
    at::Tensor quats = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, options)
                           .expand({num_gaussians, 4})
                           .contiguous();
    at::Tensor scales = torch::full({num_gaussians, 3}, 0.5f, options);
    at::Tensor opacities = torch::ones({num_gaussians}, options);
    at::Tensor colors = torch::rand({1, num_gaussians, channels}, options);
    at::Tensor opacities_bc = opacities.unsqueeze(0).contiguous();

    const float fx = static_cast<float>(width);
    const float fy = static_cast<float>(height);
    const float cx = width / 2.0f;
    const float cy = height / 2.0f;
    at::Tensor viewmats = torch::eye(4, options).unsqueeze(0);
    at::Tensor Ks = torch::tensor(
        {{{fx, 0.0f, cx}, {0.0f, fy, cy}, {0.0f, 0.0f, 1.0f}}},
        options);

    c10::intrusive_ptr<UnscentedTransformParameters> ut_params =
        c10::make_intrusive<UnscentedTransformParameters>();
    c10::intrusive_ptr<FThetaCameraDistortionParameters> ftheta_coeffs =
        default_ftheta_coeffs();

    using ProjectionResult =
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

    // Use the same operator pipeline as normal 3DGUT rasterization: project
    // world-space Gaussians, compute tile intersections, then call the low-level
    // rasterizer so the test can inspect fwd_chunk_state directly. The four
    // scalar literals (eps2d, near_plane, far_plane, radius_clip) mirror the
    // defaults of fully_fused_projection_with_ut in gsplat/rendering.py.
    ProjectionResult projection = gsplat::projection_ut_3dgs_fused(
        means,
        quats,
        scales,
        at::optional<at::Tensor>(opacities),
        viewmats,
        c10::nullopt, // viewmats1
        Ks,
        width,
        height,
        0.3,   // eps2d
        0.01,  // near_plane
        1.0e10, // far_plane
        0.0,   // radius_clip
        false, // calc_compensations
        static_cast<int64_t>(gsplat::PINHOLE),
        true,  // global_z_order
        ut_params,
        static_cast<int64_t>(ShutterType::GLOBAL),
        c10::nullopt, // radial_coeffs
        c10::nullopt, // tangential_coeffs
        c10::nullopt, // thin_prism_coeffs
        ftheta_coeffs,
        c10::nullopt, // lidar_coeffs
        c10::nullopt); // external_distortion_params
    const at::Tensor &radii = std::get<0>(projection);
    const at::Tensor &means2d = std::get<1>(projection);
    const at::Tensor &depths = std::get<2>(projection);

    using IntersectionResult = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

    // Intersect the projected Gaussians with the single image tile. Sorting is
    // enabled to match the normal production path.
    IntersectionResult intersections = gsplat::intersect_tile(
        means2d,
        radii,
        depths,
        c10::nullopt, // conics
        c10::nullopt, // opacities
        c10::nullopt, // image_ids
        c10::nullopt, // gaussian_ids
        num_images,
        tile_size,
        tile_width,
        tile_height,
        true,  // sort
        false); // segmented
    const at::Tensor &isect_ids = std::get<1>(intersections);
    const at::Tensor &flatten_ids = std::get<2>(intersections);
    at::Tensor isect_offsets =
        gsplat::intersect_offset(isect_ids, num_images, tile_width, tile_height);

    using RasterResult =
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

    // backgrounds=None makes the terminal accumulated pixel color equal the
    // public render_colors output, which gives a direct invariant for slot c=0.
    RasterResult raster = gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
        means,
        quats,
        scales,
        colors,
        opacities_bc,
        c10::nullopt, // backgrounds
        c10::nullopt, // masks
        width,
        height,
        tile_size,
        viewmats,
        c10::nullopt, // viewmats1
        Ks,
        static_cast<int64_t>(gsplat::PINHOLE),
        ut_params,
        static_cast<int64_t>(ShutterType::GLOBAL),
        c10::nullopt, // rays
        c10::nullopt, // radial_coeffs
        c10::nullopt, // tangential_coeffs
        c10::nullopt, // thin_prism_coeffs
        ftheta_coeffs,
        c10::nullopt, // lidar_coeffs
        c10::nullopt, // external_distortion_params
        isect_offsets,
        flatten_ids,
        false, // use_hit_distance
        c10::nullopt, // sample_counts
        c10::nullopt); // normals
    const at::Tensor &render_colors = std::get<0>(raster);
    const at::Tensor &render_alphas = std::get<1>(raster);
    const at::Tensor &chunks_per_tile = std::get<3>(raster);
    const at::Tensor &fwd_chunk_state = std::get<5>(raster);

    // If this precondition fails, the synthetic scene stopped exercising the
    // multi-chunk path and the terminal-slot padding invariant is meaningless.
    ASSERT_EQ(chunks_per_tile.numel(), 1);
    const int num_chunks = chunks_per_tile.cpu().item<int>();
    ASSERT_GE(num_chunks, 2)
        << "expected num_chunks >= 2 to exercise multi-chunk padding";

    // Slot c=0 is the terminal chunk boundary. In the normal flow it is
    // persisted at the final batch num_batches-1; when early exit fires before
    // that batch, slot c=0 is only ever written by the padding pass. Broken
    // padding leaves at::empty garbage in slot c=0, which diverges from the
    // public render outputs for every pixel in the tile.
    const at::Tensor terminal_slot = fwd_chunk_state.select(0, 0);
    const at::Tensor expected_T = 1.0f - render_alphas.reshape({height, width});
    const at::Tensor actual_T =
        terminal_slot.index({Slice(), 0}).reshape({tile_size, tile_size});
    const at::Tensor expected_pix = render_colors.reshape({height, width, channels});
    const at::Tensor actual_pix =
        terminal_slot.index({Slice(), Slice(1, 1 + channels)})
            .reshape({tile_size, tile_size, channels});

    EXPECT_LE(max_abs_diff(actual_T, expected_T), 1.0e-5f)
        << "slot c=0 T values differ from 1 - render_alphas. Most likely "
           "the fwd early-exit padding is broken and slot c=0 was never written.";
    EXPECT_LE(max_abs_diff(actual_pix, expected_pix), 1.0e-5f)
        << "slot c=0 pix_out values differ from render_colors. Most likely "
           "the fwd early-exit padding is broken and slot c=0 was never written.";
#endif // GSPLAT_BUILD_3DGUT
}
