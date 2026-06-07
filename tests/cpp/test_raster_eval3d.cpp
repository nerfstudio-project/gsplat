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
#include "ExternalDistortion.h"
#include "Intersect.h"
#include "PrimingChainEncoding.h"
#include "PrimingChainEncoding.cuh"
#include "Projection.h"
#include "RasterizeCSR.cuh"
#include "Rasterization.h"

namespace {

using torch::indexing::Slice;

// Test-only fp16 transmittance magnitudes used to construct packed states in a
// deterministic order without depending on raw half-precision hex literals.
constexpr uint16_t FP16_EXPONENT_SHIFT = 10u;
constexpr int32_t FP16_EXPONENT_BIAS = 15;

constexpr uint16_t fp16_power_of_two_bits(int32_t exponent)
{
    return static_cast<uint16_t>(
        static_cast<uint16_t>(exponent + FP16_EXPONENT_BIAS)
        << FP16_EXPONENT_SHIFT);
}

constexpr uint16_t T_FP16_ONE_EIGHTH_BITS =
    fp16_power_of_two_bits(-3);
constexpr uint16_t T_FP16_ONE_QUARTER_BITS =
    fp16_power_of_two_bits(-2);
constexpr uint16_t T_FP16_ONE_HALF_BITS =
    fp16_power_of_two_bits(-1);

// Native regression coverage for fwd_batch_state early-exit padding.
//
// When every pixel in a tile reaches done=true before the forward kernel has
// walked all batches, the kernel breaks out of the compositing loop and a
// per-thread padding pass fills the remaining batch-boundary slots with the
// frozen terminal state. Without that padding the affected slots hold
// uninitialised memory from at::empty, and the backward pass silently corrupts
// gradients from anything that reads them.
//
// Pytest still owns discovery and execution through tests/test_cpp.py; the
// assertions below call the gsplat C++ entry points directly.

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

c10::intrusive_ptr<FThetaCameraDistortionParameters> invalid_ftheta_coeffs()
{
    // max_angle=0 makes every generated f-theta ray invalid, including the
    // center pixel where theta=0. That lets the test below exercise the
    // ParallelBatch invalid-ray sentinel without relying on a particular
    // image-space point landing outside a finite FOV.
    return c10::make_intrusive<FThetaCameraDistortionParameters>(
        FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE,
        std::array<float, FThetaCameraDistortionParameters::PolynomialDegree>{
            0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        std::array<float, FThetaCameraDistortionParameters::PolynomialDegree>{
            0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        0.0f,
        std::array<float, 3>{});
}

float max_abs_diff(const at::Tensor &actual, const at::Tensor &expected)
{
    // Pull only the scalar maximum back to CPU. The comparison tensors can stay
    // on CUDA, matching the kernel output path under test.
    return (actual - expected).abs().max().cpu().item<float>();
}

} // namespace

class PrimingChainEncodingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA runtime is not available";
        }

#if !GSPLAT_BUILD_3DGUT
        GTEST_SKIP() << "3DGUT support is not built in";
#endif
    }

    torch::TensorOptions i32_cuda() const
    {
        return torch::TensorOptions()
            .device(torch::kCUDA)
            .dtype(torch::kInt32);
    }
};

TEST_F(PrimingChainEncodingTest, DecodeForBatchHandlesForcedOrderCases)
{
#if !GSPLAT_BUILD_3DGUT
    GTEST_SKIP() << "3DGUT support is not built in";
#else
    at::Tensor packed = torch::tensor(
        {
            static_cast<int32_t>(gsplat::priming::INIT_PACKED),
            static_cast<int32_t>(
                gsplat::priming::encode_from_fp16_bits(
                    T_FP16_ONE_HALF_BITS, 1, false)),
            static_cast<int32_t>(
                gsplat::priming::encode_from_fp16_bits(
                    T_FP16_ONE_QUARTER_BITS, 2, true)),
            static_cast<int32_t>(
                gsplat::priming::encode_from_fp16_bits(
                    T_FP16_ONE_EIGHTH_BITS, 4, false)),
            static_cast<int32_t>(
                gsplat::priming::encode_from_fp16_bits(
                    T_FP16_ONE_EIGHTH_BITS, 3, false)),
        },
        i32_cuda());
    at::Tensor batch_ids =
        torch::tensor({0, 2, 3, 3, 4}, i32_cuda());

    auto decoded = gsplat::launch_priming_decode_for_batch(packed, batch_ids);

    at::Tensor T_init = std::get<0>(decoded).cpu();
    at::Tensor stored_K = std::get<1>(decoded).cpu();
    at::Tensor stored_sat = std::get<2>(decoded).cpu();
    at::Tensor chain_saturated = std::get<3>(decoded).cpu();
    at::Tensor use_stored = std::get<4>(decoded).cpu();

    EXPECT_TRUE(torch::allclose(
        T_init,
        torch::tensor({1.0f, 0.5f, 0.25f, 1.0f, 0.125f})));
    EXPECT_TRUE(torch::equal(
        stored_K,
        torch::tensor({0, 1, 2, 4, 3}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(
        stored_sat,
        torch::tensor({false, false, true, false, false}, torch::kBool)));
    EXPECT_TRUE(torch::equal(
        chain_saturated,
        torch::tensor({false, false, true, false, false}, torch::kBool)));
    EXPECT_TRUE(torch::equal(
        use_stored,
        torch::tensor({true, true, true, false, true}, torch::kBool)));
#endif
}

class FwdBatchStateTest
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

        m_scene = make_scene(GetParam());
    }

    struct Eval3DScene {
        int tile_size;
        int width;
        int height;
        int channels;
        at::Tensor means;
        at::Tensor quats;
        at::Tensor scales;
        at::Tensor colors;
        at::Tensor opacities_bc;
        at::Tensor viewmats;
        at::Tensor Ks;
        at::Tensor isect_offsets;
        at::Tensor flatten_ids;
        c10::intrusive_ptr<UnscentedTransformParameters> ut_params;
        c10::intrusive_ptr<FThetaCameraDistortionParameters> ftheta_coeffs;
    };

    Eval3DScene &scene()
    {
        return m_scene;
    }

    gsplat::RasterizeToPixelsFromWorld3DGSFwdResult
    run_mixed_batch_fwd(const bool fwd_only, const bool return_last_ids)
    {
        Eval3DScene &scene = m_scene;
        return gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
            scene.means,
            scene.quats,
            scene.scales,
            scene.colors,
            scene.opacities_bc,
            c10::nullopt, // backgrounds
            c10::nullopt, // masks
            scene.width,
            scene.height,
            scene.tile_size,
            scene.viewmats,
            c10::nullopt, // viewmats1
            scene.Ks,
            gsplat::PINHOLE,
            scene.ut_params,
            ShutterType::GLOBAL,
            c10::nullopt, // rays
            c10::nullopt, // radial_coeffs
            c10::nullopt, // tangential_coeffs
            c10::nullopt, // thin_prism_coeffs
            scene.ftheta_coeffs,
            c10::nullopt, // lidar_coeffs
            c10::nullopt, // external_distortion_params
            scene.isect_offsets,
            scene.flatten_ids,
            false, // use_hit_distance
            gsplat::RendererConfig::MIXED_BATCH,
            fwd_only,
            return_last_ids,
            c10::nullopt, // sample_counts
            c10::nullopt, // normals
            false); // unsafe_masked_tile_outputs
    }

private:
    Eval3DScene m_scene;

    // Build the synthetic scene used by every parameter instance. Kept off the
    // SetUp path so the fixture body advertises only the runtime skip rules.
    static Eval3DScene make_scene(int tile_size)
    {
        // Synthetic scene chosen to trigger multi-batch early-exit padding:
        // - image is exactly one tile so num_batches tracks the gaussian count.
        // - many fully covering, fully opaque gaussians make every pixel reach
        //   done=true within the first batch.
        // - subsequent batch boundaries can therefore only be populated by the
        //   forward kernel's early-exit padding path.
        const int width = tile_size;
        const int height = tile_size;
        const int pixels_per_tile = tile_size * tile_size;
        // Five full tile-sized batches plus one extra Gaussian produce six
        // raster batches. With the production batch-state CSR, that means at
        // least two batch slots for this single tile.
        const int num_gaussians = pixels_per_tile * 5 + 1;
        const int channels = 3;
        const int num_images = 1;
        const int tile_width =
            static_cast<int>(std::ceil(width / static_cast<float>(tile_size)));
        const int tile_height =
            static_cast<int>(std::ceil(height / static_cast<float>(tile_size)));

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
        // world-space Gaussians, compute tile intersections, then call the
        // low-level rasterizer so the test can inspect fwd_batch_state directly.
        // The four scalar literals (eps2d, near_plane, far_plane, radius_clip)
        // mirror the defaults of fully_fused_projection_with_ut in
        // gsplat/rendering.py.
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

        // Intersect the projected Gaussians with the single image tile. Sorting
        // is enabled to match the normal production path.
        gsplat::TileIntersectResult isects = gsplat::intersect_tile(
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
        at::Tensor isect_offsets =
            gsplat::intersect_offset(isects.isect_ids, num_images, tile_width, tile_height);

        return Eval3DScene{
            .tile_size = tile_size,
            .width = width,
            .height = height,
            .channels = channels,
            .means = means,
            .quats = quats,
            .scales = scales,
            .colors = colors,
            .opacities_bc = opacities_bc,
            .viewmats = viewmats,
            .Ks = Ks,
            .isect_offsets = isect_offsets,
            .flatten_ids = isects.flatten_ids,
            .ut_params = ut_params,
            .ftheta_coeffs = ftheta_coeffs,
        };
    }
};

INSTANTIATE_TEST_SUITE_P(
    TileSize, FwdBatchStateTest, ::testing::Values(8, 16),
    [](const ::testing::TestParamInfo<int> &info) {
        return "Tile" + std::to_string(info.param);
    });

TEST_P(FwdBatchStateTest, MixedBatchFwdOnlyKeepsPublicOutputs)
{
    Eval3DScene &scene = this->scene();
    SCOPED_TRACE("tile_size=" + std::to_string(scene.tile_size));

    gsplat::RasterizeToPixelsFromWorld3DGSFwdResult exact =
        run_mixed_batch_fwd(/*fwd_only=*/false, /*return_last_ids=*/true);
    gsplat::RasterizeToPixelsFromWorld3DGSFwdResult fwd_only =
        run_mixed_batch_fwd(/*fwd_only=*/true, /*return_last_ids=*/false);

    ASSERT_TRUE(exact.batches_per_tile.defined());
    ASSERT_TRUE(exact.batch_offsets.defined());
    ASSERT_TRUE(exact.fwd_batch_state.defined());

    EXPECT_FALSE(fwd_only.batches_per_tile.defined());
    EXPECT_FALSE(fwd_only.batch_offsets.defined());
    EXPECT_FALSE(fwd_only.fwd_batch_state.defined());
    EXPECT_FALSE(fwd_only.compose_c_stop.defined());
    EXPECT_FALSE(fwd_only.priming_state.defined());
    EXPECT_FALSE(fwd_only.last_ids.defined());

    EXPECT_LE(
        max_abs_diff(fwd_only.renders, exact.renders),
        1.0e-5f);
    EXPECT_LE(
        max_abs_diff(fwd_only.alphas, exact.alphas),
        1.0e-5f);
}

TEST_P(FwdBatchStateTest, LastSlotMatchesTerminalAfterEarlyExit)
{
    Eval3DScene &scene = this->scene();
    SCOPED_TRACE("tile_size=" + std::to_string(scene.tile_size));

    // backgrounds=None makes the terminal accumulated pixel color equal the
    // public render_colors output, which gives a direct invariant for the
    // deepest slot c=num_batches-1.
    gsplat::RasterizeToPixelsFromWorld3DGSFwdResult raster =
        gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
            scene.means,
            scene.quats,
            scene.scales,
            scene.colors,
            scene.opacities_bc,
            c10::nullopt, // backgrounds
            c10::nullopt, // masks
            scene.width,
            scene.height,
            scene.tile_size,
            scene.viewmats,
            c10::nullopt, // viewmats1
            scene.Ks,
            gsplat::PINHOLE,
            scene.ut_params,
            ShutterType::GLOBAL,
            c10::nullopt, // rays
            c10::nullopt, // radial_coeffs
            c10::nullopt, // tangential_coeffs
            c10::nullopt, // thin_prism_coeffs
            scene.ftheta_coeffs,
            c10::nullopt, // lidar_coeffs
            c10::nullopt, // external_distortion_params
            scene.isect_offsets,
            scene.flatten_ids,
            false, // use_hit_distance
            gsplat::RendererConfig::MIXED_BATCH,
            false, // fwd_only
            true, // return_last_ids
            c10::nullopt, // sample_counts
            c10::nullopt, // normals
            false); // unsafe_masked_tile_outputs
    // If this precondition fails, the synthetic scene stopped exercising the
    // multi-batch path and the terminal-slot padding invariant is meaningless.
    ASSERT_EQ(raster.batches_per_tile.numel(), 1);
    const int num_batches = raster.batches_per_tile.cpu().item<int>();
    ASSERT_GE(num_batches, 2)
        << "expected num_batches >= 2 to exercise multi-batch padding";

    // Slot c=num_batches-1 is the terminal batch boundary. In the normal flow
    // it is persisted at the final batch; when early exit fires before that
    // batch, the terminal slot is only ever written by the padding pass. Broken
    // padding leaves at::empty garbage in the slot, which diverges from the
    // public render outputs for every pixel in the tile.
    // SOA layout: fwd_batch_state[c, state_element, pix]. Keeping pix as the
    // fastest-varying axis makes kernel accesses coalesced but requires the
    // test to transpose pix_out back to image order before comparison.
    const at::Tensor terminal_slot =
        raster.fwd_batch_state.select(0, num_batches - 1);
    const int64_t pix_offset =
        static_cast<int64_t>(gsplat::FWD_BATCH_STATE_PIX_OFFSET);
    const at::Tensor expected_T =
        1.0f - raster.alphas.reshape({scene.height, scene.width});
    const at::Tensor actual_T =
        terminal_slot.index({0, Slice()})
            .reshape({scene.tile_size, scene.tile_size});
    const at::Tensor expected_pix =
        raster.renders.reshape({scene.height, scene.width, scene.channels});
    const at::Tensor actual_pix =
        terminal_slot.index({
            Slice(pix_offset, pix_offset + scene.channels),
            Slice()})
            .transpose(0, 1)
            .reshape({scene.tile_size, scene.tile_size, scene.channels});

    EXPECT_LE(max_abs_diff(actual_T, expected_T), 1.0e-5f)
        << "terminal-slot T values differ from 1 - render_alphas. Most likely "
           "the fwd early-exit padding is broken and the slot was never written.";
    EXPECT_LE(max_abs_diff(actual_pix, expected_pix), 1.0e-5f)
        << "terminal-slot pix_out values differ from render_colors. Most likely "
           "the fwd early-exit padding is broken and the slot was never written.";
}

TEST_P(FwdBatchStateTest, ParallelBatchSaturatedSlotMatchesTerminal)
{
    Eval3DScene &scene = this->scene();
    SCOPED_TRACE("tile_size=" + std::to_string(scene.tile_size));

    // ParallelBatch no longer fans the post-saturation accumulator into every
    // batch slot below c_stop. Batch-scan writes the c_stop slot preamble,
    // then batch-replay consumes that preamble and overwrites the same slot
    // with the terminal state that ParallelBwd reads.
    gsplat::RasterizeToPixelsFromWorld3DGSFwdResult raster =
        gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
            scene.means,
            scene.quats,
            scene.scales,
            scene.colors,
            scene.opacities_bc,
            c10::nullopt, // backgrounds
            c10::nullopt, // masks
            scene.width,
            scene.height,
            scene.tile_size,
            scene.viewmats,
            c10::nullopt, // viewmats1
            scene.Ks,
            gsplat::PINHOLE,
            scene.ut_params,
            ShutterType::GLOBAL,
            c10::nullopt, // rays
            c10::nullopt, // radial_coeffs
            c10::nullopt, // tangential_coeffs
            c10::nullopt, // thin_prism_coeffs
            scene.ftheta_coeffs,
            c10::nullopt, // lidar_coeffs
            c10::nullopt, // external_distortion_params
            scene.isect_offsets,
            scene.flatten_ids,
            false, // use_hit_distance
            gsplat::RendererConfig::PARALLEL_BATCH,
            false, // fwd_only
            true, // return_last_ids
            c10::nullopt, // sample_counts
            c10::nullopt, // normals
            false); // unsafe_masked_tile_outputs

    ASSERT_TRUE(raster.compose_c_stop.defined());

    const int64_t pix_offset =
        static_cast<int64_t>(gsplat::FWD_BATCH_STATE_PIX_OFFSET);
    const at::Tensor compose_c_stop_i32 = raster.compose_c_stop.to(at::kInt);
    const int64_t unsaturated_pixels =
        compose_c_stop_i32.eq(static_cast<int32_t>(gsplat::COMPOSE_C_STOP_NONE))
            .sum().cpu().item<int64_t>();
    ASSERT_EQ(unsaturated_pixels, 0)
        << "expected every pixel in the synthetic dense scene to saturate";
    const int32_t c_stop_min_packed =
        compose_c_stop_i32.min().cpu().item<int32_t>();
    const int32_t c_stop_max_packed =
        compose_c_stop_i32.max().cpu().item<int32_t>();
    ASSERT_EQ(c_stop_min_packed, c_stop_max_packed)
        << "synthetic dense scene should stop every pixel in the same batch";
    const int32_t c_stop_min =
        gsplat::decode_compose_c_stop(static_cast<uint16_t>(c_stop_min_packed));
    ASSERT_GE(c_stop_min, 0);

    const at::Tensor stop_slot =
        raster.fwd_batch_state.select(0, static_cast<int64_t>(c_stop_min));

    const at::Tensor expected_T =
        1.0f - raster.alphas.reshape({scene.height, scene.width});
    const at::Tensor actual_T =
        stop_slot.index({0, Slice()})
            .reshape({scene.tile_size, scene.tile_size});
    const at::Tensor expected_pix =
        raster.renders.reshape({scene.height, scene.width, scene.channels});
    const at::Tensor actual_pix =
        stop_slot.index({
            Slice(pix_offset, pix_offset + scene.channels),
            Slice()})
            .transpose(0, 1)
            .reshape({scene.tile_size, scene.tile_size, scene.channels});

    EXPECT_LE(max_abs_diff(actual_T, expected_T), 1.0e-5f)
        << "c_stop slot T differs from 1 - render_alphas";
    EXPECT_LE(max_abs_diff(actual_pix, expected_pix), 1.0e-5f)
        << "c_stop slot pix_out differs from render_colors";
}

TEST_P(FwdBatchStateTest, ParallelBatchFwdOnlyMatchesExactWithoutMetadata)
{
    Eval3DScene &scene = this->scene();
    SCOPED_TRACE("tile_size=" + std::to_string(scene.tile_size));

    auto run_parallel_batch =
        [&scene](const bool fwd_only, const bool return_last_ids) {
            return gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
                scene.means,
                scene.quats,
                scene.scales,
                scene.colors,
                scene.opacities_bc,
                c10::nullopt, // backgrounds
                c10::nullopt, // masks
                scene.width,
                scene.height,
                scene.tile_size,
                scene.viewmats,
                c10::nullopt, // viewmats1
                scene.Ks,
                gsplat::PINHOLE,
                scene.ut_params,
                ShutterType::GLOBAL,
                c10::nullopt, // rays
                c10::nullopt, // radial_coeffs
                c10::nullopt, // tangential_coeffs
                c10::nullopt, // thin_prism_coeffs
                scene.ftheta_coeffs,
                c10::nullopt, // lidar_coeffs
                c10::nullopt, // external_distortion_params
                scene.isect_offsets,
                scene.flatten_ids,
                false, // use_hit_distance
                gsplat::RendererConfig::PARALLEL_BATCH,
                fwd_only,
                return_last_ids,
                c10::nullopt, // sample_counts
                c10::nullopt, // normals
                false); // unsafe_masked_tile_outputs
        };

    // Exact mode requests last_ids, so ParallelBatch must keep the
    // batch-replay metadata path live. This fixture saturates, making the
    // exact path exercise the replay boundary that fwd-only removes.
    gsplat::RasterizeToPixelsFromWorld3DGSFwdResult exact =
        run_parallel_batch(/*fwd_only=*/false, /*return_last_ids=*/true);
    gsplat::RasterizeToPixelsFromWorld3DGSFwdResult fwd_only =
        run_parallel_batch(/*fwd_only=*/true, /*return_last_ids=*/false);

    ASSERT_TRUE(exact.last_ids.defined());
    ASSERT_TRUE(exact.compose_c_stop.defined());
    ASSERT_FALSE(fwd_only.last_ids.defined());
    ASSERT_FALSE(fwd_only.compose_c_stop.defined());

    const at::Tensor compose_c_stop_i32 = exact.compose_c_stop.to(at::kInt);
    const int64_t replay_pixels =
        compose_c_stop_i32
            .ne(static_cast<int32_t>(gsplat::COMPOSE_C_STOP_NONE))
            .sum()
            .cpu()
            .item<int64_t>();
    ASSERT_GT(replay_pixels, 0)
        << "the fwd-only regression fixture must exercise the batch-replay "
           "boundary in exact mode";

    EXPECT_LE(max_abs_diff(fwd_only.renders, exact.renders), 1.0e-5f)
        << "fwd-only ParallelBatch changed rendered colors";
    EXPECT_LE(max_abs_diff(fwd_only.alphas, exact.alphas), 1.0e-5f)
        << "fwd-only ParallelBatch changed rendered alphas";
}

class InvalidRayStateTest
    : public ::testing::TestWithParam<int>
{
protected:
    void SetUp() override
    {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA runtime is not available";
        }

#if !GSPLAT_BUILD_3DGUT
        GTEST_SKIP() << "3DGUT support is not built in";
#endif

        m_scene = make_scene(GetParam());
    }

    struct InvalidRayScene {
        int tile_size;
        int width;
        int height;
        int channels;
        at::Tensor means;
        at::Tensor quats;
        at::Tensor scales;
        at::Tensor colors;
        at::Tensor opacities;
        at::Tensor background;
        at::Tensor sample_counts;
        at::Tensor viewmats;
        at::Tensor Ks;
        at::Tensor isect_offsets;
        at::Tensor flatten_ids;
        c10::intrusive_ptr<UnscentedTransformParameters> ut_params;
        c10::intrusive_ptr<FThetaCameraDistortionParameters> ftheta_coeffs;
    };

    InvalidRayScene &scene()
    {
        return m_scene;
    }

private:
    InvalidRayScene m_scene;

    static InvalidRayScene make_scene(int tile_size)
    {
        const int width = tile_size;
        const int height = tile_size;
        const int channels = 3;
        const int num_gaussians = 300;

        const torch::TensorOptions f32_cuda = torch::TensorOptions()
                                                 .device(torch::kCUDA)
                                                 .dtype(torch::kFloat32);
        const torch::TensorOptions i32_cuda = torch::TensorOptions()
                                                 .device(torch::kCUDA)
                                                 .dtype(torch::kInt32);

        at::Tensor means = torch::zeros({num_gaussians, 3}, f32_cuda);
        means.index_put_(
            {Slice(), 2},
            1.0f + torch::arange(num_gaussians, f32_cuda) * 1.0e-3f);
        at::Tensor quats = torch::tensor(
            {{1.0f, 0.0f, 0.0f, 0.0f}}, f32_cuda)
                               .expand({num_gaussians, 4})
                               .contiguous();
        at::Tensor scales = torch::full({num_gaussians, 3}, 100.0f, f32_cuda);
        at::Tensor colors = torch::rand({1, num_gaussians, channels}, f32_cuda);
        at::Tensor opacities =
            torch::full({1, num_gaussians}, 0.7f, f32_cuda);

        const float f = static_cast<float>(tile_size);
        at::Tensor viewmats = torch::eye(4, f32_cuda).unsqueeze(0);
        at::Tensor Ks = torch::tensor(
            {{{f, 0.0f, width / 2.0f},
              {0.0f, f, height / 2.0f},
              {0.0f, 0.0f, 1.0f}}},
            f32_cuda);
        at::Tensor isect_offsets = torch::zeros({1, 1, 1}, i32_cuda);
        at::Tensor flatten_ids = torch::arange(num_gaussians, i32_cuda);
        at::Tensor background =
            torch::tensor({{0.15f, 0.35f, 0.55f}}, f32_cuda);
        at::Tensor sample_counts = torch::empty({1, height, width}, i32_cuda);

        return InvalidRayScene{
            .tile_size = tile_size,
            .width = width,
            .height = height,
            .channels = channels,
            .means = means,
            .quats = quats,
            .scales = scales,
            .colors = colors,
            .opacities = opacities,
            .background = background,
            .sample_counts = sample_counts,
            .viewmats = viewmats,
            .Ks = Ks,
            .isect_offsets = isect_offsets,
            .flatten_ids = flatten_ids,
            .ut_params = c10::make_intrusive<UnscentedTransformParameters>(),
            .ftheta_coeffs = invalid_ftheta_coeffs(),
        };
    }
};

INSTANTIATE_TEST_SUITE_P(
    TileSize, InvalidRayStateTest, ::testing::Values(8, 16),
    [](const ::testing::TestParamInfo<int> &info) {
        return "Tile" + std::to_string(info.param);
    });

TEST_P(InvalidRayStateTest, SkipsBatchStateForInvalidRays)
{
    InvalidRayScene &scene = this->scene();
    SCOPED_TRACE("tile_size=" + std::to_string(scene.tile_size));

    gsplat::RasterizeToPixelsFromWorld3DGSFwdResult raster =
        gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
            scene.means,
            scene.quats,
            scene.scales,
            scene.colors,
            scene.opacities,
            at::optional<at::Tensor>(scene.background),
            c10::nullopt, // masks
            scene.width,
            scene.height,
            scene.tile_size,
            scene.viewmats,
            c10::nullopt, // viewmats1
            scene.Ks,
            gsplat::FTHETA,
            scene.ut_params,
            ShutterType::GLOBAL,
            c10::nullopt, // rays
            c10::nullopt, // radial_coeffs
            c10::nullopt, // tangential_coeffs
            c10::nullopt, // thin_prism_coeffs
            scene.ftheta_coeffs,
            c10::nullopt, // lidar_coeffs
            c10::nullopt, // external_distortion_params
            scene.isect_offsets,
            scene.flatten_ids,
            false, // use_hit_distance
            gsplat::RendererConfig::PARALLEL_BATCH,
            false, // fwd_only
            true, // return_last_ids
            at::optional<at::Tensor>(scene.sample_counts),
            c10::nullopt, // normals
            false); // unsafe_masked_tile_outputs

    ASSERT_TRUE(raster.compose_c_stop.defined());

    const at::Tensor expected_background = scene.background
        .view({1, 1, scene.channels})
        .expand({scene.height, scene.width, scene.channels});
    EXPECT_LE(
        max_abs_diff(
            raster.renders.reshape({scene.height, scene.width, scene.channels}),
            expected_background),
        1.0e-6f);
    EXPECT_EQ(raster.alphas.sum().cpu().item<float>(), 0.0f);
    EXPECT_EQ(raster.last_ids.min().cpu().item<int32_t>(), -1);
    EXPECT_EQ(raster.last_ids.max().cpu().item<int32_t>(), -1);
    EXPECT_EQ(scene.sample_counts.sum().cpu().item<int64_t>(), 0);

    const at::Tensor compose_c_stop_i32 = raster.compose_c_stop.to(at::kInt);
    EXPECT_EQ(
        compose_c_stop_i32.min().cpu().item<int32_t>(),
        static_cast<int32_t>(gsplat::COMPOSE_C_STOP_INVALID_RAY));
    EXPECT_EQ(
        compose_c_stop_i32.max().cpu().item<int32_t>(),
        static_cast<int32_t>(gsplat::COMPOSE_C_STOP_INVALID_RAY));
}

// The external-distortion contiguity checks live at the launcher deref site
// rather than the op layer. A non-contiguous polynomial must therefore be
// rejected by the from-world launcher itself -- this pins that the deref-site
// consolidation cannot silently drop the guard in a future refactor.
TEST_P(FwdBatchStateTest, NonContiguousExternalDistortionPolyThrows)
{
    Eval3DScene &scene = this->scene();

    const torch::TensorOptions options = torch::TensorOptions()
                                             .device(torch::kCUDA)
                                             .dtype(torch::kFloat32);
    // transpose yields a non-contiguous view without copying the storage.
    at::Tensor non_contiguous_poly =
        torch::zeros({2, 6}, options).transpose(0, 1);
    ASSERT_FALSE(non_contiguous_poly.is_contiguous());
    at::Tensor contiguous_poly = torch::zeros({6, 2}, options);

    auto external_distortion_params =
        c10::make_intrusive<gsplat::extdist::BivariateWindshieldModelParameters>();
    external_distortion_params->horizontal_poly = non_contiguous_poly;
    external_distortion_params->vertical_poly = contiguous_poly;
    external_distortion_params->horizontal_poly_inverse = contiguous_poly;
    external_distortion_params->vertical_poly_inverse = contiguous_poly;

    EXPECT_THROW(
        gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
            scene.means,
            scene.quats,
            scene.scales,
            scene.colors,
            scene.opacities_bc,
            c10::nullopt, // backgrounds
            c10::nullopt, // masks
            scene.width,
            scene.height,
            scene.tile_size,
            scene.viewmats,
            c10::nullopt, // viewmats1
            scene.Ks,
            gsplat::PINHOLE,
            scene.ut_params,
            ShutterType::GLOBAL,
            c10::nullopt, // rays
            c10::nullopt, // radial_coeffs
            c10::nullopt, // tangential_coeffs
            c10::nullopt, // thin_prism_coeffs
            scene.ftheta_coeffs,
            c10::nullopt, // lidar_coeffs
            external_distortion_params,
            scene.isect_offsets,
            scene.flatten_ids,
            false, // use_hit_distance
            gsplat::RendererConfig::MIXED_BATCH,
            false, // fwd_only
            false, // return_last_ids
            c10::nullopt, // sample_counts
            c10::nullopt, // normals
            false), // unsafe_masked_tile_outputs
        c10::Error);
}
