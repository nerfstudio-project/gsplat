/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/core/grad_mode.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "Config.h"
#include "Common.h"
#include "PrimingChainEncoding.cuh"
#include "Rasterization.h"
#include "RasterizeCSR.cuh"
#include "RasterizeToPixelsFromWorld3DGS.h"
#include "TorchUtils.h"
#include "Cameras.h"

namespace gsplat {

struct RasterizeToPixels3DGSFwdResult {
    at::Tensor renders;
    at::Tensor alphas;
    at::Tensor last_ids;
    at::Tensor means2d_absgrad;
};

template <> struct TorchArgDef<RasterizeToPixels3DGSFwdResult> {
    // last_ids is forward-internal state; alphas is exposed as float.
    static auto to(const RasterizeToPixels3DGSFwdResult &r) { return to_torch_args(
        r.renders, r.alphas.to(at::kFloat), r.means2d_absgrad
    ); }
};

// RasterizeToPixels3DGSResult lives in Rasterization.h (cross-TU orchestration
// callers). Its dispatcher boxing stays co-located with the op here.
template <> struct TorchArgDef<RasterizeToPixels3DGSResult> {
    static auto to(const RasterizeToPixels3DGSResult &r) { return to_torch_args(r.renders, r.alphas, r.means2d_absgrad); }
};

struct RasterizeToIndices3DGSResult {
    at::Tensor gaussian_ids;
    at::Tensor pixel_ids;
};

template <>
struct TorchArgDef<RasterizeToIndices3DGSResult> {
    static auto to(const RasterizeToIndices3DGSResult &r) {
        return to_torch_args(r.gaussian_ids, r.pixel_ids);
    }
};

#if GSPLAT_BUILD_3DGUT
namespace {

RendererConfig parse_renderer_config(const int64_t renderer_config)
{
    switch (renderer_config) {
    case static_cast<int64_t>(RendererConfig::MIXED_BATCH):
        return RendererConfig::MIXED_BATCH;
    case static_cast<int64_t>(RendererConfig::PARALLEL_BATCH):
        return RendererConfig::PARALLEL_BATCH;
    default:
        TORCH_CHECK(false, "unsupported 3DGUT renderer_config: ", renderer_config);
        return RendererConfig::MIXED_BATCH;
    }
}

constexpr int32_t kMaxParallelBatchesPerTile =
    static_cast<int32_t>(COMPOSE_C_STOP_INVALID_RAY);

static_assert(
    priming::detail::K_LIMIT - 1 == kMaxParallelBatchesPerTile,
    "ParallelBatch priming and compose-stop packing limits must match");

} // namespace
#endif // GSPLAT_BUILD_3DGUT

#if GSPLAT_BUILD_3DGS

////////////////////////////////////////////////////
// 3DGS
////////////////////////////////////////////////////

namespace {

void check_rasterize_to_pixels_3dgs_inputs(
    const at::Tensor &means2d, const at::Tensor &conics,
    const at::Tensor &colors, const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::Tensor &isect_offsets,
    bool packed
) {
    TORCH_CHECK(
        tile_size == 4 || tile_size == 16,
        "Only tile_size in {4, 16} is supported for 3DGS rasterization, got ",
        tile_size
    );

    TORCH_CHECK(
        means2d.dim() >= 2,
        "means2d must have shape [..., N, 2] or [nnz, 2], got ",
        means2d.sizes()
    );
    TORCH_CHECK(
        colors.dim() >= 2,
        "colors must have shape [..., N, channels] or [nnz, channels], got ",
        colors.sizes()
    );
    TORCH_CHECK(
        means2d.size(-1) == 2,
        "means2d must have shape [..., N, 2] or [nnz, 2], got ",
        means2d.sizes()
    );

    at::DimVector image_dims(means2d.sizes().slice(0, means2d.dim() - 2));
    const int64_t channels = colors.size(-1);
    if (packed) {
        const int64_t nnz = means2d.size(0);
        TORCH_CHECK(
            means2d.dim() == 2 && means2d.size(0) == nnz &&
                means2d.size(1) == 2,
            "packed means2d must have shape [nnz, 2], got ",
            means2d.sizes()
        );
        TORCH_CHECK(
            conics.dim() == 2 && conics.size(0) == nnz && conics.size(1) == 3,
            "packed conics must have shape [nnz, 3], got ",
            conics.sizes()
        );
        TORCH_CHECK(
            colors.size(0) == nnz,
            "packed colors first dimension must match nnz; got colors ",
            colors.sizes(),
            " and nnz ",
            nnz
        );
        TORCH_CHECK(
            opacities.dim() == 1 && opacities.size(0) == nnz,
            "packed opacities must have shape [nnz], got ",
            opacities.sizes()
        );
    } else {
        const int64_t N = means2d.size(-2);
        at::DimVector conics_shape(image_dims);
        conics_shape.append({N, 3});
        at::DimVector colors_shape(image_dims);
        colors_shape.append({N, channels});
        at::DimVector opacities_shape(image_dims);
        opacities_shape.append({N});

        TORCH_CHECK(
            conics.sizes() == conics_shape,
            "conics must have shape [..., N, 3], got ",
            conics.sizes()
        );
        TORCH_CHECK(
            colors.sizes() == colors_shape,
            "colors must have shape [..., N, channels], got ",
            colors.sizes()
        );
        TORCH_CHECK(
            opacities.sizes() == opacities_shape,
            "opacities must have shape [..., N], got ",
            opacities.sizes()
        );
    }

    if (backgrounds.has_value()) {
        at::DimVector backgrounds_shape(image_dims);
        backgrounds_shape.append({channels});
        TORCH_CHECK(
            backgrounds.value().sizes() == backgrounds_shape,
            "backgrounds must have shape [..., channels], got ",
            backgrounds.value().sizes()
        );
    }
    if (masks.has_value()) {
        TORCH_CHECK(
            masks.value().sizes() == isect_offsets.sizes(),
            "masks must have the same shape as isect_offsets; got masks ",
            masks.value().sizes(),
            " and isect_offsets ",
            isect_offsets.sizes()
        );
    }

    const int64_t tile_height = isect_offsets.size(-2);
    const int64_t tile_width = isect_offsets.size(-1);
    TORCH_CHECK(
        tile_height * tile_size >= image_height,
        "Assert Failed: ",
        tile_height,
        " * ",
        tile_size,
        " >= ",
        image_height
    );
    TORCH_CHECK(
        tile_width * tile_size >= image_width,
        "Assert Failed: ",
        tile_width,
        " * ",
        tile_size,
        " >= ",
        image_width
    );
}

} // namespace

// Gradients of the differentiable forward inputs. Slots the backward kernel
// does not produce stay default (undefined Tensor / nullopt).
struct RasterizeToPixels3DGSBwdResult {
    at::optional<at::Tensor> v_means2d_abs;
    at::Tensor v_means2d;
    at::Tensor v_conics;
    at::Tensor v_colors;
    at::Tensor v_opacities;
    at::Tensor v_backgrounds;
};

RasterizeToPixels3DGSFwdResult rasterize_to_pixels_3dgs_fwd(
    // Gaussian parameters
    const at::Tensor &means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor &conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor &colors,    // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities, // [..., N]  or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width, int64_t image_height, int64_t tile_size,
    // intersections
    const at::Tensor &isect_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,   // [n_isects]
    bool packed,
    bool absgrad
) {
    check_rasterize_to_pixels_3dgs_inputs(
        means2d, conics, colors, opacities,
        backgrounds, masks,
        image_width, image_height, tile_size,
        isect_offsets, packed
    );

    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(isect_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    at::TensorOptions opt = means2d.options();
    at::DimVector image_dims(
        isect_offsets.sizes().slice(0, isect_offsets.dim() - 2)
    );
    uint32_t channels = colors.size(-1);

    at::DimVector renders_dims(image_dims);
    renders_dims.append({image_height, image_width, channels});
    at::Tensor renders = at::empty(renders_dims, opt);

    at::DimVector alphas_dims(image_dims);
    alphas_dims.append({image_height, image_width, 1});
    at::Tensor alphas = at::empty(alphas_dims, opt);

    at::DimVector last_ids_dims(image_dims);
    last_ids_dims.append({image_height, image_width});
    at::Tensor last_ids = at::empty(last_ids_dims, opt.dtype(at::kInt));

    launch_rasterize_to_pixels_3dgs_fwd_kernel(
        means2d, conics, colors, opacities,
        backgrounds, masks,
        image_width, image_height, tile_size,
        isect_offsets, flatten_ids,
        renders, alphas, last_ids
    );

    at::Tensor absgrad_holder = absgrad ? at::zeros_like(means2d)
                                        : at::empty({0}, means2d.options());

    return RasterizeToPixels3DGSFwdResult{
        .renders = renders,
        .alphas = alphas,
        .last_ids = last_ids,
        .means2d_absgrad = absgrad_holder,
    };
}

// Gradients of the differentiable forward outputs.
struct RasterizeToPixels3DGSGrad {
    static constexpr bool is_grad_bundle = true;
    at::Tensor renders;
    at::Tensor alphas;
};

// Full backward for rasterize_to_pixels_3dgs.
RasterizeToPixels3DGSBwdResult rasterize_to_pixels_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor &means2d,             // [..., N, 2] or [nnz, 2]
    const at::Tensor &conics,              // [..., N, 3] or [nnz, 3]
    const at::Tensor &colors,              // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities,           // [..., N] or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor &render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor &last_ids,      // [..., image_height, image_width]
    // AbsGrad holder (in-place destination for the published abs gradient)
    at::Tensor absgrad_holder,
    // scalars
    int64_t image_width, int64_t image_height, int64_t tile_size,
    bool absgrad,
    const RasterizeToPixels3DGSGrad &grad
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(grad.renders);
    CHECK_INPUT(grad.alphas);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = at::zeros_like(means2d);
    }

    launch_rasterize_to_pixels_3dgs_bwd_kernel(
        means2d, conics, colors, opacities,
        backgrounds, masks,
        image_width, image_height, tile_size,
        tile_offsets, flatten_ids,
        render_alphas, last_ids,
        grad.renders, grad.alphas,
        as_optional_tensor(v_means2d_abs),
        v_means2d, v_conics, v_colors, v_opacities
    );

    // --- Publish optional AbsGrad output ------------------------------
    // absgrad_holder was returned (mark_non_differentiable) from forward; the
    // caller observes the abs gradient through this in-place write.
    if (absgrad) {
        absgrad_holder.copy_(v_means2d_abs);
    }

    // --- Compute background gradient ----------------------------------
    // Background contribution is not produced by the CUDA backward kernel.
    // It is the incoming color gradient weighted by the unoccluded alpha.
    at::Tensor v_backgrounds;
    if (tensor_requires_grad(backgrounds)) {
        const at::Tensor one_minus_alpha =
            at::sub(at::ones_like(render_alphas).to(at::kFloat),
                    render_alphas.to(at::kFloat));
        v_backgrounds = at::mul(grad.renders, one_minus_alpha).sum({-3, -2});
    }

    return RasterizeToPixels3DGSBwdResult{
        .v_means2d_abs = as_optional_tensor(v_means2d_abs),
        .v_means2d = v_means2d,
        .v_conics = v_conics,
        .v_colors = v_colors,
        .v_opacities = v_opacities,
        .v_backgrounds = v_backgrounds,
    };
}

namespace {

class RasterizeToPixels3DGSAutograd
    : public torch::autograd::Function<RasterizeToPixels3DGSAutograd> {
public:
    // Forward-input positions; COUNT sizes the returned grad list.
    struct FwdInput {
        enum {
            MEANS2D, CONICS, COLORS, OPACITIES,
            BACKGROUNDS, MASKS,
            IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE,
            ISECT_OFFSETS, FLATTEN_IDS,
            PACKED, ABSGRAD,
            COUNT,
        };
    };

    // Forward-output positions, in forward()'s return order.
    struct FwdOutput {
        enum { RENDERS, ALPHAS, ABSGRAD_HOLDER, COUNT };
    };

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &means2d, const at::Tensor &conics,
        const at::Tensor &colors, const at::Tensor &opacities,
        const at::optional<at::Tensor> &backgrounds,
        const at::optional<at::Tensor> &masks,
        int64_t image_width, int64_t image_height, int64_t tile_size,
        const at::Tensor &isect_offsets, const at::Tensor &flatten_ids,
        bool packed, bool absgrad
    ) {
        static_assert(
            FwdInput::COUNT == fwd_input_count<&forward>(),
            "FwdInput must have one enumerator per forward input"
        );

        // --- Run forward --------------------------------------------------
        RasterizeToPixels3DGSFwdResult outputs =
            rasterize_to_pixels_3dgs_fwd(
                means2d, conics, colors, opacities,
                backgrounds, masks,
                image_width, image_height, tile_size,
                isect_offsets, flatten_ids,
                packed, absgrad
            );

        // --- Publish optional public outputs ------------------------------
        at::Tensor absgrad_holder = outputs.means2d_absgrad;
        ctx->mark_non_differentiable({absgrad_holder});

        // --- Save state for backward --------------------------------------
        ctx_save<&rasterize_to_pixels_3dgs_bwd>(
            ctx,
            means2d, conics, colors, opacities, backgrounds, masks,
            isect_offsets, flatten_ids,
            outputs.alphas, outputs.last_ids, absgrad_holder,
            image_width, image_height, tile_size, absgrad
        );

        torch::autograd::variable_list out(FwdOutput::COUNT);
        out[FwdOutput::RENDERS]        = outputs.renders;
        out[FwdOutput::ALPHAS]         = outputs.alphas.to(at::kFloat);
        out[FwdOutput::ABSGRAD_HOLDER] = absgrad_holder;
        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        RasterizeToPixels3DGSGrad grad{
            .renders = grad_outputs[FwdOutput::RENDERS].contiguous(),
            .alphas  = grad_outputs[FwdOutput::ALPHAS].contiguous(),
        };
        RasterizeToPixels3DGSBwdResult g = apply_bwd<&rasterize_to_pixels_3dgs_bwd>(ctx, grad);

        torch::autograd::variable_list grads(FwdInput::COUNT);
        grads[FwdInput::MEANS2D]     = g.v_means2d;
        grads[FwdInput::CONICS]      = g.v_conics;
        grads[FwdInput::COLORS]      = g.v_colors;
        grads[FwdInput::OPACITIES]   = g.v_opacities;
        grads[FwdInput::BACKGROUNDS] = g.v_backgrounds;
        return grads;
    }

    // Forwards to apply() and packs the variable_list into the typed result.
    // forward() already exposes alphas as float, so no cast is needed here.
    static RasterizeToPixels3DGSResult call(
        const at::Tensor &means2d, const at::Tensor &conics,
        const at::Tensor &colors, const at::Tensor &opacities,
        const at::optional<at::Tensor> &backgrounds,
        const at::optional<at::Tensor> &masks,
        int64_t image_width, int64_t image_height, int64_t tile_size,
        const at::Tensor &isect_offsets, const at::Tensor &flatten_ids,
        bool packed, bool absgrad
    ) {
        torch::autograd::variable_list outputs = apply(
            means2d, conics,
            colors, opacities,
            backgrounds, masks,
            image_width, image_height, tile_size,
            isect_offsets, flatten_ids,
            packed, absgrad
        );
        return {
            .renders = outputs[FwdOutput::RENDERS],
            .alphas = outputs[FwdOutput::ALPHAS],
            .means2d_absgrad = outputs[FwdOutput::ABSGRAD_HOLDER],
        };
    }
};

} // namespace

RasterizeToPixels3DGSResult rasterize_to_pixels_3dgs(
    const at::Tensor &means2d, const at::Tensor &conics,
    const at::Tensor &colors, const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::Tensor &isect_offsets, const at::Tensor &flatten_ids,
    bool packed, bool absgrad
) {
    const bool use_custom_autograd = needs_custom_autograd(
        means2d, conics, colors, opacities, backgrounds, masks,
        isect_offsets, flatten_ids
    );
    if (!use_custom_autograd) {
        RasterizeToPixels3DGSFwdResult fwd = rasterize_to_pixels_3dgs_fwd(
            means2d, conics,
            colors, opacities,
            backgrounds, masks,
            image_width, image_height, tile_size,
            isect_offsets, flatten_ids,
            packed, absgrad
        );
        // last_ids is forward-internal state; alphas is exposed as float.
        return {
            .renders = fwd.renders,
            .alphas = fwd.alphas.to(at::kFloat),
            .means2d_absgrad = fwd.means2d_absgrad,
        };
    }

    return RasterizeToPixels3DGSAutograd::call(
        means2d, conics,
        colors, opacities,
        backgrounds, masks,
        image_width, image_height, tile_size,
        isect_offsets, flatten_ids,
        packed, absgrad
    );
}

RasterizeToIndices3DGSResult rasterize_to_indices_3dgs(
    int64_t range_start,
    int64_t range_end,        // iteration steps
    const at::Tensor &transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor &means2d,   // [..., N, 2]
    const at::Tensor &conics,    // [..., N, 3]
    const at::Tensor &opacities, // [..., N]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    at::TensorOptions opt = means2d.options();
    uint32_t N = means2d.size(-2); // number of gaussians
    uint32_t I = (N == 0) ? 0 : means2d.numel() / (2 * N); // number of images

    uint32_t n_isects = flatten_ids.size(0);

    // First pass: count the number of gaussians that contribute to each pixel
    int64_t n_elems;
    at::Tensor chunk_starts;
    if (n_isects) {
        at::Tensor chunk_cnts = at::zeros(
            {I * image_height * image_width}, opt.dtype(at::kInt)
        );
        launch_rasterize_to_indices_3dgs_kernel(
            range_start,
            range_end,
            transmittances,
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
            c10::nullopt, // chunk_starts
            at::optional<at::Tensor>(chunk_cnts),
            c10::nullopt, // gaussian_ids
            c10::nullopt  // pixel_ids
        );
        at::Tensor cumsum = at::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
        n_elems = cumsum[-1].item<int64_t>();
        chunk_starts = at::sub(cumsum, chunk_cnts);
    } else {
        n_elems = 0;
    }

    // Second pass: allocate memory and write out the gaussian and pixel ids.
    at::Tensor gaussian_ids = at::empty({n_elems}, opt.dtype(at::kLong));
    at::Tensor pixel_ids = at::empty({n_elems}, opt.dtype(at::kLong));
    if (n_elems) {
        launch_rasterize_to_indices_3dgs_kernel(
            range_start,
            range_end,
            transmittances,
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
            at::optional<at::Tensor>(chunk_starts),
            c10::nullopt, // chunk_cnts
            at::optional<at::Tensor>(gaussian_ids),
            at::optional<at::Tensor>(pixel_ids)
        );
    }
    return {.gaussian_ids = gaussian_ids, .pixel_ids = pixel_ids};
}

std::tuple<at::Tensor, at::Tensor> rasterize_num_contributing_gaussians(
    const at::Tensor &means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor &conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor &opacities, // [..., N] or [nnz]
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    auto opt = means2d.options();
    at::DimVector out_dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    out_dims.append({image_height, image_width});
    at::Tensor num_contributing = at::empty(out_dims, opt.dtype(at::kInt));
    at::Tensor alphas = at::empty(out_dims, opt);

    launch_rasterize_num_contributing_gaussians_kernel(
        means2d,
        conics,
        opacities,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        num_contributing,
        alphas
    );

    return std::make_tuple(num_contributing, alphas);
}

std::tuple<at::Tensor, at::Tensor> rasterize_contributing_gaussian_ids(
    const at::Tensor &means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor &conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor &opacities, // [..., N] or [nnz]
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &num_contributing_gaussians // [..., image_height, image_width]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(num_contributing_gaussians);
    TORCH_CHECK_VALUE(
        num_contributing_gaussians.scalar_type() == at::kInt,
        "num_contributing_gaussians must have dtype int32"
    );
    TORCH_CHECK_VALUE(
        tile_offsets.dim() >= 2,
        "tile_offsets must have at least two dimensions"
    );
    TORCH_CHECK_VALUE(
        num_contributing_gaussians.dim() == tile_offsets.dim(),
        "num_contributing_gaussians must have the same number of dimensions as tile_offsets"
    );
    for (int64_t d = 0; d < tile_offsets.dim() - 2; ++d) {
        TORCH_CHECK_VALUE(
            num_contributing_gaussians.size(d) == tile_offsets.size(d),
            "num_contributing_gaussians batch dimensions must match tile_offsets"
        );
    }
    TORCH_CHECK_VALUE(
        num_contributing_gaussians.size(-2) == image_height,
        "num_contributing_gaussians height must match image_height"
    );
    TORCH_CHECK_VALUE(
        num_contributing_gaussians.size(-1) == image_width,
        "num_contributing_gaussians width must match image_width"
    );

    int64_t max_num_contributing = 0;
    if (num_contributing_gaussians.numel() > 0) {
        const int32_t min_num_contributing =
            num_contributing_gaussians.min().item<int32_t>();
        TORCH_CHECK_VALUE(
            min_num_contributing >= 0,
            "num_contributing_gaussians must be non-negative"
        );
        max_num_contributing =
            num_contributing_gaussians.max().item<int32_t>();
    }

    auto opt = means2d.options();
    at::DimVector out_dims(num_contributing_gaussians.sizes());
    out_dims.append({max_num_contributing});
    at::Tensor ids = at::full(out_dims, -1, opt.dtype(at::kInt));
    at::Tensor weights = at::zeros(out_dims, opt);

    if (max_num_contributing > 0) {
        launch_rasterize_contributing_gaussian_ids_kernel(
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            tile_size,
            static_cast<uint32_t>(max_num_contributing),
            tile_offsets,
            flatten_ids,
            ids,
            weights
        );
    }

    return std::make_tuple(ids, weights);
}

std::tuple<at::Tensor, at::Tensor> rasterize_top_contributing_gaussian_ids(
    const at::Tensor &means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor &conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor &opacities, // [..., N] or [nnz]
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    int64_t num_depth_samples
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    TORCH_CHECK_VALUE(
        num_depth_samples > 0, "num_depth_samples must be greater than 0"
    );

    auto opt = means2d.options();
    at::DimVector out_dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    out_dims.append({image_height, image_width, num_depth_samples});
    at::Tensor ids = at::empty(out_dims, opt.dtype(at::kInt));
    at::Tensor weights = at::empty(out_dims, opt);

    launch_rasterize_top_contributing_gaussian_ids_kernel(
        means2d,
        conics,
        opacities,
        image_width,
        image_height,
        tile_size,
        num_depth_samples,
        tile_offsets,
        flatten_ids,
        ids,
        weights
    );

    return std::make_tuple(ids, weights);
}

#endif


#if GSPLAT_BUILD_2DGS

////////////////////////////////////////////////////
// 2DGS
////////////////////////////////////////////////////

namespace {

void check_rasterize_to_pixels_2dgs_inputs(
    const at::Tensor &means2d, const at::Tensor &ray_transforms,
    const at::Tensor &colors, const at::Tensor &opacities,
    const at::Tensor &normals, const at::Tensor &densify,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::Tensor &tile_offsets,
    bool packed
) {
    TORCH_CHECK(
        means2d.dim() >= 2,
        "means2d must have shape [..., N, 2] or [nnz, 2], got ",
        means2d.sizes()
    );
    TORCH_CHECK(
        colors.dim() >= 2,
        "colors must have shape [..., N, channels] or [nnz, channels], got ",
        colors.sizes()
    );
    TORCH_CHECK(
        tile_offsets.dim() >= 2,
        "tile_offsets must have shape [..., tile_height, tile_width], got ",
        tile_offsets.sizes()
    );
    TORCH_CHECK(
        means2d.size(-1) == 2,
        "means2d must have shape [..., N, 2] or [nnz, 2], got ",
        means2d.sizes()
    );

    at::DimVector image_dims(means2d.sizes().slice(0, means2d.dim() - 2));
    const int64_t channels = colors.size(-1);
    if (packed) {
        const int64_t nnz = means2d.size(0);
        TORCH_CHECK(
            means2d.dim() == 2 && means2d.size(0) == nnz &&
                means2d.size(1) == 2,
            "packed means2d must have shape [nnz, 2], got ",
            means2d.sizes()
        );
        TORCH_CHECK(
            ray_transforms.dim() == 3 && ray_transforms.size(0) == nnz &&
                ray_transforms.size(1) == 3 && ray_transforms.size(2) == 3,
            "packed ray_transforms must have shape [nnz, 3, 3], got ",
            ray_transforms.sizes()
        );
        TORCH_CHECK(
            colors.size(0) == nnz,
            "packed colors first dimension must match nnz; got colors ",
            colors.sizes(),
            " and nnz ",
            nnz
        );
        TORCH_CHECK(
            opacities.dim() == 1 && opacities.size(0) == nnz,
            "packed opacities must have shape [nnz], got ",
            opacities.sizes()
        );
        TORCH_CHECK(
            normals.dim() == 2 && normals.size(0) == nnz &&
                normals.size(1) == 3,
            "packed normals must have shape [nnz, 3], got ",
            normals.sizes()
        );
        TORCH_CHECK(
            densify.dim() == 2 && densify.size(0) == nnz &&
                densify.size(1) == 2,
            "packed densify must have shape [nnz, 2], got ",
            densify.sizes()
        );
    } else {
        const int64_t N = means2d.size(-2);
        at::DimVector ray_transforms_shape(image_dims);
        ray_transforms_shape.append({N, 3, 3});
        at::DimVector colors_shape(image_dims);
        colors_shape.append({N, channels});
        at::DimVector opacities_shape(image_dims);
        opacities_shape.append({N});
        at::DimVector normals_shape(image_dims);
        normals_shape.append({N, 3});
        at::DimVector densify_shape(image_dims);
        densify_shape.append({N, 2});

        TORCH_CHECK(
            ray_transforms.sizes() == ray_transforms_shape,
            "ray_transforms must have shape [..., N, 3, 3], got ",
            ray_transforms.sizes()
        );
        TORCH_CHECK(
            colors.sizes() == colors_shape,
            "colors must have shape [..., N, channels], got ",
            colors.sizes()
        );
        TORCH_CHECK(
            opacities.sizes() == opacities_shape,
            "opacities must have shape [..., N], got ",
            opacities.sizes()
        );
        TORCH_CHECK(
            normals.sizes() == normals_shape,
            "normals must have shape [..., N, 3], got ",
            normals.sizes()
        );
        TORCH_CHECK(
            densify.sizes() == densify_shape,
            "densify must have shape [..., N, 2], got ",
            densify.sizes()
        );
    }

    if (backgrounds.has_value()) {
        at::DimVector backgrounds_shape(image_dims);
        backgrounds_shape.append({channels});
        TORCH_CHECK(
            backgrounds.value().sizes() == backgrounds_shape,
            "backgrounds must have shape [..., channels], got ",
            backgrounds.value().sizes()
        );
    }
    if (masks.has_value()) {
        TORCH_CHECK(
            masks.value().sizes() == tile_offsets.sizes(),
            "masks must have the same shape as tile_offsets; got masks ",
            masks.value().sizes(),
            " and tile_offsets ",
            tile_offsets.sizes()
        );
    }

    const int64_t tile_height = tile_offsets.size(-2);
    const int64_t tile_width = tile_offsets.size(-1);
    TORCH_CHECK(
        tile_height * tile_size >= image_height,
        "Assert Failed: ",
        tile_height,
        " * ",
        tile_size,
        " >= ",
        image_height
    );
    TORCH_CHECK(
        tile_width * tile_size >= image_width,
        "Assert Failed: ",
        tile_width,
        " * ",
        tile_size,
        " >= ",
        image_width
    );
}

} // namespace

struct RasterizeToPixels2DGSFwdResult {
    at::Tensor renders;
    at::Tensor alphas;
    at::Tensor render_normals;
    at::Tensor render_distort;
    at::Tensor render_median;
    at::Tensor last_ids;
    at::Tensor median_ids;
    at::Tensor means2d_absgrad;
};

template <> struct TorchArgDef<RasterizeToPixels2DGSFwdResult> {
    // last_ids / median_ids are forward-internal state; alphas is exposed
    // as float.
    static auto to(const RasterizeToPixels2DGSFwdResult &r) { return to_torch_args(
        r.renders, r.alphas.to(at::kFloat), r.render_normals,
        r.render_distort, r.render_median, r.means2d_absgrad
    ); }
};


RasterizeToPixels2DGSFwdResult
rasterize_to_pixels_2dgs_fwd(
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor &colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities,      // [..., N]  or [nnz]
    const at::Tensor &normals,        // [..., N, 3] or [nnz, 3]
    const at::Tensor &densify,        // [..., N, 2] or [nnz, 2]
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    bool packed,
    bool absgrad,
    bool distloss
) {
    (void)distloss;
    check_rasterize_to_pixels_2dgs_inputs(
        means2d, ray_transforms,
        colors, opacities,
        normals, densify,
        backgrounds, masks,
        image_width, image_height, tile_size,
        tile_offsets, packed
    );

    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(normals);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    auto opt = means2d.options();

    at::DimVector image_dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    uint32_t channels = colors.size(-1);

    at::DimVector renders_dims(image_dims);
    renders_dims.append({image_height, image_width, channels});
    at::Tensor renders = at::empty(renders_dims, opt);

    at::DimVector alphas_dims(image_dims);
    alphas_dims.append({image_height, image_width, 1});
    at::Tensor alphas = at::empty(alphas_dims, opt);

    at::DimVector last_ids_dims(image_dims);
    last_ids_dims.append({image_height, image_width});
    at::Tensor last_ids = at::empty(last_ids_dims, opt.dtype(at::kInt));

    at::DimVector median_ids_dims(image_dims);
    median_ids_dims.append({image_height, image_width});
    at::Tensor median_ids = at::empty(median_ids_dims, opt.dtype(at::kInt));

    at::DimVector render_normals_dims(image_dims);
    render_normals_dims.append({image_height, image_width, 3});
    at::Tensor render_normals = at::empty(render_normals_dims, opt);

    at::DimVector render_distort_dims(image_dims);
    render_distort_dims.append({image_height, image_width, 1});
    at::Tensor render_distort = at::empty(render_distort_dims, opt);

    at::DimVector render_median_dims(image_dims);
    render_median_dims.append({image_height, image_width, 1});
    at::Tensor render_median = at::empty(render_median_dims, opt);

    launch_rasterize_to_pixels_2dgs_fwd_kernel(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        renders,
        alphas,
        render_normals,
        render_distort,
        render_median,
        last_ids,
        median_ids
    );

    return RasterizeToPixels2DGSFwdResult{
        .renders = renders,
        .alphas = alphas,
        .render_normals = render_normals,
        .render_distort = render_distort,
        .render_median = render_median,
        .last_ids = last_ids,
        .median_ids = median_ids,
        .means2d_absgrad = absgrad ? at::zeros_like(means2d)
                                   : at::empty({0}, means2d.options())
    };
}

// Gradients of the differentiable forward inputs. Slots the backward kernel
// does not produce stay default (undefined Tensor / nullopt).
struct RasterizeToPixels2DGSBwdResult {
    at::optional<at::Tensor> v_means2d_abs;
    at::Tensor v_means2d;
    at::Tensor v_ray_transforms;
    at::Tensor v_colors;
    at::Tensor v_opacities;
    at::Tensor v_normals;
    at::Tensor v_densify;
    at::Tensor v_backgrounds;
};

// Gradients of the differentiable forward outputs.
struct RasterizeToPixels2DGSGrad {
    static constexpr bool is_grad_bundle = true;
    at::Tensor renders;
    at::Tensor alphas;
    at::Tensor render_normals;
    at::Tensor render_distort;
    at::Tensor render_median;
};

// Full backward for rasterize_to_pixels_2dgs.
RasterizeToPixels2DGSBwdResult
rasterize_to_pixels_2dgs_bwd(
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor &colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities,      // [..., N] or [nnz]
    const at::Tensor &normals,        // [..., N, 3] or [nnz, 3]
    const at::Tensor &densify,
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // ray_crossions
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor &render_colors, // [..., image_height, image_width, channels]
    const at::Tensor &render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor &last_ids,      // [..., image_height, image_width]
    const at::Tensor &median_ids,    // [..., image_height, image_width]
    at::Tensor absgrad_holder, // in-place destination for the published abs gradient
    bool absgrad,
    // gradients of outputs
    const RasterizeToPixels2DGSGrad &grad
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(normals);
    CHECK_INPUT(densify);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_colors);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(median_ids);
    CHECK_INPUT(grad.renders);
    CHECK_INPUT(grad.alphas);
    CHECK_INPUT(grad.render_normals);
    CHECK_INPUT(grad.render_distort);
    CHECK_INPUT(grad.render_median);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_ray_transforms = at::zeros_like(ray_transforms);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_normals = at::zeros_like(normals);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = at::zeros_like(means2d);
    }
    at::Tensor v_densify = at::zeros_like(densify);

    launch_rasterize_to_pixels_2dgs_bwd_kernel(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        render_colors,
        render_alphas,
        last_ids,
        median_ids,
        grad.renders,
        grad.alphas,
        grad.render_normals,
        grad.render_distort,
        grad.render_median,
        absgrad ? c10::optional<at::Tensor>(v_means2d_abs) : c10::nullopt,
        v_means2d,
        v_ray_transforms,
        v_colors,
        v_opacities,
        v_normals,
        v_densify
    );

    // --- Publish optional AbsGrad output ------------------------------
    // absgrad_holder was returned (mark_non_differentiable) from forward; the
    // caller observes the abs gradient through this in-place write.
    if (absgrad) {
        absgrad_holder.copy_(v_means2d_abs);
    }

    // --- Compute background gradient ----------------------------------
    // Background contribution is not produced by the CUDA backward kernel.
    // It is the incoming color gradient weighted by the unoccluded alpha.
    at::Tensor v_backgrounds;
    if (tensor_requires_grad(backgrounds)) {
        const at::Tensor one_minus_alpha =
            at::sub(at::ones_like(render_alphas).to(at::kFloat),
                    render_alphas.to(at::kFloat));
        v_backgrounds = at::mul(grad.renders, one_minus_alpha).sum({-3, -2});
    }

    return RasterizeToPixels2DGSBwdResult{
        .v_means2d_abs = as_optional_tensor(v_means2d_abs),
        .v_means2d = v_means2d,
        .v_ray_transforms = v_ray_transforms,
        .v_colors = v_colors,
        .v_opacities = v_opacities,
        .v_normals = v_normals,
        .v_densify = v_densify,
        .v_backgrounds = v_backgrounds,
    };
}

template <> struct TorchArgDef<RasterizeToPixels2DGSResult> {
    static auto to(const RasterizeToPixels2DGSResult &r) { return to_torch_args(
        r.renders, r.alphas, r.render_normals, r.render_distort,
        r.render_median, r.means2d_absgrad
    ); }
};

namespace {

class RasterizeToPixels2DGSAutograd
    : public torch::autograd::Function<RasterizeToPixels2DGSAutograd> {
public:
    // Forward-input positions; COUNT sizes the returned grad list.
    struct FwdInput {
        enum {
            MEANS2D, RAY_TRANSFORMS, COLORS, OPACITIES,
            NORMALS, DENSIFY,
            BACKGROUNDS, MASKS,
            IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE,
            TILE_OFFSETS, FLATTEN_IDS,
            PACKED, ABSGRAD, DISTLOSS,
            COUNT,
        };
    };

    // Forward-output positions, in forward()'s return order.
    struct FwdOutput {
        enum {
            RENDERS, ALPHAS, RENDER_NORMALS, RENDER_DISTORT, RENDER_MEDIAN,
            ABSGRAD_HOLDER, COUNT,
        };
    };

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &means2d, const at::Tensor &ray_transforms,
        const at::Tensor &colors, const at::Tensor &opacities,
        const at::Tensor &normals, const at::Tensor &densify,
        const at::optional<at::Tensor> &backgrounds,
        const at::optional<at::Tensor> &masks,
        int64_t image_width, int64_t image_height, int64_t tile_size,
        const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
        bool packed, bool absgrad, bool distloss
    ) {
        static_assert(
            FwdInput::COUNT == fwd_input_count<&forward>(),
            "FwdInput must have one enumerator per forward input"
        );

        // --- Run forward --------------------------------------------------
        RasterizeToPixels2DGSFwdResult outputs =
            rasterize_to_pixels_2dgs_fwd(
                means2d, ray_transforms,
                colors, opacities,
                normals, densify,
                backgrounds, masks,
                image_width, image_height, tile_size,
                tile_offsets, flatten_ids,
                packed, absgrad, distloss
            );

        // --- Publish optional public outputs ------------------------------
        at::Tensor absgrad_holder = outputs.means2d_absgrad;
        ctx->mark_non_differentiable({absgrad_holder});

        // --- Save state for backward --------------------------------------
        ctx_save<&rasterize_to_pixels_2dgs_bwd>(
            ctx,
            means2d, ray_transforms, colors, opacities, normals, densify,
            backgrounds, masks, image_width, image_height, tile_size,
            tile_offsets, flatten_ids,
            outputs.renders, outputs.alphas, outputs.last_ids,
            outputs.median_ids, absgrad_holder, absgrad
        );

        torch::autograd::variable_list out(FwdOutput::COUNT);
        out[FwdOutput::RENDERS]        = outputs.renders;
        out[FwdOutput::ALPHAS]         = outputs.alphas.to(at::kFloat);
        out[FwdOutput::RENDER_NORMALS] = outputs.render_normals;
        out[FwdOutput::RENDER_DISTORT] = outputs.render_distort;
        out[FwdOutput::RENDER_MEDIAN]  = outputs.render_median;
        out[FwdOutput::ABSGRAD_HOLDER] = absgrad_holder;
        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        RasterizeToPixels2DGSGrad grad{
            .renders        = grad_outputs[FwdOutput::RENDERS].contiguous(),
            .alphas         = grad_outputs[FwdOutput::ALPHAS].contiguous(),
            .render_normals = grad_outputs[FwdOutput::RENDER_NORMALS].contiguous(),
            .render_distort = grad_outputs[FwdOutput::RENDER_DISTORT].contiguous(),
            .render_median  = grad_outputs[FwdOutput::RENDER_MEDIAN].contiguous(),
        };
        RasterizeToPixels2DGSBwdResult g = apply_bwd<&rasterize_to_pixels_2dgs_bwd>(ctx, grad);

        torch::autograd::variable_list grads(FwdInput::COUNT);
        grads[FwdInput::MEANS2D]        = g.v_means2d;
        grads[FwdInput::RAY_TRANSFORMS] = g.v_ray_transforms;
        grads[FwdInput::COLORS]         = g.v_colors;
        grads[FwdInput::OPACITIES]      = g.v_opacities;
        grads[FwdInput::NORMALS]        = g.v_normals;
        grads[FwdInput::DENSIFY]        = g.v_densify;
        grads[FwdInput::BACKGROUNDS]    = g.v_backgrounds;
        return grads;
    }

    // Forwards to apply() and packs the variable_list into the typed result.
    static RasterizeToPixels2DGSResult call(
        const at::Tensor &means2d, const at::Tensor &ray_transforms,
        const at::Tensor &colors, const at::Tensor &opacities,
        const at::Tensor &normals, const at::Tensor &densify,
        const at::optional<at::Tensor> &backgrounds,
        const at::optional<at::Tensor> &masks,
        int64_t image_width, int64_t image_height, int64_t tile_size,
        const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
        bool packed, bool absgrad, bool distloss
    ) {
        torch::autograd::variable_list outputs = apply(
            means2d, ray_transforms,
            colors, opacities,
            normals, densify,
            backgrounds, masks,
            image_width, image_height, tile_size,
            tile_offsets, flatten_ids,
            packed, absgrad, distloss
        );
        return {
            .renders         = outputs[FwdOutput::RENDERS],
            .alphas          = outputs[FwdOutput::ALPHAS],
            .render_normals  = outputs[FwdOutput::RENDER_NORMALS],
            .render_distort  = outputs[FwdOutput::RENDER_DISTORT],
            .render_median   = outputs[FwdOutput::RENDER_MEDIAN],
            .means2d_absgrad = outputs[FwdOutput::ABSGRAD_HOLDER],
        };
    }
};

} // namespace

RasterizeToPixels2DGSResult rasterize_to_pixels_2dgs(
    const at::Tensor &means2d, const at::Tensor &ray_transforms,
    const at::Tensor &colors, const at::Tensor &opacities,
    const at::Tensor &normals, const at::Tensor &densify,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    bool packed, bool absgrad, bool distloss
) {
    const bool use_custom_autograd = needs_custom_autograd(
        means2d, ray_transforms, colors, opacities, normals, densify,
        backgrounds, masks, tile_offsets, flatten_ids
    );
    if (!use_custom_autograd) {
        RasterizeToPixels2DGSFwdResult fwd = rasterize_to_pixels_2dgs_fwd(
            means2d, ray_transforms,
            colors, opacities,
            normals, densify,
            backgrounds, masks,
            image_width, image_height, tile_size,
            tile_offsets, flatten_ids,
            packed, absgrad, distloss
        );
        // alphas is exposed as float (matches the autograd path).
        return {
            .renders         = fwd.renders,
            .alphas          = fwd.alphas.to(at::kFloat),
            .render_normals  = fwd.render_normals,
            .render_distort  = fwd.render_distort,
            .render_median   = fwd.render_median,
            .means2d_absgrad = fwd.means2d_absgrad,
        };
    }

    return RasterizeToPixels2DGSAutograd::call(
        means2d, ray_transforms,
        colors, opacities,
        normals, densify,
        backgrounds, masks,
        image_width, image_height, tile_size,
        tile_offsets, flatten_ids,
        packed, absgrad, distloss
    );
}

struct RasterizeToIndices2DGSResult {
    at::Tensor gaussian_ids;
    at::Tensor pixel_ids;
};

template <>
struct TorchArgDef<RasterizeToIndices2DGSResult> {
    static auto to(const RasterizeToIndices2DGSResult &r) {
        return to_torch_args(r.gaussian_ids, r.pixel_ids);
    }
};

RasterizeToIndices2DGSResult rasterize_to_indices_2dgs(
    int64_t range_start,
    int64_t range_end,        // iteration steps
    const at::Tensor &transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3]
    const at::Tensor &opacities,      // [..., N]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    auto opt = means2d.options();
    uint32_t N = means2d.size(-2); // number of gaussians
    uint32_t I = (N == 0) ? 0 : means2d.numel() / (2 * N); // number of images

    uint32_t n_isects = flatten_ids.size(0);

    // First pass: count the number of gaussians that contribute to each pixel
    int64_t n_elems;
    at::Tensor chunk_starts;
    if (n_isects) {
        at::Tensor chunk_cnts = at::zeros(
            {I * image_height * image_width}, opt.dtype(at::kInt)
        );
        launch_rasterize_to_indices_2dgs_kernel(
            range_start,
            range_end,
            transmittances,
            means2d,
            ray_transforms,
            opacities,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
            c10::nullopt, // chunk_starts
            at::optional<at::Tensor>(chunk_cnts),
            c10::nullopt, // gaussian_ids
            c10::nullopt  // pixel_ids
        );
        at::Tensor cumsum = at::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
        n_elems = cumsum[-1].item<int64_t>();
        chunk_starts = at::sub(cumsum, chunk_cnts);
    } else {
        n_elems = 0;
    }

    // Second pass: allocate memory and write out the gaussian and pixel ids.
    at::Tensor gaussian_ids = at::empty({n_elems}, opt.dtype(at::kLong));
    at::Tensor pixel_ids = at::empty({n_elems}, opt.dtype(at::kLong));
    if (n_elems) {
        launch_rasterize_to_indices_2dgs_kernel(
            range_start,
            range_end,
            transmittances,
            means2d,
            ray_transforms,
            opacities,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
            at::optional<at::Tensor>(chunk_starts),
            c10::nullopt, // chunk_cnts
            at::optional<at::Tensor>(gaussian_ids),
            at::optional<at::Tensor>(pixel_ids)
        );
    }
    return {.gaussian_ids = gaussian_ids, .pixel_ids = pixel_ids};
}

#endif

////////////////////////////////////////////////////
// 3DGS (from world)
////////////////////////////////////////////////////

// `ext.cpp` declares only the public schema; this translation unit owns the CUDA
// and AutogradCUDA implementations. Native C++ tests include Rasterization.h
// for the internal functions they intentionally exercise.

// The forward result includes public render outputs plus optional CSR-packed
// batch state that backward consumes to skip work it would otherwise recompute.
RasterizeToPixelsFromWorld3DGSFwdResult rasterize_to_pixels_from_world_3dgs_fwd(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> masks,       // [..., C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,              // [..., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    const RendererConfig renderer_config,
    const bool fwd_only,
    const bool return_last_ids,
    const at::optional<at::Tensor> sample_counts, // [..., C, image_height, image_width] optional
    const at::optional<at::Tensor> normals, // [..., C, image_height, image_width, 3] optional output tensor
    const bool unsafe_masked_tile_outputs
) {
#if !GSPLAT_BUILD_3DGUT
    TORCH_CHECK(
        false,
        "rasterize_to_pixels_from_world_3dgs_fwd requires "
        "GSPLAT_BUILD_3DGUT=1"
    );
    return {};
#else
    // --- Validate inputs ---------------------------------------------------
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    if (sample_counts.has_value()) {
        CHECK_INPUT(sample_counts.value());
    }
    TORCH_CHECK(
        !fwd_only || !return_last_ids,
        "fwd-only eval3d rasterization cannot return last_ids; "
        "request exact metadata instead"
    );
    TORCH_CHECK(
        !fwd_only || !sample_counts.has_value(),
        "fwd-only eval3d rasterization cannot return sample_counts; "
        "request exact metadata instead"
    );

    // --- Allocate public forward outputs ----------------------------------
    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    uint32_t C = viewmats0.size(-3);     // number of cameras
    // uint32_t N = means.size(-2);         // number of gaussians
    uint32_t channels = colors.size(-1);

    at::DimVector renders_shape(batch_dims);
    renders_shape.append({C, image_height, image_width, channels});
    at::Tensor renders = at::empty(renders_shape, opt);

    at::DimVector alphas_shape(batch_dims);
    alphas_shape.append({C, image_height, image_width, 1});
    at::Tensor alphas = at::empty(alphas_shape, opt);

    at::Tensor compose_c_stop;
    at::Tensor priming_state;
    const int32_t pixels_per_tile = tile_size * tile_size;
    const bool return_normals = normals.has_value();
    const bool needs_exact_metadata = !fwd_only;
    const bool needs_last_ids = needs_exact_metadata || return_last_ids;
    at::Tensor last_ids;
    if (needs_last_ids) {
        at::DimVector last_ids_shape(batch_dims);
        last_ids_shape.append({C, image_height, image_width});
        last_ids = at::empty(last_ids_shape, opt.dtype(at::kInt));
    }
    const bool needs_batch_state =
        !fwd_only || renderer_config == RendererConfig::PARALLEL_BATCH;

    // --- Build CSR batch structure + forward persistence buffer -----------
    // Non-fwd-only callers persist exact batch state because both MixedBatch and
    // ParallelBatch use batch-parallel backward, and metadata inspection needs
    // exact traversal results. ParallelBatch forward also needs the CSR/state
    // tensors for its batch-scan/batch-replay passes. Fwd-only MixedBatch is
    // the only path that can skip this blocking readback and allocation.
    const uint32_t tile_height = static_cast<uint32_t>(tile_offsets.size(-2));
    const uint32_t tile_width = static_cast<uint32_t>(tile_offsets.size(-1));
    int64_t batch_prod = 1;
    for (size_t d = 0; d < batch_dims.size(); ++d) {
        batch_prod *= batch_dims[d];
    }
    const uint32_t I =
        static_cast<uint32_t>(batch_prod) * C;  // number of images
    const uint32_t num_tiles = I * tile_height * tile_width;
    const int64_t n_isects = flatten_ids.size(0);

    at::Tensor batches_per_tile;
    at::Tensor batch_offsets;
    at::Tensor fwd_batch_state;
    int64_t total_batches = 0;
    if (needs_batch_state) {
        std::tie(batches_per_tile, batch_offsets, total_batches) =
            compute_batch_csr(
                tile_offsets, n_isects, num_tiles, pixels_per_tile, opt);

        // Per (tile, batch boundary, pixel), stores cumulative fwd state:
        // T, pix_out[CDIM], and optional normal_out[3]. Use `at::empty`: masked
        // tiles and padded pixels are never read by the matching backward walk.
        //
        // SOA layout: `[total_batches, state_dim, pixels_per_tile]`. The pixel
        // axis is fastest-varying, so warp-aligned reads/writes of one state
        // element across pixels coalesce instead of striding by state_dim.
        const int64_t state_dim =
            FWD_BATCH_STATE_PIX_OFFSET + channels +
            (return_normals ? FWD_BATCH_STATE_NORMAL_EXTRA : 0);
        fwd_batch_state = at::empty(
            {total_batches, state_dim, pixels_per_tile},
            opt.dtype(at::kFloat));
    }

    // --- Launch selected forward kernel -----------------------------------
    if (renderer_config == RendererConfig::MIXED_BATCH) {
        launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel(
            means, quats, scales, colors, opacities,
            backgrounds, masks,
            image_width, image_height, tile_size,
            viewmats0, viewmats1, Ks,
            camera_model, ut_params, rs_type,
            rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            ftheta_coeffs, lidar_coeffs, external_distortion_params,
            tile_offsets, flatten_ids, use_hit_distance,
            unsafe_masked_tile_outputs,
            batches_per_tile, batch_offsets, total_batches,
            renders, alphas, last_ids,
            sample_counts, normals, fwd_batch_state
        );
    } else {
        // The batch-parallel forward needs a transient per-batch summary
        // (last index and sample count per pixel) for its batch-scan pass. It is
        // not part of autograd state; backward consumes only fwd_batch_state.
        // Each ushort2 stores `(last_local_plus_one, count_low15)` for one
        // (batch, pixel). Pixel 0's count lane also reserves its high bit as
        // the per-CTA batch-replay flag; see FwdPartialsMetaView.
        at::Tensor partials_meta = at::empty(
            {total_batches, pixels_per_tile, 2},
            opt.dtype(at::kUInt16));

        // Transient per-pixel and per-batch handoff from batch-scan to
        // batch-replay. These are intentionally local to ParallelBatch fwd:
        // MixedBatch runs the serial forward path and never needs them.
        const int64_t num_tiles_for_batch_replay = batches_per_tile.size(0);
        at::Tensor batch_replay_preamble;
        if (needs_exact_metadata) {
            batch_replay_preamble = at::empty(
                {num_tiles_for_batch_replay, pixels_per_tile, 2},
                opt.dtype(at::kInt));
            compose_c_stop = at::empty(
                {num_tiles_for_batch_replay, pixels_per_tile},
                opt.dtype(at::kUInt16));
        }
        // Per-pixel priming-chain state for ParallelBatch partials. Each int32
        // packs `(!sat, float16 T_cum, uint16 -next_batch)`. Partials publish
        // with atomicMin, so saturated entries win first, then the smallest
        // non-negative T wins, and ties prefer the larger batch key through the
        // negated low lane. The initial state means "no writer yet" with T=1.
        // The same producer-side cap also protects `compose_c_stop`: its real
        // batch ids are one lower than priming's `next_batch` ids.
        const int32_t max_batches_per_tile =
            batches_per_tile.max().item<int32_t>();
        TORCH_CHECK(
            max_batches_per_tile <= kMaxParallelBatchesPerTile,
            "max batches per tile must be <= ",
            kMaxParallelBatchesPerTile,
            " for u16-packed ParallelBatch priming/replay state (got ",
            max_batches_per_tile,
            ")");
        const int32_t priming_state_init =
            static_cast<int32_t>(gsplat::priming::INIT_PACKED);
        at::DimVector priming_shape(batch_dims);
        priming_shape.append({C, image_height, image_width});
        priming_state = at::full(
            priming_shape, priming_state_init, opt.dtype(at::kInt));
        // Round-major launch permutation for the ParallelBatch partials and
        // batch-replay kernels. The persisted state remains tile-major; only
        // the order in which CTAs are issued changes.
        at::Tensor bid_to_slot = compute_bid_to_slot(
            batches_per_tile, batch_offsets, total_batches, opt);
        launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_fwd_kernel(
            means, quats, scales, colors, opacities,
            backgrounds, masks,
            image_width, image_height, tile_size,
            viewmats0, viewmats1, Ks,
            camera_model, ut_params, rs_type,
            rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            ftheta_coeffs, lidar_coeffs, external_distortion_params,
            tile_offsets, flatten_ids, use_hit_distance,
            unsafe_masked_tile_outputs,
            batches_per_tile, batch_offsets, bid_to_slot, total_batches,
            fwd_only,
            renders, alphas, last_ids,
            sample_counts, normals, fwd_batch_state, partials_meta,
            batch_replay_preamble, compose_c_stop, priming_state
        );
    }

    return {
        .renders = renders,
        .alphas = alphas,
        .last_ids = last_ids,
        .batches_per_tile = batches_per_tile,
        .batch_offsets = batch_offsets,
        .fwd_batch_state = fwd_batch_state,
        .compose_c_stop = compose_c_stop,
        .priming_state = priming_state,
    };
#endif // !GSPLAT_BUILD_3DGUT
}

#if GSPLAT_BUILD_3DGUT

namespace {

// Optional public eval3d outputs, allocated by the dispatch entry points so
// both the no-grad and autograd paths share shapes/dtypes; the forward kernel
// writes into them in place.
struct Eval3DOptionalOutputs {
    at::optional<at::Tensor> sample_counts;
    at::optional<at::Tensor> normals;
};

Eval3DOptionalOutputs
allocate_eval3d_optional_outputs(
    const at::Tensor &means,
    const at::Tensor &viewmats0,
    int64_t image_width,
    int64_t image_height,
    bool return_sample_counts,
    bool return_normals
) {
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    const int64_t C = viewmats0.size(-3);

    at::optional<at::Tensor> sample_counts = c10::nullopt;
    if (return_sample_counts) {
        at::DimVector sample_counts_shape(batch_dims);
        sample_counts_shape.append({C, image_height, image_width});
        sample_counts = at::empty(
            sample_counts_shape, means.options().dtype(at::kInt));
    }

    at::optional<at::Tensor> normals = c10::nullopt;
    if (return_normals) {
        at::DimVector normals_shape(batch_dims);
        normals_shape.append({C, image_height, image_width, 3});
        normals = at::empty(normals_shape, means.options().dtype(at::kFloat));
    }

    return {
        .sample_counts = sample_counts,
        .normals = normals,
    };
}

// Gradients of the differentiable forward inputs. Slots the backward kernel
// does not produce stay default (undefined Tensor / nullopt).
struct RasterizeBwdResult {
    at::Tensor v_means;
    at::Tensor v_quats;
    at::Tensor v_scales;
    at::Tensor v_colors;
    at::Tensor v_opacities;
    at::Tensor v_backgrounds;
    at::optional<at::Tensor> v_rays;
};

// Gradients of the differentiable forward outputs.
struct RasterizeFromWorldGrad {
    static constexpr bool is_grad_bundle = true;
    at::optional<at::Tensor> renders;
    at::optional<at::Tensor> alphas;
    at::optional<at::Tensor> normals;
};

RasterizeBwdResult rasterize_to_pixels_from_world_3dgs_bwd(
    // input tensors
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    const at::Tensor &viewmats0,
    const at::optional<at::Tensor> &viewmats1,
    const at::Tensor &Ks,
    const at::optional<at::Tensor> &rays,
    const at::optional<at::Tensor> &radial_coeffs,
    const at::optional<at::Tensor> &tangential_coeffs,
    const at::optional<at::Tensor> &thin_prism_coeffs,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    // forward outputs
    const at::Tensor &render_alphas,
    const at::Tensor &last_ids,
    // CSR batch structure persisted by forward
    const at::Tensor &batches_per_tile,
    const at::Tensor &batch_offsets,
    const at::Tensor &fwd_batch_state,
    const at::optional<at::Tensor> &compose_c_stop,
    // scalars
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    CameraModelType camera_model,
    ShutterType rs_type,
    bool use_hit_distance,
    bool return_normals,
    // custom-class params
    c10::intrusive_ptr<UnscentedTransformParameters> ut_params,
    c10::intrusive_ptr<FThetaCameraDistortionParameters> ftheta_coeffs,
    at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> lidar_coeffs,
    at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> external_distortion_params,
    const RasterizeFromWorldGrad &grad
) {
    // --- Materialize output gradients ---------------------------------
    // set_materialize_grads is false and the CUDA kernel expects dense
    // color/alpha gradients, so materialize zeros for any absent one. The
    // color grad is shaped like render_alphas with colors' channel count.
    at::DimVector color_grad_shape(render_alphas.sizes());
    color_grad_shape[color_grad_shape.size() - 1] = colors.size(-1);
    at::Tensor v_render_colors =
        tensor_or_zeros(grad.renders, color_grad_shape, colors.options())
            .contiguous();
    at::Tensor v_render_alphas =
        tensor_or_zeros_like(grad.alphas, render_alphas).contiguous();

    at::optional<at::Tensor> v_render_normals;
    if (grad.normals.has_value()) {
        v_render_normals = grad.normals.value().contiguous();
    } else if (return_normals) {
        at::DimVector normal_grad_shape(render_alphas.sizes());
        normal_grad_shape[normal_grad_shape.size() - 1] = 3;
        v_render_normals =
            at::zeros(normal_grad_shape, means.options().dtype(at::kFloat));
    }

    // --- Validate restored CUDA inputs --------------------------------
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    CHECK_INPUT(batches_per_tile);
    CHECK_INPUT(batch_offsets);
    CHECK_INPUT(fwd_batch_state);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    if (rays.has_value()) {
        CHECK_INPUT(rays.value());
    }
    if (v_render_normals.has_value()) {
        CHECK_INPUT(v_render_normals.value());
    }

    // compose_c_stop is defined only for the ParallelBatch forward; an
    // undefined tensor signals the kernel to use the terminal-slot stopping path.
    at::Tensor compose_c_stop_tensor =
        compose_c_stop.value_or(at::Tensor{});
    if (compose_c_stop_tensor.defined()) {
        CHECK_INPUT(compose_c_stop_tensor);
    }

    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_quats = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::optional<at::Tensor> v_rays =
        rays.has_value() ? at::optional<at::Tensor>(at::zeros_like(rays.value()))
                         : at::optional<at::Tensor>();

    // --- Launch batch-parallel backward kernel ------------------------
    launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_bwd_kernel(
        means, quats, scales, colors, opacities,
        backgrounds, masks,
        static_cast<uint32_t>(image_width), static_cast<uint32_t>(image_height),
        static_cast<uint32_t>(tile_size),
        viewmats0, viewmats1, Ks,
        camera_model, ut_params, rs_type,
        rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        ftheta_coeffs, lidar_coeffs, external_distortion_params,
        tile_offsets, flatten_ids,
        use_hit_distance,
        render_alphas, last_ids,
        v_render_colors, v_render_alphas, v_render_normals,
        batches_per_tile, batch_offsets,
        fwd_batch_state.size(0), fwd_batch_state,
        compose_c_stop_tensor,
        v_means, v_quats, v_scales, v_colors, v_opacities, v_rays
    );

    // --- Compute background gradient ----------------------------------
    // Background contribution is not produced by the CUDA backward kernel.
    // It is the incoming color gradient weighted by the unoccluded alpha.
    at::Tensor v_backgrounds = at::Tensor{};
    if (tensor_requires_grad(backgrounds)) {
        const at::Tensor one_minus_alpha =
            at::sub(
                at::ones_like(render_alphas).to(at::kFloat),
                render_alphas.to(at::kFloat));
        v_backgrounds =
            at::mul(v_render_colors, one_minus_alpha).sum({-3, -2});
    }

    return {
        .v_means = v_means,
        .v_quats = v_quats,
        .v_scales = v_scales,
        .v_colors = v_colors,
        .v_opacities = v_opacities,
        .v_backgrounds = v_backgrounds,
        .v_rays = v_rays,
    };
}

class RasterizeToPixelsFromWorld3DGSAutograd
    : public torch::autograd::Function<RasterizeToPixelsFromWorld3DGSAutograd> {
public:
    // Forward-input positions; COUNT sizes the returned grad list and mirrors
    // the public operator's forward argument order one-to-one.
    struct FwdInput {
        enum {
            MEANS, QUATS, SCALES, COLORS, OPACITIES,
            BACKGROUNDS, MASKS,
            IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE,
            VIEWMATS0, VIEWMATS1, KS, CAMERA_MODEL,
            UT_PARAMS, RS_TYPE,
            RAYS,
            RADIAL_COEFFS, TANGENTIAL_COEFFS, THIN_PRISM_COEFFS,
            FTHETA_COEFFS, LIDAR_COEFFS, EXTERNAL_DISTORTION_PARAMS,
            TILE_OFFSETS, FLATTEN_IDS,
            RETURN_SAMPLE_COUNTS, USE_HIT_DISTANCE, RETURN_NORMALS,
            RENDERER_CONFIG, RETURN_LAST_IDS,
            UNSAFE_MASKED_TILE_OUTPUTS,
            COUNT,
        };
    };
    static_assert(FwdInput::RAYS == 16, "FwdInput::RAYS position changed");

    // Forward-output positions, in forward()'s return order.
    struct FwdOutput {
        enum { RENDERS, ALPHAS, LAST_IDS, SAMPLE_COUNTS, NORMALS, COUNT };
    };

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        // Gaussian parameters
        const at::Tensor &means,     // [..., N, 3]
        const at::Tensor &quats,     // [..., N, 4]
        const at::Tensor &scales,    // [..., N, 3]
        const at::Tensor &colors,    // [..., C, N, channels] or [nnz, channels]
        const at::Tensor &opacities, // [..., C, N] or [nnz]
        const at::optional<at::Tensor> &backgrounds, // [..., C, channels]
        const at::optional<at::Tensor> &masks,       // [..., C, tile_height, tile_width]
        // image size
        int64_t image_width, int64_t image_height, int64_t tile_size,
        // camera
        const at::Tensor &viewmats0,               // [..., C, 4, 4]
        const at::optional<at::Tensor> &viewmats1, // [..., C, 4, 4] optional for rolling shutter
        const at::Tensor &Ks,                      // [..., C, 3, 3]
        int64_t camera_model,
        // unscented transform
        const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
        int64_t rs_type,
        const at::optional<at::Tensor> &rays,              // [..., C, H, W, 6]
        const at::optional<at::Tensor> &radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
        const at::optional<at::Tensor> &tangential_coeffs, // [..., C, 2] optional
        const at::optional<at::Tensor> &thin_prism_coeffs, // [..., C, 4] optional
        const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
        const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
        const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
        // intersections
        const at::Tensor &tile_offsets, // [..., C, tile_height, tile_width]
        const at::Tensor &flatten_ids,  // [n_isects]
        bool return_sample_counts,
        bool use_hit_distance,
        bool return_normals,
        int64_t renderer_config,
        bool return_last_ids,
        bool unsafe_masked_tile_outputs
    ) {
        static_assert(
            FwdInput::COUNT == fwd_input_count<&forward>(),
            "FwdInput must have one enumerator per forward input"
        );

        ctx->set_materialize_grads(false);
        const RendererConfig parsed_renderer_config =
            parse_renderer_config(renderer_config);
        const auto camera_model_enum = static_cast<CameraModelType>(camera_model);
        const auto rs_type_enum = static_cast<ShutterType>(rs_type);

        TORCH_CHECK(
            !(unsafe_masked_tile_outputs &&
              masks.has_value() &&
              backgrounds.has_value() &&
              backgrounds.value().requires_grad()),
            "unsafe_masked_tile_outputs is not supported with differentiable backgrounds"
        );

        // --- Allocate optional public outputs + run shared forward --------
        // Backward needs exact traversal metadata, so the autograd forward
        // always runs the full forward (fwd_only=false, return_last_ids=true)
        // regardless of what the caller requested as outputs.
        auto optional_outputs = allocate_eval3d_optional_outputs(
            means, viewmats0, image_width, image_height,
            return_sample_counts, return_normals
        );

        RasterizeToPixelsFromWorld3DGSFwdResult fwd =
            rasterize_to_pixels_from_world_3dgs_fwd(
                means, quats, scales, colors, opacities,
                backgrounds, masks,
                static_cast<uint32_t>(image_width),
                static_cast<uint32_t>(image_height),
                static_cast<uint32_t>(tile_size),
                viewmats0, viewmats1, Ks,
                camera_model_enum, ut_params, rs_type_enum,
                rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                ftheta_coeffs, lidar_coeffs, external_distortion_params,
                tile_offsets, flatten_ids,
                use_hit_distance, parsed_renderer_config,
                /*fwd_only=*/false, /*return_last_ids=*/true,
                optional_outputs.sample_counts, optional_outputs.normals,
                unsafe_masked_tile_outputs
            );

        // --- Save state for backward --------------------------------------
        // compose_c_stop is nullopt for MixedBatch; it is saved positionally as
        // an optional sentinel and backward only consumes it for ParallelBatch.
        // priming_state is forward-internal and never saved.
        ctx_save<&rasterize_to_pixels_from_world_3dgs_bwd>(
            ctx,
            means, quats, scales, colors, opacities, backgrounds, masks,
            viewmats0, viewmats1, Ks, rays, radial_coeffs, tangential_coeffs,
            thin_prism_coeffs, tile_offsets, flatten_ids,
            fwd.alphas, fwd.last_ids,
            fwd.batches_per_tile, fwd.batch_offsets, fwd.fwd_batch_state,
            as_optional_tensor(fwd.compose_c_stop),
            image_width, image_height, tile_size,
            camera_model_enum, rs_type_enum, use_hit_distance, return_normals,
            ut_params, ftheta_coeffs, lidar_coeffs, external_distortion_params
        );

        // --- Normalize optional outputs for autograd ----------------------
        // torch::autograd::Function forward returns Tensor or variable_list,
        // not optional<Tensor>. Zero-length tensors preserve the output count
        // for autograd; the dispatcher adapter converts them back to nullopt
        // for the public schema when the caller did not request them. Backward
        // always saves the exact last_ids above even when the public caller did
        // not request it as an output.
        at::Tensor last_ids_output = return_last_ids
            ? fwd.last_ids
            : at::empty({0}, means.options().dtype(at::kInt));
        at::Tensor sample_counts_output = optional_outputs.sample_counts.value_or(
            at::empty({0}, means.options().dtype(at::kInt)));
        at::Tensor normals_output = optional_outputs.normals.value_or(
            at::empty({0}, means.options().dtype(at::kFloat)));

        if (!return_normals) {
            ctx->mark_non_differentiable({normals_output});
        }

        torch::autograd::variable_list out(FwdOutput::COUNT);
        out[FwdOutput::RENDERS]       = fwd.renders;
        out[FwdOutput::ALPHAS]        = fwd.alphas;
        out[FwdOutput::LAST_IDS]      = last_ids_output;
        out[FwdOutput::SAMPLE_COUNTS] = sample_counts_output;
        out[FwdOutput::NORMALS]       = normals_output;
        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        RasterizeFromWorldGrad grad{
            .renders = as_optional_tensor(grad_outputs[FwdOutput::RENDERS]),
            .alphas  = as_optional_tensor(grad_outputs[FwdOutput::ALPHAS]),
            .normals = as_optional_tensor(grad_outputs[FwdOutput::NORMALS]),
        };
        RasterizeBwdResult g =
            apply_bwd<&rasterize_to_pixels_from_world_3dgs_bwd>(ctx, grad);

        torch::autograd::variable_list grads(FwdInput::COUNT);
        grads[FwdInput::MEANS]       = g.v_means;
        grads[FwdInput::QUATS]       = g.v_quats;
        grads[FwdInput::SCALES]      = g.v_scales;
        grads[FwdInput::COLORS]      = g.v_colors;
        grads[FwdInput::OPACITIES]   = g.v_opacities;
        grads[FwdInput::BACKGROUNDS] = g.v_backgrounds;
        grads[FwdInput::RAYS]        = as_tensor(g.v_rays);
        return grads;
    }

    // Forwards to apply() and packs the variable_list into the typed result,
    // restoring optional outputs to nullopt unless their return flag is set.
    static RasterizeToPixelsFromWorld3DGSResult call(
        const at::Tensor &means,
        const at::Tensor &quats,
        const at::Tensor &scales,
        const at::Tensor &colors,
        const at::Tensor &opacities,
        const at::optional<at::Tensor> &backgrounds,
        const at::optional<at::Tensor> &masks,
        int64_t image_width, int64_t image_height, int64_t tile_size,
        const at::Tensor &viewmats0,
        const at::optional<at::Tensor> &viewmats1,
        const at::Tensor &Ks,
        int64_t camera_model,
        const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
        int64_t rs_type,
        const at::optional<at::Tensor> &rays,
        const at::optional<at::Tensor> &radial_coeffs,
        const at::optional<at::Tensor> &tangential_coeffs,
        const at::optional<at::Tensor> &thin_prism_coeffs,
        const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
        const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
        const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
        const at::Tensor &tile_offsets,
        const at::Tensor &flatten_ids,
        bool return_sample_counts,
        bool use_hit_distance,
        bool return_normals,
        int64_t renderer_config,
        bool return_last_ids,
        bool unsafe_masked_tile_outputs
    ) {
        torch::autograd::variable_list outputs = apply(
            means, quats, scales, colors, opacities,
            backgrounds, masks,
            image_width, image_height, tile_size,
            viewmats0, viewmats1, Ks,
            camera_model, ut_params, rs_type,
            rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            ftheta_coeffs, lidar_coeffs, external_distortion_params,
            tile_offsets, flatten_ids,
            return_sample_counts, use_hit_distance, return_normals,
            renderer_config, return_last_ids, unsafe_masked_tile_outputs
        );
        return {
            .renders = outputs[FwdOutput::RENDERS],
            .alphas = outputs[FwdOutput::ALPHAS],
            .last_ids = return_last_ids
                ? at::optional<at::Tensor>(outputs[FwdOutput::LAST_IDS])
                : c10::nullopt,
            .sample_counts = return_sample_counts
                ? at::optional<at::Tensor>(outputs[FwdOutput::SAMPLE_COUNTS])
                : c10::nullopt,
            .normals = return_normals
                ? at::optional<at::Tensor>(outputs[FwdOutput::NORMALS])
                : c10::nullopt,
        };
    }
};

} // namespace
#endif // GSPLAT_BUILD_3DGUT

// Single dispatcher entry for rasterize_to_pixels_from_world_3dgs, registered
// for both the CUDA and AutogradCUDA keys. needs_custom_autograd() mirrors
// apply()'s grad-attachment criterion: under no-grad / inference (or when no
// tensor input requires grad) it runs the state-light forward directly;
// otherwise it routes through the C++ custom autograd Function. Defined in
// Rasterization.cpp; bound in ext.cpp.
RasterizeToPixelsFromWorld3DGSResult
rasterize_to_pixels_from_world_3dgs(
    // Gaussian parameters
    const at::Tensor &means,     // [..., N, 3]
    const at::Tensor &quats,     // [..., N, 4]
    const at::Tensor &scales,    // [..., N, 3]
    const at::Tensor &colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor &opacities, // [..., C, N] or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> &masks,       // [..., C, tile_height, tile_width]
    // image size
    int64_t image_width, int64_t image_height, int64_t tile_size,
    // camera
    const at::Tensor &viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> &viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t camera_model,
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &rays,              // [..., C, H, W, 6]
    const at::optional<at::Tensor> &radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> &tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> &thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor &tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    bool return_sample_counts,
    bool use_hit_distance,
    bool return_normals,
    int64_t renderer_config,
    bool return_last_ids,
    bool unsafe_masked_tile_outputs
) {
#if !GSPLAT_BUILD_3DGUT
    TORCH_CHECK(
        false,
        "rasterize_to_pixels_from_world_3dgs requires GSPLAT_BUILD_3DGUT=1"
    );
    return {};
#else
    const bool use_custom_autograd = needs_custom_autograd(
        means, quats, scales, colors, opacities,
        backgrounds, masks,
        viewmats0, viewmats1, Ks,
        rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        tile_offsets, flatten_ids
    );

    if (!use_custom_autograd) {
        const RendererConfig parsed_renderer_config =
            parse_renderer_config(renderer_config);
        auto optional_outputs = allocate_eval3d_optional_outputs(
            means, viewmats0, image_width, image_height,
            return_sample_counts, return_normals
        );
        // Only the pure-render path can take the state-light fwd-only shortcut:
        // requesting last_ids or sample_counts forces the full forward (the
        // shared forward TORCH_CHECKs that combination).
        const bool fwd_only = !(return_last_ids || return_sample_counts);
        auto fwd = rasterize_to_pixels_from_world_3dgs_fwd(
            means, quats, scales, colors, opacities,
            backgrounds, masks,
            static_cast<uint32_t>(image_width),
            static_cast<uint32_t>(image_height),
            static_cast<uint32_t>(tile_size),
            viewmats0, viewmats1, Ks,
            static_cast<CameraModelType>(camera_model), ut_params,
            static_cast<ShutterType>(rs_type),
            rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            ftheta_coeffs, lidar_coeffs, external_distortion_params,
            tile_offsets, flatten_ids,
            use_hit_distance, parsed_renderer_config,
            /*fwd_only=*/fwd_only, return_last_ids,
            optional_outputs.sample_counts, optional_outputs.normals,
            unsafe_masked_tile_outputs
        );
        return {
            .renders = fwd.renders,
            .alphas = fwd.alphas,
            .last_ids = return_last_ids
                ? at::optional<at::Tensor>(fwd.last_ids)
                : c10::nullopt,
            .sample_counts = optional_outputs.sample_counts,
            .normals = optional_outputs.normals,
        };
    }

    return RasterizeToPixelsFromWorld3DGSAutograd::call(
        means, quats, scales, colors, opacities,
        backgrounds, masks,
        image_width, image_height, tile_size,
        viewmats0, viewmats1, Ks,
        camera_model, ut_params, rs_type,
        rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        ftheta_coeffs, lidar_coeffs, external_distortion_params,
        tile_offsets, flatten_ids,
        return_sample_counts, use_hit_distance, return_normals,
        renderer_config, return_last_ids, unsafe_masked_tile_outputs
    );
#endif // !GSPLAT_BUILD_3DGUT
}

void register_rasterization_cuda_impl(torch::Library &m) {
#if GSPLAT_BUILD_3DGS
    m.impl("rasterize_to_pixels_3dgs", to_torch_op<&rasterize_to_pixels_3dgs_fwd>);
    m.impl("rasterize_to_indices_3dgs", to_torch_op<&rasterize_to_indices_3dgs>);
    m.impl("rasterize_num_contributing_gaussians", &rasterize_num_contributing_gaussians);
    m.impl("rasterize_contributing_gaussian_ids", &rasterize_contributing_gaussian_ids);
    m.impl("rasterize_top_contributing_gaussian_ids", &rasterize_top_contributing_gaussian_ids);
#endif

#if GSPLAT_BUILD_2DGS
    m.impl("rasterize_to_pixels_2dgs", to_torch_op<&rasterize_to_pixels_2dgs_fwd>);
    m.impl("rasterize_to_indices_2dgs", to_torch_op<&rasterize_to_indices_2dgs>);
#endif

    m.impl(
        "rasterize_to_pixels_from_world_3dgs",
        to_torch_op<&rasterize_to_pixels_from_world_3dgs>
    );
}

void register_rasterization_autograd_cuda_impl(torch::Library &m) {
#if GSPLAT_BUILD_3DGS
    m.impl("rasterize_to_pixels_3dgs", to_torch_op<&rasterize_to_pixels_3dgs>);
#endif
#if GSPLAT_BUILD_2DGS
    m.impl("rasterize_to_pixels_2dgs", to_torch_op<&rasterize_to_pixels_2dgs>);
#endif
    m.impl(
        "rasterize_to_pixels_from_world_3dgs",
        to_torch_op<&rasterize_to_pixels_from_world_3dgs>
    );
}

} // namespace gsplat
