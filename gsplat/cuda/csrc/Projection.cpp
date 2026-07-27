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
#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "Common.h"     // where all the macros are defined
#include "Projection.h" // where the launch function is declared
#include "TorchUtils.h"
#include "Cameras.h"
#include "Lidars.h"
#include "Lidars.cuh"
#include "Config.h"

namespace gsplat
{
#if GSPLAT_BUILD_3DGS

namespace
{
    void check_projection_ewa_simple_inputs(
        const at::Tensor &means, const at::Tensor &covars, const at::Tensor &Ks, CameraModelType camera_model
    )
    {
        TORCH_CHECK(
            camera_model != CameraModelType::FTHETA,
            "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"
        );

        TORCH_CHECK(means.dim() >= 3, "means must have shape [..., C, N, 3], got ", means.sizes());
        const int64_t batch_ndim = means.dim() - 3;
        const int64_t C          = means.size(batch_ndim);
        const int64_t N          = means.size(batch_ndim + 1);
        at::DimVector batch_shape(means.sizes().slice(0, batch_ndim));

        TORCH_CHECK(means.size(batch_ndim + 2) == 3, "means must have shape [..., C, N, 3], got ", means.sizes());

        at::DimVector covars_shape(batch_shape);
        covars_shape.append({C, N, 3, 3});
        TORCH_CHECK(covars.sizes() == covars_shape, "covars must have shape [..., C, N, 3, 3], got ", covars.sizes());

        at::DimVector Ks_shape(batch_shape);
        Ks_shape.append({C, 3, 3});
        TORCH_CHECK(Ks.sizes() == Ks_shape, "Ks must have shape [..., C, 3, 3], got ", Ks.sizes());

        CHECK_INPUT(means);
        CHECK_INPUT(covars);
        CHECK_INPUT(Ks);
    }
} // namespace

struct ProjectionEWASimpleFwdResult
{
    at::Tensor means2d;
    at::Tensor covars2d;
};

template<>
struct TorchArgDef<ProjectionEWASimpleFwdResult>
{
    static auto to(const ProjectionEWASimpleFwdResult &r)
    {
        return to_torch_args(r.means2d, r.covars2d);
    }
};

ProjectionEWASimpleFwdResult projection_ewa_simple_fwd(
    const at::Tensor &means,  // [..., C, N, 3]
    const at::Tensor &covars, // [..., C, N, 3, 3]
    const at::Tensor &Ks,     // [..., C, 3, 3]
    int64_t width,
    int64_t height,
    CameraModelType camera_model
)
{
    check_projection_ewa_simple_inputs(means, covars, Ks, camera_model);

    DEVICE_GUARD(means);

    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 3));
    uint32_t C = means.size(-3);
    uint32_t N = means.size(-2);

    at::DimVector means2d_shape(batch_dims);
    means2d_shape.append({C, N, 2});
    at::Tensor means2d = at::empty(means2d_shape, opt);

    at::DimVector covars2d_shape(batch_dims);
    covars2d_shape.append({C, N, 2, 2});
    at::Tensor covars2d = at::empty(covars2d_shape, opt);

    launch_projection_ewa_simple_fwd_kernel(
        // inputs
        means,
        covars,
        Ks,
        width,
        height,
        camera_model,
        // outputs
        means2d,
        covars2d
    );
    return ProjectionEWASimpleFwdResult{
        .means2d  = means2d,
        .covars2d = covars2d,
    };
}

template<>
struct TorchArgDef<ProjectionEWA3DGSFusedFwdResult>
{
    static auto to(const ProjectionEWA3DGSFusedFwdResult &r)
    {
        return to_torch_args(r.radii, r.means2d, r.depths, r.conics, r.compensations);
    }

    template<class TT>
    static ProjectionEWA3DGSFusedFwdResult from(TT &&t)
    {
        return {
            .radii         = std::get<0>(t),
            .means2d       = std::get<1>(t),
            .depths        = std::get<2>(t),
            .conics        = std::get<3>(t),
            .compensations = std::get<4>(t),
        };
    }
};

template<>
struct TorchArgDef<ProjectionEWA3DGSPackedFwdResult>
{
    static auto to(const ProjectionEWA3DGSPackedFwdResult &r)
    {
        return to_torch_args(
            r.batch_ids, r.camera_ids, r.gaussian_ids, r.indptr, r.radii, r.means2d, r.depths, r.conics, r.compensations
        );
    }

    template<class TT>
    static ProjectionEWA3DGSPackedFwdResult from(TT &&t)
    {
        return {
            .batch_ids     = std::get<0>(t),
            .camera_ids    = std::get<1>(t),
            .gaussian_ids  = std::get<2>(t),
            .indptr        = std::get<3>(t),
            .radii         = std::get<4>(t),
            .means2d       = std::get<5>(t),
            .depths        = std::get<6>(t),
            .conics        = std::get<7>(t),
            .compensations = std::get<8>(t),
        };
    }
};

struct ProjectionEWASimpleBwdResult
{
    at::Tensor v_means;
    at::Tensor v_covars;
};

template<>
struct TorchArgDef<ProjectionEWASimpleBwdResult>
{
    static auto to(const ProjectionEWASimpleBwdResult &r)
    {
        return to_torch_args(r.v_means, r.v_covars);
    }
};

// Gradients of the differentiable forward outputs.
struct ProjectionEWASimpleGrad
{
    static constexpr bool is_grad_bundle = true;
    at::Tensor means2d;  // [..., C, N, 2]
    at::Tensor covars2d; // [..., C, N, 2, 2]
};

template<>
struct TorchArgDef<ProjectionEWASimpleGrad>
{
    static auto to(const ProjectionEWASimpleGrad &g)
    {
        return to_torch_args(g.means2d, g.covars2d);
    }

    template<class TT>
    static ProjectionEWASimpleGrad from(TT &&t)
    {
        return {
            .means2d  = std::get<0>(t),
            .covars2d = std::get<1>(t),
        };
    }
};

// Full backward for projection_ewa_simple.
ProjectionEWASimpleBwdResult projection_ewa_simple_bwd(
    const at::Tensor &means,  // [..., C, N, 3]
    const at::Tensor &covars, // [..., C, N, 3, 3]
    const at::Tensor &Ks,     // [..., C, 3, 3]
    int64_t width,
    int64_t height,
    CameraModelType camera_model,
    const ProjectionEWASimpleGrad &grad
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);
    CHECK_DENSE(grad.means2d);
    CHECK_INPUT(grad.means2d);
    CHECK_DENSE(grad.covars2d);
    CHECK_INPUT(grad.covars2d);

    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 3));
    uint32_t C = means.size(-3);
    uint32_t N = means.size(-2);

    at::DimVector v_means_shape(batch_dims);
    v_means_shape.append({C, N, 3});
    at::Tensor v_means = at::empty(v_means_shape, opt);

    at::DimVector v_covars_shape(batch_dims);
    v_covars_shape.append({C, N, 3, 3});
    at::Tensor v_covars = at::empty(v_covars_shape, opt);

    launch_projection_ewa_simple_bwd_kernel(
        // inputs
        means,
        covars,
        Ks,
        width,
        height,
        camera_model,
        grad.means2d,
        grad.covars2d,
        // outputs
        v_means,
        v_covars
    );
    return {
        .v_means  = v_means,
        .v_covars = v_covars,
    };
}

namespace
{
    void check_projection_ewa_3dgs_fused_inputs(
        const at::Tensor &means,
        const at::optional<at::Tensor> &covars,
        const at::optional<at::Tensor> &quats,
        const at::optional<at::Tensor> &scales,
        const at::optional<at::Tensor> &opacities,
        const at::Tensor &viewmats,
        const at::Tensor &Ks,
        CameraModelType camera_model
    )
    {
        TORCH_CHECK(
            camera_model != CameraModelType::FTHETA,
            "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"
        );

        TORCH_CHECK(means.dim() >= 2, "means must have shape [..., N, 3], got ", means.sizes());
        const int64_t batch_ndim = means.dim() - 2;
        const int64_t N          = means.size(batch_ndim);
        at::DimVector batch_shape(means.sizes().slice(0, batch_ndim));
        TORCH_CHECK(means.size(batch_ndim + 1) == 3, "means must have shape [..., N, 3], got ", means.sizes());
        TORCH_CHECK(
            viewmats.dim() == batch_ndim + 3, "viewmats must have shape [..., C, 4, 4], got ", viewmats.sizes()
        );
        TORCH_CHECK(Ks.dim() == batch_ndim + 3, "Ks must have shape [..., C, 3, 3], got ", Ks.sizes());

        const int64_t C = viewmats.size(batch_ndim);
        at::DimVector viewmats_shape(batch_shape);
        viewmats_shape.append({C, 4, 4});
        TORCH_CHECK(
            viewmats.sizes() == viewmats_shape, "viewmats must have shape [..., C, 4, 4], got ", viewmats.sizes()
        );

        at::DimVector Ks_shape(batch_shape);
        Ks_shape.append({C, 3, 3});
        TORCH_CHECK(Ks.sizes() == Ks_shape, "Ks must have shape [..., C, 3, 3], got ", Ks.sizes());

        CHECK_INPUT(means);
        CHECK_INPUT(viewmats);
        CHECK_INPUT(Ks);

        if(covars.has_value())
        {
            const at::Tensor &covars_tensor = covars.value();
            at::DimVector covars_shape(batch_shape);
            covars_shape.append({N, 6});
            TORCH_CHECK(
                covars_tensor.sizes() == covars_shape, "covars must have shape [..., N, 6], got ", covars_tensor.sizes()
            );
            CHECK_INPUT(covars_tensor);
        }
        else
        {
            TORCH_CHECK(quats.has_value(), "covars or quats is required");
            TORCH_CHECK(scales.has_value(), "covars or scales is required");

            const at::Tensor &quats_tensor  = quats.value();
            const at::Tensor &scales_tensor = scales.value();
            at::DimVector quats_shape(batch_shape);
            quats_shape.append({N, 4});
            TORCH_CHECK(
                quats_tensor.sizes() == quats_shape, "quats must have shape [..., N, 4], got ", quats_tensor.sizes()
            );

            at::DimVector scales_shape(batch_shape);
            scales_shape.append({N, 3});
            TORCH_CHECK(
                scales_tensor.sizes() == scales_shape, "scales must have shape [..., N, 3], got ", scales_tensor.sizes()
            );
            CHECK_INPUT(quats_tensor);
            CHECK_INPUT(scales_tensor);
        }

        if(opacities.has_value())
        {
            const at::Tensor &opacities_tensor = opacities.value();
            at::DimVector opacities_shape(batch_shape);
            opacities_shape.append({N});
            TORCH_CHECK(
                opacities_tensor.sizes() == opacities_shape,
                "opacities must have shape [..., N], got ",
                opacities_tensor.sizes()
            );
            CHECK_INPUT(opacities_tensor);
        }
    }
} // namespace

ProjectionEWA3DGSFusedFwdResult projection_ewa_3dgs_fused_fwd(
    const at::Tensor &means,                   // [..., N, 3]
    const at::optional<at::Tensor> &covars,    // [..., N, 6] optional
    const at::optional<at::Tensor> &quats,     // [..., N, 4] optional
    const at::optional<at::Tensor> &scales,    // [..., N, 3] optional
    const at::optional<at::Tensor> &opacities, // [..., N] optional
    const at::Tensor &viewmats,                // [..., C, 4, 4]
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    CameraModelType camera_model
)
{
    check_projection_ewa_3dgs_fused_inputs(means, covars, quats, scales, opacities, viewmats, Ks, camera_model);

    DEVICE_GUARD(means);

    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras

    at::DimVector radii_shape(batch_dims);
    radii_shape.append({C, N, 2});
    at::Tensor radii = at::empty(radii_shape, opt.dtype(at::kInt));
    at::DimVector means2d_shape(batch_dims);
    means2d_shape.append({C, N, 2});
    at::Tensor means2d = at::empty(means2d_shape, opt);
    at::DimVector depths_shape(batch_dims);
    depths_shape.append({C, N});
    at::Tensor depths = at::empty(depths_shape, opt);
    at::DimVector conics_shape(batch_dims);
    conics_shape.append({C, N, 3});
    at::Tensor conics = at::empty(conics_shape, opt);
    at::Tensor compensations;
    if(calc_compensations)
    {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        at::DimVector compensations_shape(batch_dims);
        compensations_shape.append({C, N});
        compensations = at::zeros(compensations_shape, opt);
    }

    launch_projection_ewa_3dgs_fused_fwd_kernel(
        // inputs
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        camera_model,
        // outputs
        radii,
        means2d,
        depths,
        conics,
        calc_compensations ? at::optional<at::Tensor>(compensations) : c10::nullopt
    );
    return ProjectionEWA3DGSFusedFwdResult{
        .radii         = radii,
        .means2d       = means2d,
        .depths        = depths,
        .conics        = conics,
        .compensations = calc_compensations ? at::optional<at::Tensor>(compensations) : c10::nullopt
    };
}

struct ProjectionEWA3DGSFusedBwdResult
{
    at::Tensor v_means;
    at::optional<at::Tensor> v_covars;
    at::optional<at::Tensor> v_quats;
    at::optional<at::Tensor> v_scales;
    at::optional<at::Tensor> v_viewmats;
};

template<>
struct TorchArgDef<ProjectionEWA3DGSFusedBwdResult>
{
    static auto to(const ProjectionEWA3DGSFusedBwdResult &r)
    {
        return to_torch_args(r.v_means, r.v_covars, r.v_quats, r.v_scales, r.v_viewmats);
    }
};

// Gradients of the differentiable forward outputs.
struct ProjectionEWA3DGSFusedGrad
{
    static constexpr bool is_grad_bundle = true;
    at::Tensor means2d;                     // [..., C, N, 2]
    at::Tensor depths;                      // [..., C, N]
    at::Tensor conics;                      // [..., C, N, 3]
    at::optional<at::Tensor> compensations; // [..., C, N]
};

template<>
struct TorchArgDef<ProjectionEWA3DGSFusedGrad>
{
    static auto to(const ProjectionEWA3DGSFusedGrad &g)
    {
        return to_torch_args(g.means2d, g.depths, g.conics, g.compensations);
    }

    template<class TT>
    static ProjectionEWA3DGSFusedGrad from(TT &&t)
    {
        return {
            .means2d       = std::get<0>(t),
            .depths        = std::get<1>(t),
            .conics        = std::get<2>(t),
            .compensations = std::get<3>(t),
        };
    }
};

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::optional<at::Tensor>
>
    projection_ewa_3dgs_fused_fwd_privateuseone(
        const at::Tensor &means,                   // [..., N, 3]
        const at::optional<at::Tensor> &covars,    // [..., N, 6] optional
        const at::optional<at::Tensor> &quats,     // [..., N, 4] optional
        const at::optional<at::Tensor> &scales,    // [..., N, 3] optional
        const at::optional<at::Tensor> &opacities, // [..., N] optional
        const at::Tensor &viewmats,                // [..., C, 4, 4]
        const at::Tensor &Ks,                      // [..., C, 3, 3]
        int64_t image_width,
        int64_t image_height,
        double eps2d,
        double near_plane,
        double far_plane,
        double radius_clip,
        bool calc_compensations,
        int64_t camera_model
    )
{
    check_projection_ewa_3dgs_fused_inputs(
        means, covars, quats, scales, opacities, viewmats, Ks, static_cast<CameraModelType>(camera_model)
    );

    DEVICE_GUARD(means);
    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    uint32_t N = means.size(-2);
    uint32_t C = viewmats.size(-3);

    at::DimVector radii_shape(batch_dims);
    radii_shape.append({C, N, 2});
    at::Tensor radii = at::empty(radii_shape, opt.dtype(at::kInt));
    at::DimVector means2d_shape(batch_dims);
    means2d_shape.append({C, N, 2});
    at::Tensor means2d = at::empty(means2d_shape, opt);
    at::DimVector depths_shape(batch_dims);
    depths_shape.append({C, N});
    at::Tensor depths = at::empty(depths_shape, opt);
    at::DimVector conics_shape(batch_dims);
    conics_shape.append({C, N, 3});
    at::Tensor conics = at::empty(conics_shape, opt);
    at::Tensor compensations;
    if(calc_compensations)
    {
        at::DimVector compensations_shape(batch_dims);
        compensations_shape.append({C, N});
        compensations = at::zeros(compensations_shape, opt);
    }

    launch_projection_ewa_3dgs_fused_fwd_kernels(
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        static_cast<CameraModelType>(camera_model),
        radii,
        means2d,
        depths,
        conics,
        calc_compensations ? at::optional<at::Tensor>(compensations) : c10::nullopt
    );
    return std::make_tuple(
        radii, means2d, depths, conics, calc_compensations ? at::optional<at::Tensor>(compensations) : c10::nullopt
    );
}

// Full backward for projection_ewa_3dgs_fused.
// `viewmats_requires_grad` selects whether to compute the viewmats gradient. It
// is an explicit argument rather than something inferred from a tensor's
// requires_grad, so this functional op's behavior follows only from its inputs.
ProjectionEWA3DGSFusedBwdResult projection_ewa_3dgs_fused_bwd(
    // fwd inputs
    const at::Tensor &means,                // [..., N, 3]
    const at::optional<at::Tensor> &covars, // [..., N, 6] optional
    const at::optional<at::Tensor> &quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> &scales, // [..., N, 3] optional
    const at::Tensor &viewmats,             // [..., C, 4, 4]
    const at::Tensor &Ks,                   // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    CameraModelType camera_model,
    // fwd outputs
    const at::Tensor &radii,                       // [..., C, N, 2]
    const at::Tensor &conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> &compensations, // [..., C, N] optional
    const ProjectionEWA3DGSFusedGrad &grad,
    bool viewmats_requires_grad
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if(covars.has_value())
    {
        CHECK_INPUT(covars.value());
    }
    else
    {
        TORCH_INTERNAL_ASSERT(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_DENSE(grad.means2d);
    CHECK_DENSE(grad.depths);
    CHECK_DENSE(grad.conics);
    at::Tensor grad_means2d = grad.means2d.contiguous();
    at::Tensor grad_depths  = grad.depths.contiguous();
    at::Tensor grad_conics  = grad.conics.contiguous();
    CHECK_INPUT(grad_means2d);
    CHECK_INPUT(grad_depths);
    CHECK_INPUT(grad_conics);
    if(compensations.has_value())
    {
        CHECK_INPUT(compensations.value());
    }
    // A compensation gradient is meaningful only when compensations were
    // computed; under materialize_grads a compensations output that was not
    // requested can still receive a zero gradient, which is ignored here.
    at::optional<at::Tensor> compensation_grad;
    at::Tensor grad_compensations;
    if(compensations.has_value() && grad.compensations.has_value())
    {
        CHECK_DENSE(grad.compensations.value());
        grad_compensations = grad.compensations.value().contiguous();
        CHECK_INPUT(grad_compensations);
        compensation_grad = grad_compensations;
    }

    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_covars, v_quats, v_scales; // optional
    if(covars.has_value())
    {
        v_covars = at::zeros_like(covars.value());
    }
    else
    {
        v_quats  = at::zeros_like(quats.value());
        v_scales = at::zeros_like(scales.value());
    }
    at::Tensor v_viewmats;
    if(viewmats_requires_grad)
    {
        v_viewmats = at::zeros_like(viewmats);
    }

    launch_projection_ewa_3dgs_fused_bwd_kernel(
        // inputs
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        camera_model,
        radii,
        conics,
        compensations,
        grad_means2d,
        grad_depths,
        grad_conics,
        compensation_grad,
        viewmats_requires_grad,
        // outputs
        v_means,
        v_covars,
        v_quats,
        v_scales,
        v_viewmats
    );

    return {
        .v_means    = v_means,
        .v_covars   = as_optional_tensor(v_covars),
        .v_quats    = as_optional_tensor(v_quats),
        .v_scales   = as_optional_tensor(v_scales),
        .v_viewmats = as_optional_tensor(v_viewmats),
    };
}

ProjectionEWA3DGSFusedResult projection_ewa_3dgs_fused(
    const at::Tensor &means,
    const at::optional<at::Tensor> &covars,
    const at::optional<at::Tensor> &quats,
    const at::optional<at::Tensor> &scales,
    const at::optional<at::Tensor> &opacities,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    CameraModelType camera_model
)
{
    // Dispatch through the registered op so the Python-side register_autograd
    // kernel makes it differentiable. Args are passed in the forward op's
    // parameter order.
    return call_torch_op<&projection_ewa_3dgs_fused_fwd>(
        "gsplat::projection_ewa_3dgs_fused",
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        camera_model
    );
}

std::tuple<
    at::Tensor,
    at::optional<at::Tensor>,
    at::optional<at::Tensor>,
    at::optional<at::Tensor>,
    at::optional<at::Tensor>
>
    projection_ewa_3dgs_fused_bwd_privateuseone(
        // fwd inputs
        const at::Tensor &means,                // [..., N, 3]
        const at::optional<at::Tensor> &covars, // [..., N, 6] optional
        const at::optional<at::Tensor> &quats,  // [..., N, 4] optional
        const at::optional<at::Tensor> &scales, // [..., N, 3] optional
        const at::Tensor &viewmats,             // [..., C, 4, 4]
        const at::Tensor &Ks,                   // [..., C, 3, 3]
        int64_t image_width,
        int64_t image_height,
        double eps2d,
        int64_t camera_model,
        // fwd outputs
        const at::Tensor &radii,                       // [..., C, N, 2]
        const at::Tensor &conics,                      // [..., C, N, 3]
        const at::optional<at::Tensor> &compensations, // [..., C, N] optional
        // grad outputs
        const at::Tensor &v_means2d,                     // [..., C, N, 2]
        const at::Tensor &v_depths,                      // [..., C, N]
        const at::Tensor &v_conics,                      // [..., C, N, 3]
        const at::optional<at::Tensor> &v_compensations, // [..., C, N] optional
        bool viewmats_requires_grad
    )
{
    DEVICE_GUARD(means);
    check_projection_ewa_3dgs_fused_inputs(
        means, covars, quats, scales, c10::nullopt, viewmats, Ks, static_cast<CameraModelType>(camera_model)
    );
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if(compensations.has_value())
    {
        CHECK_INPUT(compensations.value());
    }
    at::optional<at::Tensor> compensation_grad;
    if(compensations.has_value() && v_compensations.has_value())
    {
        CHECK_INPUT(v_compensations.value());
        compensation_grad = v_compensations;
    }

    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_covars, v_quats, v_scales;
    if(covars.has_value())
    {
        v_covars = at::zeros_like(covars.value());
    }
    else
    {
        v_quats  = at::zeros_like(quats.value());
        v_scales = at::zeros_like(scales.value());
    }
    at::Tensor v_viewmats;
    if(viewmats_requires_grad)
    {
        v_viewmats = at::zeros_like(viewmats);
    }

    launch_projection_ewa_3dgs_fused_bwd_kernels(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        static_cast<CameraModelType>(camera_model),
        radii,
        conics,
        compensations,
        v_means2d,
        v_depths,
        v_conics,
        compensation_grad,
        viewmats_requires_grad,
        v_means,
        v_covars,
        v_quats,
        v_scales,
        v_viewmats
    );

    return std::make_tuple(
        v_means,
        as_optional_tensor(v_covars),
        as_optional_tensor(v_quats),
        as_optional_tensor(v_scales),
        as_optional_tensor(v_viewmats)
    );
}

namespace
{
    void check_projection_ewa_3dgs_packed_inputs(
        const at::Tensor &means,
        const at::optional<at::Tensor> &covars,
        const at::optional<at::Tensor> &quats,
        const at::optional<at::Tensor> &scales,
        const at::optional<at::Tensor> &opacities,
        const at::Tensor &viewmats,
        const at::Tensor &Ks,
        bool sparse_grad,
        CameraModelType camera_model
    )
    {
        check_projection_ewa_3dgs_fused_inputs(means, covars, quats, scales, opacities, viewmats, Ks, camera_model);
        TORCH_CHECK(!sparse_grad || means.dim() == 2, "sparse_grad does not support batch dimensions");
    }
} // namespace

ProjectionEWA3DGSPackedFwdResult projection_ewa_3dgs_packed_fwd(
    const at::Tensor &means,                   // [..., N, 3]
    const at::optional<at::Tensor> &covars,    // [..., N, 6] optional
    const at::optional<at::Tensor> &quats,     // [..., N, 4] optional
    const at::optional<at::Tensor> &scales,    // [..., N, 3] optional
    const at::optional<at::Tensor> &opacities, // [..., N] optional
    const at::Tensor &viewmats,                // [..., C, 4, 4]
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool sparse_grad,
    bool calc_compensations,
    CameraModelType camera_model
)
{
    check_projection_ewa_3dgs_packed_inputs(
        means, covars, quats, scales, opacities, viewmats, Ks, sparse_grad, camera_model
    );

    DEVICE_GUARD(means);

    uint32_t N = means.size(-2);                                                  // number of gaussians
    uint32_t C = viewmats.size(-3);                                               // number of cameras
    uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2)); // number of batches
    auto opt   = means.options();

    uint32_t nrows          = B * C;
    uint32_t ncols          = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    // first pass
    int32_t nnz;
    at::Tensor block_accum;
    if(B && C && N)
    {
        at::Tensor block_cnts = at::empty({nrows * blocks_per_row}, opt.dtype(at::kInt));
        launch_projection_ewa_3dgs_packed_fwd_kernel(
            // inputs
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            c10::nullopt, // block_accum
            camera_model,
            // outputs
            block_cnts,
            c10::nullopt, // indptr
            c10::nullopt, // batch_ids
            c10::nullopt, // camera_ids
            c10::nullopt, // gaussian_ids
            c10::nullopt, // radii
            c10::nullopt, // means2d
            c10::nullopt, // depths
            c10::nullopt, // conics
            // pass in as an indicator on whether compensation will be applied or not.
            calc_compensations ? at::optional<at::Tensor>(at::empty({1}, opt)) : c10::nullopt
        );
        block_accum = at::cumsum(block_cnts, 0, at::kInt);
        nnz         = block_accum[-1].item<int32_t>();
    }
    else
    {
        nnz = 0;
    }

    // second pass
    at::Tensor indptr       = at::empty({B * C + 1}, opt.dtype(at::kInt));
    at::Tensor batch_ids    = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor camera_ids   = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor gaussian_ids = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor radii        = at::empty({nnz, 2}, opt.dtype(at::kInt));
    at::Tensor means2d      = at::empty({nnz, 2}, opt);
    at::Tensor depths       = at::empty({nnz}, opt);
    at::Tensor conics       = at::empty({nnz, 3}, opt);
    at::Tensor compensations;
    if(calc_compensations)
    {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = at::zeros({nnz}, opt);
    }

    if(nnz)
    {
        launch_projection_ewa_3dgs_packed_fwd_kernel(
            // inputs
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            block_accum,
            camera_model,
            // outputs
            c10::nullopt, // block_cnts
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            calc_compensations ? at::optional<at::Tensor>(compensations) : c10::nullopt
        );
    }
    else
    {
        indptr.fill_(0);
    }

    return ProjectionEWA3DGSPackedFwdResult{
        .batch_ids     = batch_ids,
        .camera_ids    = camera_ids,
        .gaussian_ids  = gaussian_ids,
        .indptr        = indptr,
        .radii         = radii,
        .means2d       = means2d,
        .depths        = depths,
        .conics        = conics,
        .compensations = calc_compensations ? at::optional<at::Tensor>(compensations) : c10::nullopt
    };
}

struct ProjectionEWA3DGSPackedBwdResult
{
    at::Tensor v_means;
    at::optional<at::Tensor> v_covars;
    at::optional<at::Tensor> v_quats;
    at::optional<at::Tensor> v_scales;
    at::optional<at::Tensor> v_viewmats;
};

template<>
struct TorchArgDef<ProjectionEWA3DGSPackedBwdResult>
{
    static auto to(const ProjectionEWA3DGSPackedBwdResult &r)
    {
        return to_torch_args(r.v_means, r.v_covars, r.v_quats, r.v_scales, r.v_viewmats);
    }
};

// Gradients of the differentiable forward outputs.
struct ProjectionEWA3DGSPackedGrad
{
    static constexpr bool is_grad_bundle = true;
    at::Tensor means2d;                     // [nnz, 2]
    at::Tensor depths;                      // [nnz]
    at::Tensor conics;                      // [nnz, 3]
    at::optional<at::Tensor> compensations; // [nnz]
};

template<>
struct TorchArgDef<ProjectionEWA3DGSPackedGrad>
{
    static auto to(const ProjectionEWA3DGSPackedGrad &g)
    {
        return to_torch_args(g.means2d, g.depths, g.conics, g.compensations);
    }

    template<class TT>
    static ProjectionEWA3DGSPackedGrad from(TT &&t)
    {
        return {
            .means2d       = std::get<0>(t),
            .depths        = std::get<1>(t),
            .conics        = std::get<2>(t),
            .compensations = std::get<3>(t),
        };
    }
};

// Full backward for projection_ewa_3dgs_packed.
// `viewmats_requires_grad` selects whether to compute the viewmats gradient. It
// is an explicit argument rather than something inferred from a tensor's
// requires_grad, so this functional op's behavior follows only from its inputs.
ProjectionEWA3DGSPackedBwdResult projection_ewa_3dgs_packed_bwd(
    // fwd inputs
    const at::Tensor &means,                // [..., N, 3]
    const at::optional<at::Tensor> &covars, // [..., N, 6]
    const at::optional<at::Tensor> &quats,  // [..., N, 4]
    const at::optional<at::Tensor> &scales, // [..., N, 3]
    const at::Tensor &viewmats,             // [..., C, 4, 4]
    const at::Tensor &Ks,                   // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    CameraModelType camera_model,
    bool sparse_grad,
    // fwd outputs
    const at::Tensor &batch_ids,                   // [nnz]
    const at::Tensor &camera_ids,                  // [nnz]
    const at::Tensor &gaussian_ids,                // [nnz]
    const at::Tensor &conics,                      // [nnz, 3]
    const at::optional<at::Tensor> &compensations, // [nnz] optional
    // grad outputs
    const ProjectionEWA3DGSPackedGrad &grad,
    bool viewmats_requires_grad
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if(covars.has_value())
    {
        CHECK_INPUT(covars.value());
    }
    else
    {
        TORCH_INTERNAL_ASSERT(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(batch_ids);
    CHECK_INPUT(camera_ids);
    CHECK_INPUT(gaussian_ids);
    CHECK_INPUT(conics);
    CHECK_DENSE(grad.means2d);
    CHECK_DENSE(grad.depths);
    CHECK_DENSE(grad.conics);
    at::Tensor grad_means2d = grad.means2d.contiguous();
    at::Tensor grad_depths  = grad.depths.contiguous();
    at::Tensor grad_conics  = grad.conics.contiguous();
    CHECK_INPUT(grad_means2d);
    CHECK_INPUT(grad_depths);
    CHECK_INPUT(grad_conics);
    if(compensations.has_value())
    {
        CHECK_INPUT(compensations.value());
    }
    // A compensation gradient is meaningful only when compensations were
    // computed; under materialize_grads a compensations output that was not
    // requested can still receive a zero gradient, which is ignored here.
    at::optional<at::Tensor> compensation_grad;
    at::Tensor grad_compensations;
    if(compensations.has_value() && grad.compensations.has_value())
    {
        CHECK_DENSE(grad.compensations.value());
        grad_compensations = grad.compensations.value().contiguous();
        CHECK_INPUT(grad_compensations);
        compensation_grad = grad_compensations;
    }

    auto opt     = means.options();
    uint32_t nnz = batch_ids.size(0);
    at::Tensor v_means, v_covars, v_quats, v_scales, v_viewmats;
    if(sparse_grad)
    {
        v_means = at::zeros({nnz, 3}, opt);
        if(covars.has_value())
        {
            v_covars = at::zeros({nnz, 6}, opt);
        }
        else
        {
            v_quats  = at::zeros({nnz, 4}, opt);
            v_scales = at::zeros({nnz, 3}, opt);
        }
    }
    else
    {
        v_means = at::zeros_like(means);
        if(covars.has_value())
        {
            v_covars = at::zeros_like(covars.value(), opt);
        }
        else
        {
            v_quats  = at::zeros_like(quats.value(), opt);
            v_scales = at::zeros_like(scales.value(), opt);
        }
    }
    if(viewmats_requires_grad)
    {
        v_viewmats = at::zeros_like(viewmats, opt);
    }

    launch_projection_ewa_3dgs_packed_bwd_kernel(
        // fwd inputs
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        camera_model,
        // fwd outputs
        batch_ids,
        camera_ids,
        gaussian_ids,
        conics,
        compensations,
        // grad outputs
        grad_means2d,
        grad_depths,
        grad_conics,
        compensation_grad,
        sparse_grad,
        // outputs
        v_means,
        v_covars.defined() ? at::optional<at::Tensor>(v_covars) : c10::nullopt,
        v_quats.defined() ? at::optional<at::Tensor>(v_quats) : c10::nullopt,
        v_scales.defined() ? at::optional<at::Tensor>(v_scales) : c10::nullopt,
        v_viewmats.defined() ? at::optional<at::Tensor>(v_viewmats) : c10::nullopt
    );

    // When sparse_grad is set, the kernel writes per-nnz dense gradients indexed
    // by gaussian_ids; wrap them as sparse COO over the full per-input shapes so
    // the optimizer can scatter-update only the touched gaussians. Coalesced
    // when there is a single batch (each gaussian_id appears once).
    if(sparse_grad)
    {
        const bool is_coalesced              = viewmats.size(0) == 1;
        const at::Tensor sparse_grad_indices = gaussian_ids.unsqueeze(0);
        v_means = make_sparse_coo_grad(sparse_grad_indices, v_means, means.sizes(), is_coalesced);
        if(covars.has_value())
        {
            v_covars = make_sparse_coo_grad(sparse_grad_indices, v_covars, covars.value().sizes(), is_coalesced);
        }
        else
        {
            v_quats  = make_sparse_coo_grad(sparse_grad_indices, v_quats, quats.value().sizes(), is_coalesced);
            v_scales = make_sparse_coo_grad(sparse_grad_indices, v_scales, scales.value().sizes(), is_coalesced);
        }
    }

    return ProjectionEWA3DGSPackedBwdResult{
        .v_means    = v_means,
        .v_covars   = as_optional_tensor(v_covars),
        .v_quats    = as_optional_tensor(v_quats),
        .v_scales   = as_optional_tensor(v_scales),
        .v_viewmats = as_optional_tensor(v_viewmats),
    };
}

ProjectionEWA3DGSPackedResult projection_ewa_3dgs_packed(
    const at::Tensor &means,
    const at::optional<at::Tensor> &covars,
    const at::optional<at::Tensor> &quats,
    const at::optional<at::Tensor> &scales,
    const at::optional<at::Tensor> &opacities,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool sparse_grad,
    bool calc_compensations,
    CameraModelType camera_model
)
{
    // Invoke the op through the dispatcher so its registered autograd is
    // recorded and the result is differentiable.
    return call_torch_op<&projection_ewa_3dgs_packed_fwd>(
        "gsplat::projection_ewa_3dgs_packed",
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        sparse_grad,
        calc_compensations,
        camera_model
    );
}

#endif

#if GSPLAT_BUILD_2DGS

namespace
{
    void check_projection_2dgs_inputs(
        const at::Tensor &means,
        const at::Tensor &quats,
        const at::Tensor &scales,
        const at::Tensor &viewmats,
        const at::Tensor &Ks
    )
    {
        TORCH_CHECK(means.dim() >= 2, "means must have shape [..., N, 3], got ", means.sizes());
        const int64_t batch_ndim = means.dim() - 2;
        const int64_t N          = means.size(batch_ndim);
        at::DimVector batch_shape(means.sizes().slice(0, batch_ndim));

        TORCH_CHECK(means.size(batch_ndim + 1) == 3, "means must have shape [..., N, 3], got ", means.sizes());
        TORCH_CHECK(quats.dim() == batch_ndim + 2, "quats must have shape [..., N, 4], got ", quats.sizes());
        TORCH_CHECK(scales.dim() == batch_ndim + 2, "scales must have shape [..., N, 3], got ", scales.sizes());
        TORCH_CHECK(
            viewmats.dim() == batch_ndim + 3, "viewmats must have shape [..., C, 4, 4], got ", viewmats.sizes()
        );
        TORCH_CHECK(Ks.dim() == batch_ndim + 3, "Ks must have shape [..., C, 3, 3], got ", Ks.sizes());

        const int64_t C = viewmats.size(batch_ndim);
        at::DimVector quats_shape(batch_shape);
        quats_shape.append({N, 4});
        TORCH_CHECK(quats.sizes() == quats_shape, "quats must have shape [..., N, 4], got ", quats.sizes());

        at::DimVector scales_shape(batch_shape);
        scales_shape.append({N, 3});
        TORCH_CHECK(scales.sizes() == scales_shape, "scales must have shape [..., N, 3], got ", scales.sizes());

        at::DimVector viewmats_shape(batch_shape);
        viewmats_shape.append({C, 4, 4});
        TORCH_CHECK(
            viewmats.sizes() == viewmats_shape, "viewmats must have shape [..., C, 4, 4], got ", viewmats.sizes()
        );

        at::DimVector Ks_shape(batch_shape);
        Ks_shape.append({C, 3, 3});
        TORCH_CHECK(Ks.sizes() == Ks_shape, "Ks must have shape [..., C, 3, 3], got ", Ks.sizes());

        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(scales);
        CHECK_INPUT(viewmats);
        CHECK_INPUT(Ks);
    }
} // namespace

template<>
struct TorchArgDef<Projection2DGSFusedResult>
{
    static auto to(const Projection2DGSFusedResult &r)
    {
        return to_torch_args(r.radii, r.means2d, r.depths, r.ray_transforms, r.normals);
    }

    template<class TT>
    static Projection2DGSFusedResult from(TT &&t)
    {
        return {
            .radii          = std::get<0>(t),
            .means2d        = std::get<1>(t),
            .depths         = std::get<2>(t),
            .ray_transforms = std::get<3>(t),
            .normals        = std::get<4>(t),
        };
    }
};

using Projection2DGSFusedFwdResult = Projection2DGSFusedResult;

Projection2DGSFusedFwdResult projection_2dgs_fused_fwd(
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip
)
{
    check_projection_2dgs_inputs(means, quats, scales, viewmats, Ks);

    DEVICE_GUARD(means);

    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras

    at::DimVector radii_shape(batch_dims);
    radii_shape.append({C, N, 2});
    at::Tensor radii = at::empty(radii_shape, opt.dtype(at::kInt));

    at::DimVector means2d_shape(batch_dims);
    means2d_shape.append({C, N, 2});
    at::Tensor means2d = at::empty(means2d_shape, opt);

    at::DimVector depths_shape(batch_dims);
    depths_shape.append({C, N});
    at::Tensor depths = at::empty(depths_shape, opt);

    at::DimVector ray_transforms_shape(batch_dims);
    ray_transforms_shape.append({C, N, 3, 3});
    at::Tensor ray_transforms = at::empty(ray_transforms_shape, opt);

    at::DimVector normals_shape(batch_dims);
    normals_shape.append({C, N, 3});
    at::Tensor normals = at::zeros(normals_shape, opt);

    launch_projection_2dgs_fused_fwd_kernel(
        // inputs
        means,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        near_plane,
        far_plane,
        radius_clip,
        // outputs
        radii,
        means2d,
        depths,
        ray_transforms,
        normals
    );
    return Projection2DGSFusedFwdResult{
        .radii          = radii,
        .means2d        = means2d,
        .depths         = depths,
        .ray_transforms = ray_transforms,
        .normals        = normals,
    };
}

struct Projection2DGSFusedBwdResult
{
    at::Tensor v_means;
    at::Tensor v_quats;
    at::Tensor v_scales;
    at::optional<at::Tensor> v_viewmats;
};

template<>
struct TorchArgDef<Projection2DGSFusedBwdResult>
{
    static auto to(const Projection2DGSFusedBwdResult &r)
    {
        return to_torch_args(r.v_means, r.v_quats, r.v_scales, r.v_viewmats);
    }
};

// Gradients of the differentiable forward outputs.
struct Projection2DGSFusedGrad
{
    static constexpr bool is_grad_bundle = true;
    at::Tensor means2d;        // [..., C, N, 2]
    at::Tensor depths;         // [..., C, N]
    at::Tensor ray_transforms; // [..., C, N, 3, 3]
    at::Tensor normals;        // [..., C, N, 3]
};

template<>
struct TorchArgDef<Projection2DGSFusedGrad>
{
    static auto to(const Projection2DGSFusedGrad &g)
    {
        return to_torch_args(g.means2d, g.depths, g.ray_transforms, g.normals);
    }

    template<class TT>
    static Projection2DGSFusedGrad from(TT &&t)
    {
        return {
            .means2d        = std::get<0>(t),
            .depths         = std::get<1>(t),
            .ray_transforms = std::get<2>(t),
            .normals        = std::get<3>(t),
        };
    }
};

// Full backward for projection_2dgs_fused.
// `viewmats_requires_grad` selects whether to compute the viewmats gradient. It
// is an explicit argument rather than something inferred from a tensor's
// requires_grad, so this functional op's behavior follows only from its inputs.
Projection2DGSFusedBwdResult projection_2dgs_fused_bwd(
    // fwd inputs
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    // fwd outputs
    const at::Tensor &radii,          // [..., C, N, 2]
    const at::Tensor &ray_transforms, // [..., C, N, 3, 3]
    const Projection2DGSFusedGrad &grad,
    bool viewmats_requires_grad
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(ray_transforms);
    CHECK_DENSE(grad.means2d);
    CHECK_INPUT(grad.means2d);
    CHECK_DENSE(grad.depths);
    CHECK_INPUT(grad.depths);
    CHECK_DENSE(grad.normals);
    CHECK_INPUT(grad.normals);
    CHECK_DENSE(grad.ray_transforms);
    CHECK_INPUT(grad.ray_transforms);

    at::Tensor v_means  = at::zeros_like(means);
    at::Tensor v_quats  = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);
    at::Tensor v_viewmats;
    if(viewmats_requires_grad)
    {
        v_viewmats = at::zeros_like(viewmats);
    }

    launch_projection_2dgs_fused_bwd_kernel(
        // inputs
        means,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        radii,
        ray_transforms,
        grad.means2d,
        grad.depths,
        grad.normals,
        grad.ray_transforms,
        viewmats_requires_grad,
        // outputs
        v_means,
        v_quats,
        v_scales,
        v_viewmats
    );

    return Projection2DGSFusedBwdResult{
        .v_means    = v_means,
        .v_quats    = v_quats,
        .v_scales   = v_scales,
        .v_viewmats = as_optional_tensor(v_viewmats),
    };
}

Projection2DGSFusedFwdResult projection_2dgs_fused(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip
)
{
    // Invoke the op through the dispatcher so its registered autograd is
    // recorded and the result is differentiable.
    return call_torch_op<&projection_2dgs_fused_fwd>(
        "gsplat::projection_2dgs_fused",
        means,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip
    );
}

template<>
struct TorchArgDef<Projection2DGSPackedResult>
{
    static auto to(const Projection2DGSPackedResult &r)
    {
        return to_torch_args(
            r.batch_ids,
            r.camera_ids,
            r.gaussian_ids,
            r.indptr,
            r.radii,
            r.means2d,
            r.depths,
            r.ray_transforms,
            r.normals
        );
    }

    template<class TT>
    static Projection2DGSPackedResult from(TT &&t)
    {
        return {
            .batch_ids      = std::get<0>(t),
            .camera_ids     = std::get<1>(t),
            .gaussian_ids   = std::get<2>(t),
            .indptr         = std::get<3>(t),
            .radii          = std::get<4>(t),
            .means2d        = std::get<5>(t),
            .depths         = std::get<6>(t),
            .ray_transforms = std::get<7>(t),
            .normals        = std::get<8>(t),
        };
    }
};

using Projection2DGSPackedFwdResult = Projection2DGSPackedResult;

Projection2DGSPackedFwdResult projection_2dgs_packed_fwd(
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool sparse_grad
)
{
    DEVICE_GUARD(means);
    check_projection_2dgs_inputs(means, quats, scales, viewmats, Ks);

    uint32_t N = means.size(-2);                                                  // number of gaussians
    uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2)); // number of batches
    uint32_t C = viewmats.size(-3);                                               // number of cameras
    auto opt   = means.options();

    uint32_t nrows          = B * C;
    uint32_t ncols          = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    // first pass
    int32_t nnz;
    at::Tensor block_accum;
    if(B && C && N)
    {
        at::Tensor block_cnts = at::empty({nrows * blocks_per_row}, opt.dtype(at::kInt));
        launch_projection_2dgs_packed_fwd_kernel(
            // inputs
            means,
            quats,
            scales,
            viewmats,
            Ks,
            image_width,
            image_height,
            near_plane,
            far_plane,
            radius_clip,
            c10::nullopt, // block_accum
            // outputs
            block_cnts,
            c10::nullopt, // indptr
            c10::nullopt, // batch_ids
            c10::nullopt, // camera_ids
            c10::nullopt, // gaussian_ids
            c10::nullopt, // radii
            c10::nullopt, // means2d
            c10::nullopt, // depths
            c10::nullopt, // ray_transforms
            c10::nullopt  // normals
        );
        block_accum = at::cumsum(block_cnts, 0, at::kInt);
        nnz         = block_accum[-1].item<int32_t>();
    }
    else
    {
        nnz = 0;
    }

    // second pass
    at::Tensor indptr         = at::empty({B * C + 1}, opt.dtype(at::kInt));
    at::Tensor batch_ids      = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor camera_ids     = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor gaussian_ids   = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor radii          = at::empty({nnz, 2}, opt.dtype(at::kInt));
    at::Tensor means2d        = at::empty({nnz, 2}, opt);
    at::Tensor depths         = at::empty({nnz}, opt);
    at::Tensor ray_transforms = at::empty({nnz, 3, 3}, opt);
    at::Tensor normals        = at::empty({nnz, 3}, opt);

    if(nnz)
    {
        launch_projection_2dgs_packed_fwd_kernel(
            // inputs
            means,
            quats,
            scales,
            viewmats,
            Ks,
            image_width,
            image_height,
            near_plane,
            far_plane,
            radius_clip,
            block_accum,
            // outputs
            c10::nullopt, // block_cnts
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals
        );
    }
    else
    {
        indptr.fill_(0);
    }

    return Projection2DGSPackedFwdResult{
        .batch_ids      = batch_ids,
        .camera_ids     = camera_ids,
        .gaussian_ids   = gaussian_ids,
        .indptr         = indptr,
        .radii          = radii,
        .means2d        = means2d,
        .depths         = depths,
        .ray_transforms = ray_transforms,
        .normals        = normals,
    };
}

struct Projection2DGSPackedBwdResult
{
    at::Tensor v_means;
    at::Tensor v_quats;
    at::Tensor v_scales;
    at::optional<at::Tensor> v_viewmats;
};

template<>
struct TorchArgDef<Projection2DGSPackedBwdResult>
{
    static auto to(const Projection2DGSPackedBwdResult &r)
    {
        return to_torch_args(r.v_means, r.v_quats, r.v_scales, r.v_viewmats);
    }
};

// Gradients of the differentiable forward outputs.
struct Projection2DGSPackedGrad
{
    static constexpr bool is_grad_bundle = true;
    at::Tensor means2d;        // [nnz, 2]
    at::Tensor depths;         // [nnz]
    at::Tensor ray_transforms; // [nnz, 3, 3]
    at::Tensor normals;        // [nnz, 3]
};

template<>
struct TorchArgDef<Projection2DGSPackedGrad>
{
    static auto to(const Projection2DGSPackedGrad &g)
    {
        return to_torch_args(g.means2d, g.depths, g.ray_transforms, g.normals);
    }

    template<class TT>
    static Projection2DGSPackedGrad from(TT &&t)
    {
        return {
            .means2d        = std::get<0>(t),
            .depths         = std::get<1>(t),
            .ray_transforms = std::get<2>(t),
            .normals        = std::get<3>(t),
        };
    }
};

// Full backward for projection_2dgs_packed.
// `viewmats_requires_grad` selects whether to compute the viewmats gradient. It
// is an explicit argument rather than something inferred from a tensor's
// requires_grad, so this functional op's behavior follows only from its inputs.
Projection2DGSPackedBwdResult projection_2dgs_packed_bwd(
    // fwd inputs
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    bool sparse_grad,
    // fwd outputs
    const at::Tensor &batch_ids,      // [nnz]
    const at::Tensor &camera_ids,     // [nnz]
    const at::Tensor &gaussian_ids,   // [nnz]
    const at::Tensor &ray_transforms, // [nnz, 3, 3]
    // grad outputs
    const Projection2DGSPackedGrad &grad,
    bool viewmats_requires_grad
)
{
    DEVICE_GUARD(means);
    check_projection_2dgs_inputs(means, quats, scales, viewmats, Ks);
    CHECK_DENSE(grad.means2d);
    CHECK_INPUT(grad.means2d);
    CHECK_DENSE(grad.depths);
    CHECK_INPUT(grad.depths);
    CHECK_DENSE(grad.normals);
    CHECK_INPUT(grad.normals);
    CHECK_DENSE(grad.ray_transforms);
    CHECK_INPUT(grad.ray_transforms);

    auto opt     = means.options();
    uint32_t nnz = batch_ids.size(0);

    at::Tensor v_means, v_quats, v_scales, v_viewmats;
    if(sparse_grad)
    {
        v_means  = at::zeros({nnz, 3}, opt);
        v_quats  = at::zeros({nnz, 4}, opt);
        v_scales = at::zeros({nnz, 3}, opt);
    }
    else
    {
        v_means  = at::zeros_like(means, opt);
        v_quats  = at::zeros_like(quats, opt);
        v_scales = at::zeros_like(scales, opt);
    }
    if(viewmats_requires_grad)
    {
        v_viewmats = at::zeros_like(viewmats, opt);
    }

    launch_projection_2dgs_packed_bwd_kernel(
        // fwd inputs
        means,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        // fwd outputs
        batch_ids,
        camera_ids,
        gaussian_ids,
        ray_transforms,
        // grad outputs
        grad.means2d,
        grad.depths,
        grad.ray_transforms,
        grad.normals,
        sparse_grad,
        // outputs
        v_means,
        v_quats,
        v_scales,
        v_viewmats.defined() ? at::optional<at::Tensor>(v_viewmats) : c10::nullopt
    );

    // When sparse_grad is set, the kernel writes per-nnz dense gradients indexed
    // by gaussian_ids; wrap them as sparse COO over the full per-input shapes so
    // the optimizer can scatter-update only the touched gaussians. Coalesced
    // when there is a single batch (each gaussian_id appears once).
    if(sparse_grad)
    {
        const bool is_coalesced              = viewmats.size(0) == 1;
        const at::Tensor sparse_grad_indices = gaussian_ids.unsqueeze(0);
        v_means  = make_sparse_coo_grad(sparse_grad_indices, v_means, means.sizes(), is_coalesced);
        v_quats  = make_sparse_coo_grad(sparse_grad_indices, v_quats, quats.sizes(), is_coalesced);
        v_scales = make_sparse_coo_grad(sparse_grad_indices, v_scales, scales.sizes(), is_coalesced);
    }

    return Projection2DGSPackedBwdResult{
        .v_means    = v_means,
        .v_quats    = v_quats,
        .v_scales   = v_scales,
        .v_viewmats = as_optional_tensor(v_viewmats),
    };
}

Projection2DGSPackedResult projection_2dgs_packed(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool sparse_grad
)
{
    // Invoke the op through the dispatcher so its registered autograd is
    // recorded and the result is differentiable.
    return call_torch_op<&projection_2dgs_packed_fwd>(
        "gsplat::projection_2dgs_packed",
        means,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        near_plane,
        far_plane,
        radius_clip,
        sparse_grad
    );
}

#endif

#if GSPLAT_BUILD_3DGUT

ProjectionUT3DGSFusedResult projection_ut_3dgs_fused_impl(
    const at::Tensor means,                   // [..., N, 3]
    const at::Tensor quats,                   // [..., N, 4]
    const at::Tensor scales,                  // [..., N, 3]
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model,
    const bool global_z_order,
    // uncented transform
    const at::optional<c10::intrusive_ptr<UnscentedTransformParameters>> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const at::optional<c10::intrusive_ptr<FThetaCameraDistortionParameters>>
        &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    if(opacities.has_value())
    {
        CHECK_INPUT(opacities.value());
    }
    CHECK_INPUT(viewmats0);
    if(viewmats1.has_value())
    {
        CHECK_INPUT(viewmats1.value());
    }
    CHECK_INPUT(Ks);
    if(radial_coeffs.has_value())
    {
        CHECK_INPUT(radial_coeffs.value());
    }
    if(tangential_coeffs.has_value())
    {
        CHECK_INPUT(tangential_coeffs.value());
    }
    if(thin_prism_coeffs.has_value())
    {
        CHECK_INPUT(thin_prism_coeffs.value());
    }

    if(external_distortion_params.has_value())
    {
        CHECK_CONTIGUOUS(external_distortion_params.value()->horizontal_poly);
        CHECK_CONTIGUOUS(external_distortion_params.value()->vertical_poly);
    }

    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    uint32_t N = means.size(-2); // number of gaussians
    uint32_t C = Ks.size(-3);    // number of cameras

    // Validate inputs.
    // Contiguity/device/dtype are already covered by the CHECK_INPUT calls above.
    {
        at::DimVector means_shape(batch_dims);
        means_shape.append({N, 3});
        TORCH_CHECK(means.sizes() == means_shape, "means must have shape [..., N, 3], got ", means.sizes());
        at::DimVector quats_shape(batch_dims);
        quats_shape.append({N, 4});
        TORCH_CHECK(quats.sizes() == quats_shape, "quats must have shape [..., N, 4], got ", quats.sizes());
        at::DimVector scales_shape(batch_dims);
        scales_shape.append({N, 3});
        TORCH_CHECK(scales.sizes() == scales_shape, "scales must have shape [..., N, 3], got ", scales.sizes());
        if(opacities.has_value())
        {
            at::DimVector opacities_shape(batch_dims);
            opacities_shape.append({N});
            TORCH_CHECK(
                opacities.value().sizes() == opacities_shape,
                "opacities must have shape [..., N], got ",
                opacities.value().sizes()
            );
        }
        at::DimVector viewmats_shape(batch_dims);
        viewmats_shape.append({C, 4, 4});
        TORCH_CHECK(
            viewmats0.sizes() == viewmats_shape, "viewmats must have shape [..., C, 4, 4], got ", viewmats0.sizes()
        );
        at::DimVector Ks_shape(batch_dims);
        Ks_shape.append({C, 3, 3});
        TORCH_CHECK(Ks.sizes() == Ks_shape, "Ks must have shape [..., C, 3, 3], got ", Ks.sizes());
        if(camera_model == CameraModelType::ORTHO)
        {
            TORCH_CHECK(
                !radial_coeffs.has_value() && !tangential_coeffs.has_value() && !thin_prism_coeffs.has_value(),
                "ortho camera model does not support radial_coeffs, tangential_coeffs, or thin_prism_coeffs parameters"
            );
        }
        if(radial_coeffs.has_value())
        {
            const at::Tensor &radial = radial_coeffs.value();
            at::DimVector radial_prefix(batch_dims);
            radial_prefix.append({C});
            // Guard the prefix slice on rank >= 1 so a sub-1-D radial fails this
            // check cleanly instead of underflowing slice()'s size_t length.
            const int64_t radial_ndim = radial.dim();
            const int64_t radial_last = radial_ndim >= 1 ? radial.size(-1) : -1;
            TORCH_CHECK(
                radial_ndim >= 1
                    && radial.sizes().slice(0, radial_ndim - 1) == radial_prefix
                    && (radial_last == 6 || radial_last == 4),
                "radial_coeffs must have shape [..., C, 6] or [..., C, 4], got ",
                radial.sizes()
            );
        }
        if(tangential_coeffs.has_value())
        {
            at::DimVector tangential_shape(batch_dims);
            tangential_shape.append({C, 2});
            TORCH_CHECK(
                tangential_coeffs.value().sizes() == tangential_shape,
                "tangential_coeffs must have shape [..., C, 2], got ",
                tangential_coeffs.value().sizes()
            );
        }
        if(thin_prism_coeffs.has_value())
        {
            at::DimVector thin_prism_shape(batch_dims);
            thin_prism_shape.append({C, 4});
            TORCH_CHECK(
                thin_prism_coeffs.value().sizes() == thin_prism_shape,
                "thin_prism_coeffs must have shape [..., C, 4], got ",
                thin_prism_coeffs.value().sizes()
            );
        }
        if(viewmats1.has_value())
        {
            at::DimVector viewmats1_shape(batch_dims);
            viewmats1_shape.append({C, 4, 4});
            TORCH_CHECK(
                viewmats1.value().sizes() == viewmats1_shape,
                "viewmats_rs must have shape [..., C, 4, 4], got ",
                viewmats1.value().sizes()
            );
        }
    }

    auto opt = means.options();

    at::DimVector radii_shape(batch_dims);
    radii_shape.append({C, N, 2});
    at::Tensor radii = at::empty(radii_shape, opt.dtype(at::kInt));

    at::DimVector means2d_shape(batch_dims);
    means2d_shape.append({C, N, 2});
    at::Tensor means2d = at::empty(means2d_shape, opt);

    at::DimVector depths_shape(batch_dims);
    depths_shape.append({C, N});
    at::Tensor depths = at::empty(depths_shape, opt);

    at::DimVector conics_shape(batch_dims);
    conics_shape.append({C, N, 3});
    at::Tensor conics = at::empty(conics_shape, opt);

    at::Tensor compensations;
    if(calc_compensations)
    {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        at::DimVector compensations_shape(batch_dims);
        compensations_shape.append({C, N});
        compensations = at::zeros(compensations_shape, opt);
    }

    launch_projection_ut_3dgs_fused_kernel(
        // inputs
        means,
        quats,
        scales,
        opacities,
        viewmats0,
        viewmats1,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        camera_model,
        global_z_order,
        // uncented transform
        ut_params,
        rs_type,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_params,
        // outputs
        radii,
        means2d,
        depths,
        conics,
        calc_compensations ? at::optional<at::Tensor>(compensations) : at::nullopt
    );
    return {
        .radii         = radii,
        .means2d       = means2d,
        .depths        = depths,
        .conics        = conics,
        .compensations = calc_compensations ? at::optional<at::Tensor>(compensations) : at::nullopt
    };
}

#endif // GSPLAT_BUILD_3DGUT

ProjectionUT3DGSFusedResult projection_ut_3dgs_fused(
    const at::Tensor &means,                   // [..., N, 3]
    const at::Tensor &quats,                   // [..., N, 4]
    const at::Tensor &scales,                  // [..., N, 3]
    const at::optional<at::Tensor> &opacities, // [..., N] optional
    const at::Tensor &viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> &viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    CameraModelType camera_model,
    bool global_z_order,
    const at::optional<c10::intrusive_ptr<UnscentedTransformParameters>> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> &radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> &tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> &thin_prism_coeffs, // [..., C, 4] optional
    const at::optional<c10::intrusive_ptr<FThetaCameraDistortionParameters>> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params
)
{
#if !GSPLAT_BUILD_3DGUT
    TORCH_CHECK(false, "projection_ut_3dgs_fused requires GSPLAT_BUILD_3DGUT=1");
    return {};
#else
    return projection_ut_3dgs_fused_impl(
        means,
        quats,
        scales,
        opacities,
        viewmats0,
        viewmats1,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        camera_model,
        global_z_order,
        ut_params,
        rs_type,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_params
    );
#endif // !GSPLAT_BUILD_3DGUT
}

void register_projection_cuda_impl(torch::Library &m)
{
#if GSPLAT_BUILD_3DGS
    m.impl("projection_ewa_simple", to_torch_op<&projection_ewa_simple_fwd>);
    m.impl("projection_ewa_simple_bwd", to_torch_op<&projection_ewa_simple_bwd>);
    m.impl("projection_ewa_3dgs_fused", to_torch_op<&projection_ewa_3dgs_fused_fwd>);
    m.impl("projection_ewa_3dgs_fused_bwd", to_torch_op<&projection_ewa_3dgs_fused_bwd>);
    m.impl("projection_ewa_3dgs_packed", to_torch_op<&projection_ewa_3dgs_packed_fwd>);
    m.impl("projection_ewa_3dgs_packed_bwd", to_torch_op<&projection_ewa_3dgs_packed_bwd>);
#endif

#if GSPLAT_BUILD_2DGS
    m.impl("projection_2dgs_fused", to_torch_op<&projection_2dgs_fused_fwd>);
    m.impl("projection_2dgs_fused_bwd", to_torch_op<&projection_2dgs_fused_bwd>);
    m.impl("projection_2dgs_packed", to_torch_op<&projection_2dgs_packed_fwd>);
    m.impl("projection_2dgs_packed_bwd", to_torch_op<&projection_2dgs_packed_bwd>);
#endif

    m.impl("projection_ut_3dgs_fused", to_torch_op<&projection_ut_3dgs_fused>);
}

void register_projection_privateuseone_impl(torch::Library &m)
{
#if GSPLAT_BUILD_3DGS
    m.impl("projection_ewa_3dgs_fused", to_torch_op<&projection_ewa_3dgs_fused_fwd_privateuseone>);
    m.impl("projection_ewa_3dgs_fused_bwd", to_torch_op<&projection_ewa_3dgs_fused_bwd_privateuseone>);
#endif
}
} // namespace gsplat
