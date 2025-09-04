#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"     // where all the macros are defined
#include "Ops.h"        // a collection of all gsplat operators
#include "Projection.h" // where the launch function is declared
#include "Cameras.h"

namespace gsplat {

std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_fwd(
    const at::Tensor means,  // [..., C, N, 3]
    const at::Tensor covars, // [..., C, N, 3, 3]
    const at::Tensor Ks,     // [..., C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);

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
    return std::make_tuple(means2d, covars2d);
}

std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_bwd(
    const at::Tensor means,  // [..., C, N, 3]
    const at::Tensor covars, // [..., C, N, 3, 3]
    const at::Tensor Ks,     // [..., C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d, // [..., C, N, 2]
    const at::Tensor v_covars2d // [..., C, N, 2, 2]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_covars2d);

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
        v_means2d,
        v_covars2d,
        // outputs
        v_means,
        v_covars
    );
    return std::make_tuple(v_means, v_covars);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_fused_fwd(
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

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
    if (calc_compensations) {
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
        calc_compensations ? at::optional<at::Tensor>(compensations)
                           : c10::nullopt
    );
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [..., C, N, 2]
    const at::Tensor conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> compensations, // [..., C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [..., C, N, 2]
    const at::Tensor v_depths,                      // [..., C, N]
    const at::Tensor v_conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [..., C, N] optional
    const bool viewmats_requires_grad
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_covars, v_quats, v_scales; // optional
    if (covars.has_value()) {
        v_covars = at::zeros_like(covars.value());
    } else {
        v_quats = at::zeros_like(quats.value());
        v_scales = at::zeros_like(scales.value());
    }
    at::Tensor v_viewmats;
    if (viewmats_requires_grad) {
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
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        viewmats_requires_grad,
        // outputs
        v_means,
        v_covars,
        v_quats,
        v_scales,
        v_viewmats
    );

    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_packed_fwd(
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t C = viewmats.size(-3);       // number of cameras
    uint32_t B = means.numel() / (N * 3); // number of batches
    auto opt = means.options();

    uint32_t nrows = B * C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    // first pass
    int32_t nnz;
    at::Tensor block_accum;
    if (B && C && N) {
        at::Tensor block_cnts =
            at::empty({nrows * blocks_per_row}, opt.dtype(at::kInt));
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
            calc_compensations ? at::optional<at::Tensor>(at::empty({1}, opt))
                               : c10::nullopt
        );
        block_accum = at::cumsum(block_cnts, 0, at::kInt);
        nnz = block_accum[-1].item<int32_t>();
    } else {
        nnz = 0;
    }

    // second pass
    at::Tensor indptr = at::empty({B * C + 1}, opt.dtype(at::kInt));
    at::Tensor batch_ids = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor camera_ids = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor gaussian_ids = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor radii = at::empty({nnz, 2}, opt.dtype(at::kInt));
    at::Tensor means2d = at::empty({nnz, 2}, opt);
    at::Tensor depths = at::empty({nnz}, opt);
    at::Tensor conics = at::empty({nnz, 3}, opt);
    at::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = at::zeros({nnz}, opt);
    }

    if (nnz) {
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
            calc_compensations ? at::optional<at::Tensor>(compensations)
                               : c10::nullopt
        );
    } else {
        indptr.fill_(0);
    }

    return std::make_tuple(
        indptr,
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_packed_bwd(
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6]
    const at::optional<at::Tensor> quats,  // [..., N, 4]
    const at::optional<at::Tensor> scales, // [..., N, 3]
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor batch_ids,                     // [nnz]
    const at::Tensor camera_ids,                    // [nnz]
    const at::Tensor gaussian_ids,                  // [nnz]
    const at::Tensor conics,                        // [nnz, 3]
    const at::optional<at::Tensor> compensations,   // [nnz] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [nnz, 2]
    const at::Tensor v_depths,                      // [nnz]
    const at::Tensor v_conics,                      // [nnz, 3]
    const at::optional<at::Tensor> v_compensations, // [nnz] optional
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(batch_ids);
    CHECK_INPUT(camera_ids);
    CHECK_INPUT(gaussian_ids);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    auto opt = means.options();
    uint32_t nnz = batch_ids.size(0);
    at::Tensor v_means, v_covars, v_quats, v_scales, v_viewmats;
    if (sparse_grad) {
        v_means = at::zeros({nnz, 3}, opt);
        if (covars.has_value()) {
            v_covars = at::zeros({nnz, 6}, opt);
        } else {
            v_quats = at::zeros({nnz, 4}, opt);
            v_scales = at::zeros({nnz, 3}, opt);
        }
    } else {
        v_means = at::zeros_like(means);
        if (covars.has_value()) {
            v_covars = at::zeros_like(covars.value(), opt);
        } else {
            v_quats = at::zeros_like(quats.value(), opt);
            v_scales = at::zeros_like(scales.value(), opt);
        }
    }
    if (viewmats_requires_grad) {
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
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        sparse_grad,
        // outputs
        v_means,
        v_covars.defined() ? at::optional<at::Tensor>(v_covars) : c10::nullopt,
        v_quats.defined() ? at::optional<at::Tensor>(v_quats) : c10::nullopt,
        v_scales.defined() ? at::optional<at::Tensor>(v_scales) : c10::nullopt,
        v_viewmats.defined() ? at::optional<at::Tensor>(v_viewmats)
                             : c10::nullopt
    );
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

// std::tuple<
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor>
// projection_2dgs_fused_fwd(
//     const at::Tensor means,    // [..., N, 3]
//     const at::Tensor quats,    // [..., N, 4]
//     const at::Tensor scales,   // [..., N, 3]
//     const at::Tensor viewmats, // [..., C, 4, 4]
//     const at::Tensor Ks,       // [..., C, 3, 3]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const float eps2d,
//     const float near_plane,
//     const float far_plane,
//     const float radius_clip
// ) {
//     DEVICE_GUARD(means);
//     CHECK_INPUT(means);
//     CHECK_INPUT(quats);
//     CHECK_INPUT(scales);
//     CHECK_INPUT(viewmats);
//     CHECK_INPUT(Ks);

//     auto opt = means.options();
//     at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
//     uint32_t N = means.size(-2);          // number of gaussians
//     uint32_t C = viewmats.size(-3);       // number of cameras

//     at::DimVector radii_shape(batch_dims);
//     radii_shape.append({C, N, 2});
//     at::Tensor radii = at::empty(radii_shape, opt.dtype(at::kInt));

//     at::DimVector means2d_shape(batch_dims);
//     means2d_shape.append({C, N, 2});
//     at::Tensor means2d = at::empty(means2d_shape, opt);

//     at::DimVector depths_shape(batch_dims);
//     depths_shape.append({C, N});
//     at::Tensor depths = at::empty(depths_shape, opt);

//     at::DimVector ray_transforms_shape(batch_dims);
//     ray_transforms_shape.append({C, N, 3, 3});
//     at::Tensor ray_transforms = at::empty(ray_transforms_shape, opt);

//     at::DimVector normals_shape(batch_dims);
//     normals_shape.append({C, N, 3});
//     at::Tensor normals = at::zeros(normals_shape, opt);

//     launch_projection_2dgs_fused_fwd_kernel(
//         // inputs
//         means,
//         quats,
//         scales,
//         viewmats,
//         Ks,
//         image_width,
//         image_height,
//         near_plane,
//         far_plane,
//         radius_clip,
//         // outputs
//         radii,
//         means2d,
//         depths,
//         ray_transforms,
//         normals
//     );
//     return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
// }

// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
// projection_2dgs_fused_bwd(
//     // fwd inputs
//     const at::Tensor means,    // [..., N, 3]
//     const at::Tensor quats,    // [..., N, 4]
//     const at::Tensor scales,   // [..., N, 3]
//     const at::Tensor viewmats, // [..., C, 4, 4]
//     const at::Tensor Ks,       // [..., C, 3, 3]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     // fwd outputs
//     const at::Tensor radii,          // [..., C, N, 2]
//     const at::Tensor ray_transforms, // [..., C, N, 3, 3]
//     // grad outputs
//     const at::Tensor v_means2d,        // [..., C, N, 2]
//     const at::Tensor v_depths,         // [..., C, N]
//     const at::Tensor v_normals,        // [..., C, N, 3]
//     const at::Tensor v_ray_transforms, // [..., C, N, 3, 3]
//     const bool viewmats_requires_grad
// ) {
//     DEVICE_GUARD(means);
//     CHECK_INPUT(means);
//     CHECK_INPUT(quats);
//     CHECK_INPUT(scales);
//     CHECK_INPUT(viewmats);
//     CHECK_INPUT(Ks);
//     CHECK_INPUT(radii);
//     CHECK_INPUT(ray_transforms);
//     CHECK_INPUT(v_means2d);
//     CHECK_INPUT(v_depths);
//     CHECK_INPUT(v_normals);
//     CHECK_INPUT(v_ray_transforms);

//     at::Tensor v_means = at::zeros_like(means);
//     at::Tensor v_quats = at::zeros_like(quats);
//     at::Tensor v_scales = at::zeros_like(scales);
//     at::Tensor v_viewmats;
//     if (viewmats_requires_grad) {
//         v_viewmats = at::zeros_like(viewmats);
//     }

//     launch_projection_2dgs_fused_bwd_kernel(
//         // inputs
//         means,
//         quats,
//         scales,
//         viewmats,
//         Ks,
//         image_width,
//         image_height,
//         radii,
//         ray_transforms,
//         v_means2d,
//         v_depths,
//         v_normals,
//         v_ray_transforms,
//         viewmats_requires_grad,
//         // outputs
//         v_means,
//         v_quats,
//         v_scales,
//         v_viewmats
//     );

//     return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
// }

// std::tuple<
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor>
// projection_2dgs_packed_fwd(
//     const at::Tensor means,    // [..., N, 3]
//     const at::Tensor quats,    // [..., N, 4]
//     const at::Tensor scales,   // [..., N, 3]
//     const at::Tensor viewmats, // [..., C, 4, 4]
//     const at::Tensor Ks,       // [..., C, 3, 3]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const float near_plane,
//     const float far_plane,
//     const float radius_clip
// ) {
//     DEVICE_GUARD(means);
//     CHECK_INPUT(means);
//     CHECK_INPUT(quats);
//     CHECK_INPUT(scales);
//     CHECK_INPUT(viewmats);
//     CHECK_INPUT(Ks);

//     uint32_t N = means.size(-2);          // number of gaussians
//     uint32_t B = means.numel() / (N * 3); // number of batches
//     uint32_t C = viewmats.size(-3);       // number of cameras
//     auto opt = means.options();

//     uint32_t nrows = B * C;
//     uint32_t ncols = N;
//     uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

//     // first pass
//     int32_t nnz;
//     at::Tensor block_accum;
//     if (B && C && N) {
//         at::Tensor block_cnts =
//             at::empty({nrows * blocks_per_row}, opt.dtype(at::kInt));
//         launch_projection_2dgs_packed_fwd_kernel(
//             // inputs
//             means,
//             quats,
//             scales,
//             viewmats,
//             Ks,
//             image_width,
//             image_height,
//             near_plane,
//             far_plane,
//             radius_clip,
//             c10::nullopt, // block_accum
//             // outputs
//             block_cnts,
//             c10::nullopt, // indptr
//             c10::nullopt, // batch_ids
//             c10::nullopt, // camera_ids
//             c10::nullopt, // gaussian_ids
//             c10::nullopt, // radii
//             c10::nullopt, // means2d
//             c10::nullopt, // depths
//             c10::nullopt, // ray_transforms
//             c10::nullopt  // normals
//         );
//         block_accum = at::cumsum(block_cnts, 0, at::kInt);
//         nnz = block_accum[-1].item<int32_t>();
//     } else {
//         nnz = 0;
//     }

//     // second pass
//     at::Tensor indptr = at::empty({B * C + 1}, opt.dtype(at::kInt));
//     at::Tensor batch_ids = at::empty({nnz}, opt.dtype(at::kLong));
//     at::Tensor camera_ids = at::empty({nnz}, opt.dtype(at::kLong));
//     at::Tensor gaussian_ids = at::empty({nnz}, opt.dtype(at::kLong));
//     at::Tensor radii = at::empty({nnz, 2}, opt.dtype(at::kInt));
//     at::Tensor means2d = at::empty({nnz, 2}, opt);
//     at::Tensor depths = at::empty({nnz}, opt);
//     at::Tensor ray_transforms = at::empty({nnz, 3, 3}, opt);
//     at::Tensor normals = at::empty({nnz, 3}, opt);

//     if (nnz) {
//         launch_projection_2dgs_packed_fwd_kernel(
//             // inputs
//             means,
//             quats,
//             scales,
//             viewmats,
//             Ks,
//             image_width,
//             image_height,
//             near_plane,
//             far_plane,
//             radius_clip,
//             block_accum,
//             // outputs
//             c10::nullopt, // block_cnts
//             indptr,
//             batch_ids,
//             camera_ids,
//             gaussian_ids,
//             radii,
//             means2d,
//             depths,
//             ray_transforms,
//             normals
//         );
//     } else {
//         indptr.fill_(0);
//     }

//     return std::make_tuple(
//         indptr,
//         batch_ids,
//         camera_ids,
//         gaussian_ids,
//         radii,
//         means2d,
//         depths,
//         ray_transforms,
//         normals
//     );
// }

// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
// projection_2dgs_packed_bwd(
//     // fwd inputs
//     const at::Tensor means,    // [..., N, 3]
//     const at::Tensor quats,    // [..., N, 4]
//     const at::Tensor scales,   // [..., N, 3]
//     const at::Tensor viewmats, // [..., C, 4, 4]
//     const at::Tensor Ks,       // [..., C, 3, 3]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     // fwd outputs
//     const at::Tensor batch_ids,      // [nnz]
//     const at::Tensor camera_ids,     // [nnz]
//     const at::Tensor gaussian_ids,   // [nnz]
//     const at::Tensor ray_transforms, // [nnz, 3, 3]
//     // grad outputs
//     const at::Tensor v_means2d,        // [nnz, 2]
//     const at::Tensor v_depths,         // [nnz]
//     const at::Tensor v_ray_transforms, // [nnz, 3, 3]
//     const at::Tensor v_normals,        // [nnz, 3]
//     const bool viewmats_requires_grad,
//     const bool sparse_grad
// ) {
//     DEVICE_GUARD(means);
//     CHECK_INPUT(means);
//     CHECK_INPUT(quats);
//     CHECK_INPUT(scales);
//     CHECK_INPUT(viewmats);
//     CHECK_INPUT(Ks);
//     CHECK_INPUT(batch_ids);
//     CHECK_INPUT(camera_ids);
//     CHECK_INPUT(gaussian_ids);
//     CHECK_INPUT(ray_transforms);
//     CHECK_INPUT(v_means2d);
//     CHECK_INPUT(v_depths);
//     CHECK_INPUT(v_normals);
//     CHECK_INPUT(v_ray_transforms);

//     auto opt = means.options();
//     uint32_t N = means.size(-2);          // number of gaussians
//     uint32_t B = means.numel() / (N * 3); // number of batches
//     uint32_t C = viewmats.size(-3);       // number of cameras
//     uint32_t nnz = batch_ids.size(0);

//     at::Tensor v_means, v_quats, v_scales, v_viewmats;
//     if (sparse_grad) {
//         v_means = at::zeros({nnz, 3}, opt);
//         v_quats = at::zeros({nnz, 4}, opt);
//         v_scales = at::zeros({nnz, 3}, opt);
//     } else {
//         v_means = at::zeros_like(means, opt);
//         v_quats = at::zeros_like(quats, opt);
//         v_scales = at::zeros_like(scales, opt);
//     }
//     if (viewmats_requires_grad) {
//         v_viewmats = at::zeros_like(viewmats, opt);
//     }
    
//     launch_projection_2dgs_packed_bwd_kernel(
//         // fwd inputs
//         means,
//         quats,
//         scales,
//         viewmats,
//         Ks,
//         image_width,
//         image_height,
//         // fwd outputs
//         batch_ids,
//         camera_ids,
//         gaussian_ids,
//         ray_transforms,
//         // grad outputs
//         v_means2d,
//         v_depths,
//         v_ray_transforms,
//         v_normals,
//         sparse_grad,
//         // outputs
//         v_means,
//         v_quats,
//         v_scales,
//         v_viewmats.defined() ? at::optional<at::Tensor>(v_viewmats)
//                              : c10::nullopt
//     );
//     return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
// }

// std::tuple<
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor,
//     at::Tensor>
// projection_ut_3dgs_fused(
//     const at::Tensor means,                   // [..., N, 3]
//     const at::Tensor quats,                   // [..., N, 4]
//     const at::Tensor scales,                  // [..., N, 3]
//     const at::optional<at::Tensor> opacities, // [..., N] optional
//     const at::Tensor viewmats0,               // [..., C, 4, 4]
//     const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
//     const at::Tensor Ks,                      // [..., C, 3, 3]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const float eps2d,
//     const float near_plane,
//     const float far_plane,
//     const float radius_clip,
//     const bool calc_compensations,
//     const CameraModelType camera_model,
//     // uncented transform
//     const UnscentedTransformParameters ut_params,
//     ShutterType rs_type,
//     const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
//     const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
//     const at::optional<at::Tensor> thin_prism_coeffs,  // [..., C, 4] optional
//     const FThetaCameraDistortionParameters ftheta_coeffs // shared parameters for all cameras
// ) {
//     DEVICE_GUARD(means);
//     CHECK_INPUT(means);
//     CHECK_INPUT(quats);
//     CHECK_INPUT(scales);
//     if (opacities.has_value()) {
//         CHECK_INPUT(opacities.value());
//     }
//     CHECK_INPUT(viewmats0);
//     if (viewmats1.has_value()) {
//         CHECK_INPUT(viewmats1.value());
//     }
//     CHECK_INPUT(Ks);
//     if (radial_coeffs.has_value()) {
//         CHECK_INPUT(radial_coeffs.value());
//     }
//     if (tangential_coeffs.has_value()) {
//         CHECK_INPUT(tangential_coeffs.value());
//     }
//     if (thin_prism_coeffs.has_value()) {
//         CHECK_INPUT(thin_prism_coeffs.value());
//     }

//     at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
//     uint32_t N = means.size(-2);    // number of gaussians
//     uint32_t C = Ks.size(-3);       // number of cameras
//     auto opt = means.options();

//     at::DimVector radii_shape(batch_dims);
//     radii_shape.append({C, N, 2});
//     at::Tensor radii = at::empty(radii_shape, opt.dtype(at::kInt));

//     at::DimVector means2d_shape(batch_dims);
//     means2d_shape.append({C, N, 2});
//     at::Tensor means2d = at::empty(means2d_shape, opt);

//     at::DimVector depths_shape(batch_dims);
//     depths_shape.append({C, N});
//     at::Tensor depths = at::empty(depths_shape, opt);
    
//     at::DimVector conics_shape(batch_dims);
//     conics_shape.append({C, N, 3});
//     at::Tensor conics = at::empty(conics_shape, opt);

//     at::Tensor compensations;
//     if (calc_compensations) {
//         // we dont want NaN to appear in this tensor, so we zero intialize it
//         at::DimVector compensations_shape(batch_dims);
//         compensations_shape.append({C, N});
//         compensations = at::zeros(compensations_shape, opt);
//     }

//     launch_projection_ut_3dgs_fused_kernel(
//         // inputs
//         means,
//         quats,
//         scales,
//         opacities,
//         viewmats0,
//         viewmats1,
//         Ks,
//         image_width,
//         image_height,
//         eps2d,
//         near_plane,
//         far_plane,
//         radius_clip,
//         camera_model,
//         // uncented transform
//         ut_params,
//         rs_type,
//         radial_coeffs,
//         tangential_coeffs,
//         thin_prism_coeffs,
//         ftheta_coeffs,
//         // outputs
//         radii,
//         means2d,
//         depths,
//         conics,
//         calc_compensations ? at::optional<at::Tensor>(compensations)
//                            : at::nullopt
//     );
//     return std::make_tuple(radii, means2d, depths, conics, compensations);
// }

} // namespace gsplat
