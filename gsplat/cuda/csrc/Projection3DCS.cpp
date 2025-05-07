#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"     // where all the macros are defined
#include "Ops.h"        // a collection of all gsplat operators
#include "Projection3DCS.h" // where the launch function is declared

namespace gsplat {

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dcs_fused_fwd(
    const at::Tensor convex_points,               // [N, 6, 3]
    const at::Tensor cumsum_of_points_per_convex, // [N]
    const at::Tensor delta,                       // [N]
    const at::Tensor sigma,                       // [N]
    at::Tensor scaling,                     // [N]
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,                    // [C, 4, 4]
    const at::Tensor Ks,                          // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t total_nb_points,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(convex_points);
    CHECK_INPUT(convex_points);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = convex_points.size(0);    // number of 3D convex shapes
    uint32_t C = viewmats.size(0);         // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    at::Tensor radii = at::empty({C, N, 2}, convex_points.options().dtype(at::kInt));
    at::Tensor means2d = at::empty({C, N, 2}, convex_points.options());

    at::Tensor normals = at::empty({C, total_nb_points, 2}, convex_points.options());
    at::Tensor offsets = at::empty({C, total_nb_points}, convex_points.options());
    at::Tensor p_image = at::empty({C, total_nb_points, 2}, convex_points.options());
    at::Tensor hull = at::empty({C, 2*total_nb_points}, convex_points.options().dtype(at::kInt));
    at::Tensor num_points_per_convex_view = at::empty({C, N}, convex_points.options().dtype(at::kInt));
    at::Tensor indices = at::empty({C, total_nb_points}, convex_points.options().dtype(at::kInt));

    at::Tensor depths = at::empty({C, N}, convex_points.options());
    at::Tensor conics = at::empty({C, N, 3}, convex_points.options());
    at::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = at::zeros({C, N}, convex_points.options());
    }

    launch_projection_ewa_3dcs_fused_fwd_kernel(
        // inputs
        convex_points,
        cumsum_of_points_per_convex,
        delta,
        sigma,
        scaling,
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
        normals,
        offsets,
        p_image,
        hull,
        num_points_per_convex_view,
        indices,
        means2d,
        depths,
        conics,
        calc_compensations ? at::optional<at::Tensor>(compensations)
                           : c10::nullopt
    );
    return std::make_tuple(normals, offsets, p_image, hull, num_points_per_convex_view, indices, radii, means2d, depths, conics, compensations);
}

std::tuple<at::Tensor, at::Tensor>
projection_ewa_3dcs_fused_bwd(
    // fwd inputs
    const at::Tensor convex_points,                // [N, 3]
    const at::Tensor cumsum_of_points_per_convex, // [N]
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [C, N, 2]
    const at::Tensor hull,                        // [C, 2*total_nb_points]
    const at::Tensor num_points_per_convex_view,  // [C, N]
    //const at::Tensor normals,                     // [C, total_nb_points, 2]
    const at::Tensor offsets,                     // [C, total_nb_points]
    const at::Tensor p_image,                     // [C, total_nb_points, 2]
    const at::Tensor indices,                     // [C, total_nb_points]
    const at::Tensor conics,                      // [C, N, 3]
    const at::optional<at::Tensor> compensations, // [C, N] optional
    // grad inputs
    const at::Tensor v_normals,                     // [C, total_nb_points, 2]
    const at::Tensor v_offsets,                     // [C, total_nb_points]
    // grad outputs
    const at::Tensor v_means2d,                     // [C, N, 2]
    const at::Tensor v_depths,                      // [C, N]
    const at::Tensor v_conics,                      // [C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [C, N] optional
    const bool viewmats_requires_grad
) {
    DEVICE_GUARD(convex_points);
    CHECK_INPUT(convex_points);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_normals);
    CHECK_INPUT(v_offsets);
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

    at::Tensor v_convex_points = at::zeros_like(convex_points);
    at::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = at::zeros_like(viewmats);
    }

    launch_projection_ewa_3dcs_fused_bwd_kernel(
        // inputs
        convex_points,
        cumsum_of_points_per_convex,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        camera_model,
        radii,
        hull,
        num_points_per_convex_view,
        //normals,
        offsets,
        p_image,
        indices,
        conics,
        compensations,
        v_normals,
        v_offsets,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        viewmats_requires_grad,
        // outputs
        v_convex_points,
        v_viewmats
    );

    return std::make_tuple(v_convex_points, v_viewmats);
}

} // namespace gsplat
