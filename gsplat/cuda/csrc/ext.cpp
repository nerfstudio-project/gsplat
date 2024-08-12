#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_sh_fwd", &gsplat::compute_sh_fwd_tensor);
    m.def("compute_sh_bwd", &gsplat::compute_sh_bwd_tensor);

    m.def(
        "quat_scale_to_covar_preci_fwd",
        &gsplat::quat_scale_to_covar_preci_fwd_tensor
    );
    m.def(
        "quat_scale_to_covar_preci_bwd",
        &gsplat::quat_scale_to_covar_preci_bwd_tensor
    );

    m.def("persp_proj_fwd", &gsplat::persp_proj_fwd_tensor);
    m.def("persp_proj_bwd", &gsplat::persp_proj_bwd_tensor);

    m.def("world_to_cam_fwd", &gsplat::world_to_cam_fwd_tensor);
    m.def("world_to_cam_bwd", &gsplat::world_to_cam_bwd_tensor);

    m.def(
        "fully_fused_projection_fwd", &gsplat::fully_fused_projection_fwd_tensor
    );
    m.def(
        "fully_fused_projection_bwd", &gsplat::fully_fused_projection_bwd_tensor
    );

    m.def("isect_tiles", &gsplat::isect_tiles_tensor);
    m.def("isect_offset_encode", &gsplat::isect_offset_encode_tensor);

    m.def("rasterize_to_pixels_fwd", &gsplat::rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);

    m.def(
        "rasterize_to_indices_in_range",
        &gsplat::rasterize_to_indices_in_range_tensor
    );

    // packed version
    m.def(
        "fully_fused_projection_packed_fwd",
        &gsplat::fully_fused_projection_packed_fwd_tensor
    );
    m.def(
        "fully_fused_projection_packed_bwd",
        &gsplat::fully_fused_projection_packed_bwd_tensor
    );

    m.def("compute_relocation", &gsplat::compute_relocation_tensor);
}