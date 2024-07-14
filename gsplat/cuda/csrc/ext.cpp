#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_sh_fwd", &compute_sh_fwd_tensor);
    m.def("compute_sh_bwd", &compute_sh_bwd_tensor);

    m.def("quat_scale_to_covar_preci_fwd", &quat_scale_to_covar_preci_fwd_tensor);
    m.def("quat_scale_to_covar_preci_bwd", &quat_scale_to_covar_preci_bwd_tensor);

    m.def("persp_proj_fwd", &persp_proj_fwd_tensor);
    m.def("persp_proj_bwd", &persp_proj_bwd_tensor);

    m.def("world_to_cam_fwd", &world_to_cam_fwd_tensor);
    m.def("world_to_cam_bwd", &world_to_cam_bwd_tensor);

    m.def("fully_fused_projection_fwd", &fully_fused_projection_fwd_tensor);
    m.def("fully_fused_projection_bwd", &fully_fused_projection_bwd_tensor);

    m.def("isect_tiles", &isect_tiles_tensor);
    m.def("isect_offset_encode", &isect_offset_encode_tensor);

    m.def("rasterize_to_pixels_fwd", &rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &rasterize_to_pixels_bwd_tensor);

    m.def("raytracing_to_pixels_fwd", &raytracing_to_pixels_fwd_tensor);
    m.def("raytracing_to_pixels_bwd", &raytracing_to_pixels_bwd_tensor);

    m.def("rasterize_to_indices_in_range", &rasterize_to_indices_in_range_tensor);

    m.def("compute_3D_smoothing_filter_fwd", &compute_3D_smoothing_filter_fwd_tensor);

    // packed version
    m.def("fully_fused_projection_packed_fwd", &fully_fused_projection_packed_fwd_tensor);
    m.def("fully_fused_projection_packed_bwd", &fully_fused_projection_packed_bwd_tensor);
    
    m.def("compute_relocation", &compute_relocation_tensor);
}