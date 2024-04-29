#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quat_scale_to_covar_perci_fwd", &quat_scale_to_covar_perci_fwd_tensor);
    m.def("quat_scale_to_covar_perci_bwd", &quat_scale_to_covar_perci_bwd_tensor);

    m.def("persp_proj_fwd", &persp_proj_fwd_tensor);
    m.def("persp_proj_bwd", &persp_proj_bwd_tensor);

    m.def("world_to_cam_fwd", &world_to_cam_fwd_tensor);
    m.def("world_to_cam_bwd", &world_to_cam_bwd_tensor);

    m.def("projection_fwd", &projection_fwd_tensor);
    m.def("projection_bwd", &projection_bwd_tensor);

    m.def("isect_tiles", &isect_tiles_tensor);
    m.def("isect_offset_encode", &isect_offset_encode_tensor);

    m.def("rasterize_to_pixels_fwd", &rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &rasterize_to_pixels_bwd_tensor);
}