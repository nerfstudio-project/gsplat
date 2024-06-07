#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions
    // m.def("nd_rasterize_forward", &nd_rasterize_forward_tensor);
    // m.def("nd_rasterize_backward", &nd_rasterize_backward_tensor);
    //====== 2DGS ======//
    m.def("rasterize_forward_2dgs", &rasterize_forward_tensor_2dgs);
    m.def("rasterize_backward_2dgs", &rasterize_backward_tensor_2dgs);
    m.def("project_gaussians_forward_2dgs", &project_gaussians_forward_tensor_2dgs);
    m.def("project_gaussians_backward_2dgs", &project_gaussians_backward_tensor_2dgs);


    m.def("compute_sh_forward", &compute_sh_forward_tensor);
    m.def("compute_sh_backward", &compute_sh_backward_tensor);
    // utils
    m.def("compute_cov2d_bounds", &compute_cov2d_bounds_tensor);
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);
}
