from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


nd_rasterize_forward_3dgs = _make_lazy_cuda_func("nd_rasterize_forward_3dgs")
nd_rasterize_backward_3dgs = _make_lazy_cuda_func("nd_rasterize_backward_3dgs")
rasterize_forward_3dgs = _make_lazy_cuda_func("rasterize_forward_3dgs")
rasterize_backward_3dgs = _make_lazy_cuda_func("rasterize_backward_3dgs")
compute_cov2d_bounds = _make_lazy_cuda_func("compute_cov2d_bounds")
project_gaussians_forward_3dgs = _make_lazy_cuda_func("project_gaussians_forward_3dgs")
project_gaussians_backward_3dgs = _make_lazy_cuda_func("project_gaussians_backward_3dgs")
compute_sh_forward = _make_lazy_cuda_func("compute_sh_forward")
compute_sh_backward = _make_lazy_cuda_func("compute_sh_backward")
map_gaussian_to_intersects = _make_lazy_cuda_func("map_gaussian_to_intersects")
get_tile_bin_edges = _make_lazy_cuda_func("get_tile_bin_edges")
rasterize_forward_3dgs = _make_lazy_cuda_func("rasterize_forward_3dgs")
nd_rasterize_forward_3dgs = _make_lazy_cuda_func("nd_rasterize_forward_3dgs")

###### 2DGS ######
rasterize_forward_2dgs = _make_lazy_cuda_func("rasterize_forward_2dgs")
rasterize_backward_2dgs = _make_lazy_cuda_func("rasterize_backward_2dgs")
project_gaussians_forward_2dgs = _make_lazy_cuda_func("project_gaussians_forward_2dgs")
project_gaussians_backward_2dgs = _make_lazy_cuda_func("project_gaussians_backward_2dgs")
