#include "rasterize.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// function to interface torch tensors and lower level pointer operations
std::tuple<
    int,
    torch::Tensor, // output image
    torch::Tensor // ouptut radii
>
rasterize_forward_tensor(
        const torch::Tensor& means3d,
        const torch::Tensor& scales,
        const float glob_scale,
        const torch::Tensor& rotations_quat,
        const torch::Tensor& colors,
        const torch::Tensor& opacity,
        const torch::Tensor& view_matrix,
        const torch::Tensor& proj_matrix,
        const int img_height,
        const int img_width
) {
    CHECK_INPUT(means3d);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations_quat);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacity);
    CHECK_INPUT(view_matrix);
    CHECK_INPUT(proj_matrix);

    if (means3d.ndimension() != 2 || means3d.size(1) != 3) {
        AT_ERROR("means3d must have dimensions (num_points, 3)");
    }

    int rendered = 0;
    const int num_points = means3d.size(0);
    auto int_opts = means3d.options().dtype(torch::kInt32);
    auto float_opts = means3d.options().dtype(torch::kFloat32);
    torch::Tensor out_img = torch::full({3, img_height, img_width}, 0.0, float_opts);
    torch::Tensor out_radii = torch::full({num_points}, 0, int_opts);

    return  std::make_tuple(rendered, out_img, out_radii);
}


int rasterize_forward_impl(
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *quats,
    const float *colors,
    const float *opacity,
    const float *view_matrix,
    const float *proj_matrix,
    const int img_height,
    const int img_width,
    float *out_img,
    int *out_radii
) {
    // launch projection of 3d gaussians into 2d
    // project_gaussians_forward_impl(...)

    // do the fancy logic with allocating resources according to tiles we touch
    // and sort the gaussians that we've touched

    // launch final rasterization method
    // render_forward_impl(...)
}
