#include "backward.cuh"
#include "forward.cuh"
#include "helpers.cuh"
#include "rasterize.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#define CHECK_CUDA(x)                                                          \
    AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)


std::
    tuple<
        torch::Tensor, // output image
        torch::Tensor, // final_Ts
        torch::Tensor, // final_idx
        torch::Tensor, // tile_bins
        torch::Tensor, // gaussian_ids_sorted
        torch::Tensor, // gaussian_ids_unsorted
        torch::Tensor, // isect_ids_sorted
        torch::Tensor  // isect_ids_unsorted
        >
    slow_rasterize_forward_tensor(
        const torch::Tensor &xys,
        const torch::Tensor &depths,
        const torch::Tensor &radii,
        const torch::Tensor &conics,
        const torch::Tensor &num_tiles_hit,
        const torch::Tensor &colors,
        const torch::Tensor &opacity,
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &background
    ) {
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(num_tiles_hit);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacity);
    CHECK_INPUT(background);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (N, 2)");
    }

    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have dimensions (N, 3)");
    }

    const int channels = colors.size(1);
    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1};
    const dim3 block = {BLOCK_X, BLOCK_Y, 1};
    const dim3 img_size = {img_width, img_height, 1};
    int num_tiles = tile_bounds.x * tile_bounds.y;

    // int32_t *cum_tiles_hit_d;
    // cudaMalloc((void **)&cum_tiles_hit_d, num_points * sizeof(int32_t));
    torch::Tensor cum_tiles_hit =
        torch::zeros({num_points}, xys.options().dtype(torch::kInt32));

    int32_t num_intersects;
    compute_cumulative_intersects(
        num_points,
        num_tiles_hit.contiguous().data_ptr<int32_t>(),
        num_intersects,
        cum_tiles_hit.contiguous().data_ptr<int32_t>()
    );

    torch::Tensor gaussian_ids_sorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_sorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));
    torch::Tensor tile_bins =
        torch::zeros({num_tiles, 2}, xys.options().dtype(torch::kInt32));
    // allocate temporary variables
    // TODO dunno be smarter about this?
    // int64_t *isect_ids_unsorted_d;
    // int32_t *gaussian_ids_unsorted_d;
    // int64_t *isect_ids_sorted_d;
    // cudaMalloc((void **)&isect_ids_sorted_d, num_intersects *
    // sizeof(int64_t)); cudaMalloc(
    //     (void **)&isect_ids_unsorted_d, num_intersects * sizeof(int64_t)
    // );
    // cudaMalloc(
    //     (void **)&gaussian_ids_unsorted_d, num_intersects * sizeof(int32_t)
    // );
    // int32_t *sorted_ids_ptr =
    // gaussian_ids_sorted.contiguous().data_ptr<int32_t>(); int2 *bins_ptr =
    // (int2 *) tile_bins.contiguous().data_ptr<int>(); float2 *xys_ptr =
    // (float2 *) xys.contiguous().data_ptr<float>();

    bin_and_sort_gaussians(
        num_points,
        num_intersects,
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        (int32_t *)cum_tiles_hit.contiguous().data_ptr<int>(),
        tile_bounds,
        // isect_ids_unsorted_d,
        // gaussian_ids_unsorted_d,
        // isect_ids_sorted_d,
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>(),
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
        // sorted_ids_ptr,
        // bins_ptr
    );

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    slow_rasterize_forward_impl(
        tile_bounds,
        block,
        img_size,
        channels,
        // sorted_ids_ptr,
        // bins_ptr,
        // xys_ptr,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        out_img.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>()
    );

    // cudaFree(cum_tiles_hit_d);
    // cudaFree(isect_ids_unsorted_d);
    // cudaFree(isect_ids_sorted_d);
    // cudaFree(gaussian_ids_unsorted_d);

    return std::make_tuple(
        out_img,
        final_Ts,
        final_idx,
        tile_bins,
        gaussian_ids_sorted,
        isect_ids_sorted,
        gaussian_ids_unsorted,
        isect_ids_unsorted
    );
}


std::
    tuple<
        torch::Tensor, // output image
        torch::Tensor, // final_Ts
        torch::Tensor, // final_idx
        torch::Tensor, // tile_bins
        torch::Tensor, // gaussian_ids_sorted
        torch::Tensor, // gaussian_ids_unsorted
        torch::Tensor, // isect_ids_sorted
        torch::Tensor  // isect_ids_unsorted
        >
    rasterize_forward_tensor(
        const torch::Tensor &xys,
        const torch::Tensor &depths,
        const torch::Tensor &radii,
        const torch::Tensor &conics,
        const torch::Tensor &num_tiles_hit,
        const torch::Tensor &colors,
        const torch::Tensor &opacity,
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &background
    ) {
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(num_tiles_hit);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacity);
    CHECK_INPUT(background);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (N, 2)");
    }

    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have dimensions (N, 3)");
    }

    const int channels = colors.size(1);
    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1};
    const dim3 block = {BLOCK_X, BLOCK_Y, 1};
    const dim3 img_size = {img_width, img_height, 1};
    int num_tiles = tile_bounds.x * tile_bounds.y;

    // int32_t *cum_tiles_hit_d;
    // cudaMalloc((void **)&cum_tiles_hit_d, num_points * sizeof(int32_t));
    torch::Tensor cum_tiles_hit =
        torch::zeros({num_points}, xys.options().dtype(torch::kInt32));

    int32_t num_intersects;
    compute_cumulative_intersects(
        num_points,
        num_tiles_hit.contiguous().data_ptr<int32_t>(),
        num_intersects,
        cum_tiles_hit.contiguous().data_ptr<int32_t>()
    );

    torch::Tensor gaussian_ids_sorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_sorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));
    torch::Tensor tile_bins =
        torch::zeros({num_tiles, 2}, xys.options().dtype(torch::kInt32));
    // allocate temporary variables
    // TODO dunno be smarter about this?
    // int64_t *isect_ids_unsorted_d;
    // int32_t *gaussian_ids_unsorted_d;
    // int64_t *isect_ids_sorted_d;
    // cudaMalloc((void **)&isect_ids_sorted_d, num_intersects *
    // sizeof(int64_t)); cudaMalloc(
    //     (void **)&isect_ids_unsorted_d, num_intersects * sizeof(int64_t)
    // );
    // cudaMalloc(
    //     (void **)&gaussian_ids_unsorted_d, num_intersects * sizeof(int32_t)
    // );
    // int32_t *sorted_ids_ptr =
    // gaussian_ids_sorted.contiguous().data_ptr<int32_t>(); int2 *bins_ptr =
    // (int2 *) tile_bins.contiguous().data_ptr<int>(); float2 *xys_ptr =
    // (float2 *) xys.contiguous().data_ptr<float>();

    bin_and_sort_gaussians(
        num_points,
        num_intersects,
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        (int32_t *)cum_tiles_hit.contiguous().data_ptr<int>(),
        tile_bounds,
        // isect_ids_unsorted_d,
        // gaussian_ids_unsorted_d,
        // isect_ids_sorted_d,
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>(),
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
        // sorted_ids_ptr,
        // bins_ptr
    );

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    rasterize_forward_impl(
        tile_bounds,
        block,
        img_size,
        channels,
        // sorted_ids_ptr,
        // bins_ptr,
        // xys_ptr,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        out_img.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>()
    );

    // cudaFree(cum_tiles_hit_d);
    // cudaFree(isect_ids_unsorted_d);
    // cudaFree(isect_ids_sorted_d);
    // cudaFree(gaussian_ids_unsorted_d);

    return std::make_tuple(
        out_img,
        final_Ts,
        final_idx,
        tile_bins,
        gaussian_ids_sorted,
        isect_ids_sorted,
        gaussian_ids_unsorted,
        isect_ids_unsorted
    );
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    slow_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output // dL_dout_color
    ) {

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1};
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    torch::Tensor workspace;
    if (channels > 3){
        workspace = torch::zeros({img_height, img_width, channels}, xys.options().dtype(torch::kFloat32));
    } else {
        workspace = torch::zeros({0}, xys.options().dtype(torch::kFloat32));
    }

    slow_rasterize_backward_impl(
        tile_bounds,
        block,
        img_size,
        channels,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        v_output.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        workspace.data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output // dL_dout_color
    ) {

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1};
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    torch::Tensor workspace;
    if (channels > 3){
        workspace = torch::zeros({img_height, img_width, channels}, xys.options().dtype(torch::kFloat32));
    } else {
        workspace = torch::zeros({0}, xys.options().dtype(torch::kFloat32));
    }

    rasterize_backward_impl(
        tile_bounds,
        block,
        img_size,
        channels,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        v_output.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        workspace.data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}
