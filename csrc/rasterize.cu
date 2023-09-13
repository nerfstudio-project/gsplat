#include "forward.cuh"
#include "backward.cuh"
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

// function to interface torch tensors and lower level pointer operations
std::tuple<
    torch::Tensor, // output image
    torch::Tensor, // final_Ts
    torch::Tensor, // final_idx
    torch::Tensor, // tile_bins
    torch::Tensor, // gaussian_ids_sorted 
    torch::Tensor, // gaussian_ids_unsorted 
    torch::Tensor, // isect_ids_sorted 
    torch::Tensor // isect_ids_unsorted 
>
rasterize_forward_tensor(
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &conics,
    const torch::Tensor &num_tiles_hit,
    const torch::Tensor &colors,
    const torch::Tensor &opacity,
    const int img_height,
    const int img_width
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(num_tiles_hit);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacity);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (N, 2)");
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
    printf("%d num intersects\n", num_intersects);

    torch::Tensor gaussian_ids_sorted = torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor gaussian_ids_unsorted = torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_sorted = torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));
    torch::Tensor isect_ids_unsorted = torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));
    torch::Tensor tile_bins = torch::zeros({num_tiles, 2}, xys.options().dtype(torch::kInt32));
    // allocate temporary variables
    // TODO dunno be smarter about this?
    // int64_t *isect_ids_unsorted_d;
    // int32_t *gaussian_ids_unsorted_d;
    // int64_t *isect_ids_sorted_d;
    // cudaMalloc((void **)&isect_ids_sorted_d, num_intersects * sizeof(int64_t));
    // cudaMalloc(
    //     (void **)&isect_ids_unsorted_d, num_intersects * sizeof(int64_t)
    // );
    // cudaMalloc(
    //     (void **)&gaussian_ids_unsorted_d, num_intersects * sizeof(int32_t)
    // );
    // int32_t *sorted_ids_ptr = gaussian_ids_sorted.contiguous().data_ptr<int32_t>();
    // int2 *bins_ptr = (int2 *) tile_bins.contiguous().data_ptr<int>();
    // float2 *xys_ptr = (float2 *) xys.contiguous().data_ptr<float>();

    bin_and_sort_gaussians(
        num_points,
        num_intersects,
        (float2 *) xys.contiguous().data_ptr<float>(),
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
        (int2 *) tile_bins.contiguous().data_ptr<int>()
        // sorted_ids_ptr,
        // bins_ptr
    );

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts =
        torch::zeros({img_width, img_height}, xys.options().dtype(torch::kFloat32));
    torch::Tensor final_idx =
        torch::zeros({img_width, img_height}, xys.options().dtype(torch::kInt32));

    rasterize_forward_impl(
        tile_bounds,
        block,
        img_size,
        channels,
        // sorted_ids_ptr,
        // bins_ptr,
        // xys_ptr,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *) tile_bins.contiguous().data_ptr<int>(),
        (float2 *) xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        out_img.contiguous().data_ptr<float>()
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

//std::tuple<
//    torch::Tensor, // covs3d
//    torch::Tensor, // xys
//    torch::Tensor, // depths
//    torch::Tensor, // radiii
//    torch::Tensor, // conics
//    torch::Tensor // num_tiles_hit
//    >
//void project_gaussians_tensor(
//    const torch::Tensor &means3d,
//    const torch::Tensor &scales,
//    const float glob_scale,
//    const torch::Tensor &rotations_quat,
//    const torch::Tensor &colors,
//    const torch::Tensor &opacity,
//    const torch::Tensor &view_matrix,
//    const torch::Tensor &proj_matrix,
//    const int img_height,
//    const int img_width,
//    const float fx,
//    const float fy,
//    const int channels
//) {
//    CHECK_INPUT(means3d);
//    CHECK_INPUT(scales);
//    CHECK_INPUT(rotations_quat);
//    CHECK_INPUT(colors);
//    CHECK_INPUT(opacity);
//    CHECK_INPUT(view_matrix);
//    CHECK_INPUT(proj_matrix);
//
//    if (means3d.ndimension() != 2 || means3d.size(1) != 3) {
//        AT_ERROR("means3d must have dimensions (num_points, 3)");
//    }
//    const int num_points = means3d.size(0);
//    torch::Tensor covs3d =
//        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
//
//    torch::Tensor xys =
//        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
//
//    torch::Tensor depths =
//        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
//
//    torch::Tensor radii =
//        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
//
//    torch::Tensor conics =
//        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
//
//    torch::Tensor num_tiles_hit =
//        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
//
//    const dim3 tile_bounds = {
//        (img_width + BLOCK_X - 1) / BLOCK_X,
//        (img_height + BLOCK_Y - 1) / BLOCK_Y,
//        1};
//    const dim3 img_size = {img_width, img_height, 1};
// 
//    project_gaussians_forward_impl(
//        num_points,
//        (float3 *) means3d.contiguous().data_ptr<float>(),
//        (float3 *) scales.contiguous().data_ptr<float>(),
//        glob_scale,
//        (float4 *) rotations_quat.contiguous().data_ptr<float>(),
//        view_matrix.contiguous().data_ptr<float>(),
//        proj_matrix.contiguous().data_ptr<float>(),
//        fx,
//        fy,
//        img_size,
//        tile_bounds,
//        covs3d.contiguous().data_ptr<float>(),
//        (float2 *) xys.contiguous().data_ptr<float>(),
//        depths.contiguous().data_ptr<float>(),
//        radii.contiguous().data_ptr<int>(),
//        (float3 *) conics.contiguous().data_ptr<float>(),
//        num_tiles_hit.contiguous().data_ptr<int>()
//    );
//    return std::make_tuple(covs3d, xys, depths, radii, conics, num_tiles_hit);
//}

int render_gaussians_forward(
    const int num_points,
    const int channels,
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
    const float fx,
    const float fy,
    float *out_img,
    float *radii,
    float *final_Ts,
    int *final_idx,
    int *gaussian_ids_sorted,
    int *tile_bins,
    float *xy,
    float *conics
) {
    const int W = img_width;
    const int H = img_height;

    // 1. launch projection of 3d gaussians into 2d
    // project_gaussians_forward_impl(...)
    int num_cov3d = num_points * 6;
    float *covs3d = new float[num_cov3d];
    float *z = new float[num_points]; // depths
    int32_t *num_tiles_hit = new int32_t[num_points]; // num_tiles_hit[gauss_idx]=tile_area i.e. tiles that 2D gaussian projects to within 3 stds. 

    cudaMalloc((void **)&covs3d, num_cov3d * sizeof(float));
    cudaMalloc((void **)&xy, num_points * sizeof(float2));
    cudaMalloc((void **)&z, num_points * sizeof(float));
    cudaMalloc((void **)&conics, num_points * sizeof(float3));
    cudaMalloc((void **)&num_tiles_hit, num_points * sizeof(int32_t));

    const dim3 img_size = {img_width, img_height, 1};
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1};
    
    project_gaussians_forward_impl(
        num_points,
        (float3 *)means3d,
        (float3 *)scales,
        1.f,
        (float4 *)quats,
        view_matrix,
        proj_matrix,
        fx,
        fy,
        img_size,
        tile_bounds,
        covs3d,
        (float2 *)xy,
        z,
        (int *)radii,
        (float3 *)conics,
        num_tiles_hit
    );

    // 2. sort projected gaussians
    // bin_and_sort_gaussians(...)
    int32_t num_intersects;
    int32_t *cum_tiles_hit = new int32_t[num_points];
    cudaMalloc((void **)&cum_tiles_hit, num_points * sizeof(int32_t));
    compute_cumulative_intersects(
        num_points, num_tiles_hit, num_intersects, cum_tiles_hit
    );

    int64_t *isect_ids_sorted;
    cudaMalloc((void **)&isect_ids_sorted, num_intersects * sizeof(int64_t));

    int64_t *isect_ids_unsorted;
    int32_t *gaussian_ids_unsorted; // sorted by tile and depth
    cudaMalloc(
        (void **)&isect_ids_unsorted, num_intersects * sizeof(int64_t)
    );
    cudaMalloc(
        (void **)&gaussian_ids_unsorted, num_intersects * sizeof(int32_t)
    );

    bin_and_sort_gaussians(
        num_points,
        num_intersects,
        (float2 *)xy,
        z,
        (int *)radii,
        cum_tiles_hit,
        tile_bounds,
        isect_ids_unsorted,
        gaussian_ids_unsorted,
        isect_ids_sorted,
        gaussian_ids_sorted,
        (int2 *)tile_bins
    );

    // 3. launch final rasterization method
    // rasterize_forward_impl(...)
    const dim3 block = {
        BLOCK_X, BLOCK_Y, 1}; // TODO: make this a user custom setting.

    rasterize_forward_impl( // Should this be renamed? it is overloaded with two implementations
        tile_bounds,
        block,
        img_size,
        channels,
        (int32_t *)gaussian_ids_sorted,
        (int2 *)tile_bins,
        (float2 *)xy,
        (float3 *)conics,
        colors,
        opacity,
        final_Ts,
        final_idx,
        out_img
    );

    return 0;
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor // dL_dopacity
        >
    rasterize_backward_tensor(
        const int img_height,
        const int img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output //dL_dout_color
    ){
    
        CHECK_INPUT(xys);
        CHECK_INPUT(colors);

        if (xys.ndimension() != 2 || xys.size(1) != 2) {
            AT_ERROR("xys must have dimensions (num_points, 2)");
        }

        const int num_points = xys.size(0);
        const dim3 tile_bounds = {
            (img_width + BLOCK_X - 1) / BLOCK_X,
            (img_height + BLOCK_Y - 1) / BLOCK_Y,
            1};
        const dim3 block(BLOCK_X, BLOCK_Y, 1);
        const dim3 img_size = {img_width,img_height,1};
        const int channels = colors.size(1);

        torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
        torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
        torch::Tensor v_colors = torch::zeros({num_points, channels}, xys.options());
        torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

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
            final_Ts.contiguous().data_ptr<float>(),
            final_idx.contiguous().data_ptr<int>(),
            v_output.contiguous().data_ptr<float>(),
            (float2 *)v_xy.contiguous().data_ptr<float>(),
            (float3 *)v_conic.contiguous().data_ptr<float>(),
            v_colors.contiguous().data_ptr<float>(),
            v_opacity.contiguous().data_ptr<float>());

        return std::make_tuple(v_xy,v_conic,v_colors,v_opacity);
    }
