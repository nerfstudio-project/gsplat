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
std::
    tuple<
        int,
        torch::Tensor, // output image
        torch::Tensor,  // output radii
        torch::Tensor, // final_Ts
        torch::Tensor, // final_idx
        torch::Tensor, // gaussian_ids_sorted 
        torch::Tensor, // tile_bins
        torch::Tensor, // xys
        torch::Tensor  // conics
        >
    rasterize_forward_tensor(
        const torch::Tensor &means3d,
        const torch::Tensor &scales,
        const float glob_scale,
        const torch::Tensor &rotations_quat,
        const torch::Tensor &colors,
        const torch::Tensor &opacity,
        const torch::Tensor &view_matrix,
        const torch::Tensor &proj_matrix,
        const int img_height,
        const int img_width,
        const float fx,
        const float fy
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


    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, 3}, means3d.options().dtype(torch::kFloat32)
    );
    torch::Tensor out_radii =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));

    torch::Tensor final_Ts =
        torch::zeros({img_width,img_height}, means3d.options().dtype(torch::kFloat32));

    torch::Tensor final_idx =
        torch::zeros({img_width,img_height}, means3d.options().dtype(torch::kInt32));

    torch::Tensor gaussian_ids_sorted =
        torch::zeros({img_width,img_height}, means3d.options().dtype(torch::kInt32));
    
    // TILE BINS
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1};

    int num_tiles = tile_bounds.x * tile_bounds.y;

    torch::Tensor tile_bins =  torch::zeros({num_tiles, 1}, means3d.options().dtype(torch::kInt32));
    
    torch::Tensor xy =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));

    torch::Tensor conics =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));

    rasterize_forward_impl(
        num_points,
        means3d.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        glob_scale,
        rotations_quat.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        view_matrix.contiguous().data_ptr<float>(),
        proj_matrix.contiguous().data_ptr<float>(),
        img_height,
        img_width,
        fx,
        fy,
        out_img.contiguous().data_ptr<float>(),
        out_radii.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        gaussian_ids_sorted.contiguous().data_ptr<int>(),
        tile_bins.contiguous().data_ptr<int>(),
        xy.contiguous().data_ptr<float>(),
        conics.contiguous().data_ptr<float>()
    );

    return std::make_tuple(rendered, out_img, out_radii, final_Ts, final_idx, gaussian_ids_sorted, tile_bins, xy, conics);
}

int rasterize_forward_impl(
    const int num_points,
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
    float *out_radii,
    float *final_Ts,
    int *final_idx,
    int *gaussian_ids_sorted,
    int *tile_bins,
    float *xy,
    float *conics
) {
    const int channels = 3; // TODO: make this a var
    const int W = img_width;
    const int H = img_height;

    // launch projection of 3d gaussians into 2d
    // project_gaussians_forward_impl(...)
    float3 *scales_d, *means_d;
    float *rgbs_d;
    float4 *quats_d;
    float *viewmat_d;
    float *opacities_d;
    int num_view = 16; // 16 entries in 4x4 projection matrix
    cudaMalloc((void **)&scales_d, num_points * sizeof(float3));
    cudaMalloc((void **)&means_d, num_points * sizeof(float3));
    cudaMalloc((void **)&quats_d, num_points * sizeof(float4));
    cudaMalloc((void **)&rgbs_d, num_points * sizeof(float) * channels);
    cudaMalloc((void **)&opacities_d, num_points * sizeof(float));
    cudaMalloc((void **)&viewmat_d, num_view * sizeof(float));

    cudaMemcpy(
        scales_d, scales, num_points * sizeof(float3), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        means_d, means3d, num_points * sizeof(float3), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        rgbs_d, colors, num_points * sizeof(float) * channels, cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        opacities_d, opacity, num_points * sizeof(float), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        quats_d, quats, num_points * sizeof(float4), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        viewmat_d, view_matrix, num_view * sizeof(float), cudaMemcpyHostToDevice
    );

    // allocate memory for outputs
    int num_cov3d = num_points * 6;
    float *covs3d = new float[num_cov3d];
    //float2 *xy = new float2[num_points]; // 2d mean locations
    float *z = new float[num_points]; // depths
    // float3 *conics = new float3[num_points]; // inverse of cov2ds
    int32_t *num_tiles_hit = new int32_t[num_points]; // num_tiles_hit[gauss_idx]=tile_area i.e. tiles that 2D gaussian projects to within 3 stds. 

    float *covs3d_d, *z_d;
    float2 *xy_d;
    float3 *conics_d;
    int *radii_d; // radius of 2D gaussians in screen space
    int32_t *num_tiles_hit_d;
    cudaMalloc((void **)&covs3d_d, num_cov3d * sizeof(float));
    cudaMalloc((void **)&xy_d, num_points * sizeof(float2));
    cudaMalloc((void **)&z_d, num_points * sizeof(float));
    cudaMalloc((void **)&radii_d, num_points * sizeof(int));
    cudaMalloc((void **)&conics_d, num_points * sizeof(float3));
    cudaMalloc((void **)&num_tiles_hit_d, num_points * sizeof(int32_t));

    const dim3 img_size = {img_width, img_height, 1};
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1};
    project_gaussians_forward_impl(
        num_points,
        means_d,
        scales_d,
        1.f,
        quats_d,
        viewmat_d,
        viewmat_d,
        fx,
        fy,
        img_size,
        tile_bounds,
        covs3d_d,
        xy_d,
        z_d,
        radii_d,
        conics_d,
        num_tiles_hit_d
    );
    cudaMemcpy(
        covs3d, covs3d_d, num_cov3d * sizeof(float), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(xy, xy_d, num_points * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, z_d, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        out_radii, radii_d, num_points * sizeof(int), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        num_tiles_hit,
        num_tiles_hit_d,
        num_points * sizeof(int32_t),
        cudaMemcpyDeviceToHost
    );

    int32_t num_intersects;
    int32_t *cum_tiles_hit = new int32_t[num_points];
    int32_t *cum_tiles_hit_d;
    cudaMalloc((void **)&cum_tiles_hit_d, num_points * sizeof(int32_t));
    compute_cumulative_intersects(
        num_points, num_tiles_hit_d, num_intersects, cum_tiles_hit_d
    );

    int64_t *isect_ids_sorted_d;
    int32_t *gaussian_ids_sorted_d; // sorted by tile and depth
    int64_t *isect_ids_sorted = new int64_t[num_intersects];
    //int32_t *gaussian_ids_sorted = new int32_t[num_intersects];
    cudaMalloc((void **)&isect_ids_sorted_d, num_intersects * sizeof(int64_t));
    cudaMalloc(
        (void **)&gaussian_ids_sorted_d, num_intersects * sizeof(int32_t)
    );

    int64_t *isect_ids_unsorted_d;
    int32_t *gaussian_ids_unsorted_d; // sorted by tile and depth
    int64_t *isect_ids_unsorted = new int64_t[num_intersects];
    int32_t *gaussian_ids_unsorted = new int32_t[num_intersects];
    cudaMalloc(
        (void **)&isect_ids_unsorted_d, num_intersects * sizeof(int64_t)
    );
    cudaMalloc(
        (void **)&gaussian_ids_unsorted_d, num_intersects * sizeof(int32_t)
    );

    int num_tiles = tile_bounds.x * tile_bounds.y;
    uint2 *tile_bins_d; // start and end indices for each tile
    //uint2 *tile_bins = new uint2[num_tiles];
    cudaMalloc((void **)&tile_bins_d, num_tiles * sizeof(uint2));

    bin_and_sort_gaussians(
        num_points,
        num_intersects,
        xy_d,
        z_d,
        radii_d,
        cum_tiles_hit_d,
        tile_bounds,
        isect_ids_unsorted_d,
        gaussian_ids_unsorted_d,
        isect_ids_sorted_d,
        gaussian_ids_sorted_d,
        tile_bins_d
    );

    cudaMemcpy(
        isect_ids_unsorted,
        isect_ids_unsorted_d,
        num_intersects * sizeof(int64_t),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        gaussian_ids_unsorted,
        gaussian_ids_unsorted_d,
        num_intersects * sizeof(int32_t),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        isect_ids_sorted,
        isect_ids_sorted_d,
        num_intersects * sizeof(int64_t),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        gaussian_ids_sorted,
        gaussian_ids_sorted_d,
        num_intersects * sizeof(int32_t),
        cudaMemcpyDeviceToHost
    );

    // launch final rasterization method
    // rasterize_forward_impl(...)
    float *final_Ts_d;
    int *final_idx_d;
    float *out_img_d;
    cudaMalloc((void **)&out_img_d, W * H * sizeof(float)*channels);
    cudaMalloc((void **)&final_Ts_d, W * H * sizeof(float));
    cudaMalloc((void **)&final_idx_d, W * H * sizeof(int));

    const dim3 block = {
        BLOCK_X, BLOCK_Y, 1}; // TODO: make this a user custom setting.

    

    rasterize_forward_impl( // Should this be renamed? it is overloaded with two implementations
        tile_bounds,
        block,
        img_size,
        channels,
        gaussian_ids_sorted_d,
        tile_bins_d,
        xy_d,
        conics_d,
        rgbs_d,
        opacities_d,
        final_Ts_d,
        final_idx_d,
        out_img_d
    );

    // Handle outputs
    cudaMemcpy(
        out_img, out_img_d, W * H * sizeof(float3), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        final_Ts, final_Ts_d, W * H * sizeof(float3), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        final_idx, final_idx_d, W * H * sizeof(int), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        conics, conics_d, num_points * sizeof(float3), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        tile_bins, tile_bins_d, num_tiles * sizeof(uint2), cudaMemcpyDeviceToHost
    );

    return 0;
}


std::
    tuple<
        torch::Tensor, // dL_dcolors
        torch::Tensor // dL_dopacity
        >
    rasterize_backward_tensor(
        const torch::Tensor &means3D,
        const torch::Tensor &colors,
        const torch::Tensor &scales,
        const torch::Tensor &v_output, //dL_dout_color
        const int img_height,
        const int img_width,
        const float fx,
        const float fy,
        const torch::Tensor gaussians_ids_sorted,
        const torch::Tensor tile_bins,
        const torch::Tensor xy,
        const torch::Tensor conics,
        const torch::Tensor opacities,
        const torch::Tensor final_Ts,
        const torch::Tensor final_idx
    ){
    
        CHECK_INPUT(means3D);
        CHECK_INPUT(scales);
        CHECK_INPUT(colors);

        const int num_points = means3D.size(0);
        const int H = v_output.size(1);
        const int W = v_output.size(2);

        torch::Tensor v_xy = torch::zeros({num_points, 3}, means3D.options());
        torch::Tensor v_colors = torch::zeros({num_points, 3}, means3D.options());
        torch::Tensor v_conic = torch::zeros({num_points, 2, 2}, means3D.options());
        torch::Tensor v_opacity = torch::zeros({num_points, 1}, means3D.options());

        const dim3 tile_bounds = {
            (img_width + BLOCK_X - 1) / BLOCK_X,
            (img_height + BLOCK_Y - 1) / BLOCK_Y,
            1};

        if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
            AT_ERROR("means3d must have dimensions (num_points, 3)");
        }
        
	    const dim3 block(BLOCK_X, BLOCK_Y, 1);

        const dim3 img_size = {img_width,img_height,1};
        
        const int channels = 3; // TODO: make this a user var

        rasterize_backward_impl(
            tile_bounds,
            block,
            img_size,
            channels,
            gaussians_ids_sorted.contiguous().data_ptr<int>(),
            (uint2 *)tile_bins.contiguous().data_ptr<int>(), 
            (float2 *)xy.contiguous().data_ptr<float>(),
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

        return std::make_tuple(v_colors,v_opacity);
    }