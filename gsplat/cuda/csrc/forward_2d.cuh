#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const unsigned block_width,
    const float clip_thresh,
    float* __restrict__ conv3d,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    int32_t* __restrict__ num_tiles_hit,
    float* __restrict__ transMats
);

// compute output color image from binned and sorted gaussians
__global__ void rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
);

// compute output color image from binned and sorted gaussians
__global__ void nd_rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float* __restrict__ out_img,
    const float* __restrict__ background
);

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    float *cov3d
);

__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    const unsigned block_width,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);

__global__ void get_tile_bin_edges(
    const int num_intersects,
    const int64_t* __restrict__ isect_ids_sorted,
    int2* __restrict__ tile_bins
);


__device__ bool build_H(
    const float3& __restrict__ mean3d,
    const float4 __restrict__ intrins,
    const float3 __restrict__ scale,
    const float4 __restrict__ quat,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float* transMat,
    float3 &normal
);

__device__ bool build_AABB(
    const float *transMat,
    float2 & center,
    float2 & extent
);