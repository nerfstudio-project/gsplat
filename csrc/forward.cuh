#include <cuda.h>
#include <cuda_runtime.h>

#include "helpers.cuh"

// compute the 2d gaussian parameters from 3d gaussian parameters
void project_gaussians_forward_impl(
    const int num_points,
    const float3 *means3d,
    const float3 *scales,
    const float glob_scale,
    const float4 *quats,
    const float *viewmat,
    const float *projmat,
    const float fx,
    const float fy,
    const dim3 img_size,
    const dim3 tile_bounds,
    float *covs3d,
    float2 *xys,
    float *depths,
    int *radii,
    float3 *conics,
    uint32_t *num_tiles_hit
);

void compute_cumulative_intersects(
    const int num_points,
    const uint32_t *num_tiles_hit,
    uint32_t &num_intersects,
    uint32_t *cum_tiles_hit
);

// bin and sort gaussians by tile and depth
void bin_and_sort_gaussians(
    const int num_points,
    const int num_intersects,
    const float2 *xys,
    const float *depths,
    const int *radii,
    const uint32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    uint64_t *isect_ids_unsorted,
    uint32_t *gaussian_ids_unsorted,
    uint64_t *isect_ids_sorted,
    uint32_t *gaussian_ids_sorted,
    uint2 *tile_bins
);

// compute output color image from binned and sorted gaussians
void rasterize_forward_impl(
    const dim3 tile_bounds,
    const dim3 block,
    const dim3 img_size,
    const uint32_t *gaussian_ids_sorted,
    const uint2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float3 *rgbs,
    const float *opacities,
    float3 *out_img
);

// device helper to approximate projected 2d cov from 3d mean and cov
__device__ float3 project_cov3d_ewa(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy
    // const float tan_fovx,
    // const float tan_fovy,
);

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
);
