#include <cuda.h>
#include <cuda_runtime.h>

#include "helpers.cuh"

void project_gaussians_forward_impl(
    const int num_points,
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *quats,
    const float *viewmat,
    const float *projmat,
    const float fx,
    const float fy,
    const int W,
    const int H,
    const dim3 tile_bounds,
    float *covs3d,
    float *xys,
    float *depths,
    int *radii,
    uint32_t *num_tiles_hit
);

void compute_cumulative_intersects(
    const int num_points,
    const uint32_t *num_tiles_hit,
    uint32_t &num_intersects,
    uint32_t *cum_tiles_hit
);

void bin_and_sort_gaussians(
    const int num_points,
    const int num_intersects,
    const float *xys,
    const float *depths,
    const int *radii,
    const uint32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    uint32_t *gaussian_ids_sorted,
    uint2 *tile_bins
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
__device__ void compute_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
);
