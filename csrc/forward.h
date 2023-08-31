#include <cuda.h>
#include <cuda_runtime.h>


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
    float *covs3d,
    float *xys,
    float *depths,
    int *radii
);

__device__ float3 project_cov3d_ewa(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy
    // const float tan_fovx,
    // const float tan_fovy,
);

__device__ void compute_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
);
