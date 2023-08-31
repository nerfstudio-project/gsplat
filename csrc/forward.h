#include <cuda.h>
#include <cuda_runtime.h>


void project_gaussians_forward_impl(
    const int num_points,
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *rots_quat,
    const float *viewmat,
    float *covs3d,
    float *covs2d
);

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *rots_quat,
    const float *viewmat,
    float *covs3d,
    float *covs2d
);

__device__ void project_cov3d_ewa(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    // const float tan_fovx,
    // const float tan_fovy,
    float *cov2d
);

__device__ void compute_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
);
