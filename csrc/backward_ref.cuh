#include "cuda_runtime.h"


__host__ __device__ float3 projectMean2DBackward(
    const float3 m, const float* proj, const float2 dL_dmean2D
);

__host__ __device__ void computeCov3DBackward(
    const float3 scale,
    const float mod,
    const float4 rot,
    const float* dL_dcov3D,
    float3 &dL_dscale,
    float4 &dL_dq
);

__host__ __device__ float3 computeConicBackward(
    const float3 &cov2D,
    const float3 &dL_dconic
);

__host__ __device__ void computeCov2DBackward(
    const float3 &mean,
    const float *cov3D,
    const float *view_matrix,
    const float h_x,
    const float h_y,
	const float tan_fovx,
    const float tan_fovy,
    const float3 &dL_dcov2d,
    float3 &dL_dmean,
    float *dL_dcov
);
