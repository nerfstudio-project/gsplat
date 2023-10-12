#include "cuda_runtime.h"

__host__ __device__ float3 projectMean2DBackward(
    const float3 m, const float *proj, const float2 dL_dmean2D
);

__host__ __device__ void computeCov3DBackward(
    const float3 scale,
    const float mod,
    const float4 rot,
    const float *dL_dcov3D,
    float3 &dL_dscale,
    float4 &dL_dq
);

__host__ __device__ float3
computeConicBackward(const float3 &cov2D, const float3 &dL_dconic);

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

__host__ __device__ void rasterizeBackward(
    const int N,
    const int W,
    const int H,
    const float2 pixf,
    const float2 *collected_xy,
    const float4 *collected_conic_opacity,
    const float *collected_colors,
    const float T_final,
    const float *dL_dpixel,
    float *dL_dcolor,
    float *dL_dopacity,
    float2 *dL_dmean2D,
    float3 *dL_dconic2D
);

template <int CHANNELS>
__host__ __device__ void rasterize_vjp(
    const int N,
    const float2 p,
    const float2 *xys,
    const float3 *conics,
    const float *opacities,
    const float *rgbs,
    const float T_final,
    const float *v_out,
    float *dL_dcolor,
    float *dL_dopacity,
    float2 *dL_dmean2D,
    float3 *dL_dconic2D
);
