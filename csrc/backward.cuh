#include <cuda.h>
#include <cuda_runtime.h>

// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. vT J -> R(n)
void project_gaussians_backward_impl(
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
    const float *cov3d,
    const int *radii,
    const float3 *conics,
    const float2 *v_xy,
    const float3 *v_conic,
    float3 *v_cov2d,
    float *v_cov3d,
    float3 *v_mean3d,
    float3 *v_scale,
    float4 *v_quat
);

// compute jacobians of output image wrt binned and sorted gaussians
void rasterize_backward_impl(
    const dim3 tile_bounds,
    const dim3 block,
    const dim3 img_size,
    const int *gaussians_ids_sorted,
    const int2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float *rgbs,
    const float *opacities,
    const float *background,
    const float *final_Ts,
    const int *final_index,
    const float *v_output,
    float2 *v_xy,
    float3 *v_conic,
    float *v_rgb,
    float *v_opacity
);

__host__ __device__ void project_cov3d_ewa_vjp(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float3 &v_cov2d,
    float3 &v_mean3d,
    float *v_cov3d
);

__host__ __device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    float3 &v_scale,
    float4 &v_quat
);
