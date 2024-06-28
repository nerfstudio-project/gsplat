#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "third_party/glm/glm/glm.hpp"


// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. vT J -> R(n)
__global__ void project_gaussians_backward_kernel_3dgs(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float* __restrict__ compensation,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    const float* __restrict__ v_compensation,
    float3* __restrict__ v_cov2d,
    float* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
);

// compute jacobians of output image wrt binned and sorted gaussians
__global__ void nd_rasterize_backward_kernel_3dgs(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);

__global__ void rasterize_backward_kernel_3dgs(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);

__device__ void project_cov3d_ewa_vjp(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float3 &v_cov2d,
    float3 &v_mean3d,
    float *v_cov3d
);

__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    float3 &v_scale,
    float4 &v_quat
);

//====== 2DGS ======//
// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. VT J -> R(n)
__global__ void project_gaussians_backward_kernel_2dgs(
    const int num_points,
    const float3* __restrict__ means3d,
    const float2* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float* __restrict__ ray_transformations,

    // grad input
    float* __restrict__ dL_dray_transformations,
    // const float* __restrict__ dL_dnormal3Ds,

    // grad output
    float3* __restrict__ v_mean3Ds,
    float2* __restrict__ v_scales,
    float4* __restrict__ v_quats,
    float3* __restrict__ v_mean2Ds
);

__global__ void rasterize_backward_kernel_2dgs(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float* __restrict__ ray_transformations,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,

    // grad input
    const float3* __restrict__ dL_doutput,
    const float* __restrict__ dL_doutput_alpha,

    // grad_output
    float2* __restrict__ v_mean2D,
    float* __restrict__ dL_dray_transformation,
    float3* __restrict__ dL_drgb,
    float* __restrict__ dL_dopacity
);

__device__ void build_transform_and_AABB(
    const glm::vec3& p_world,
    const float4& quat,
    const float2& scale,
    const float* viewmat,
    const float4& intrins,
    const int* radii,
    float tan_fovx,
    float tan_fovy,
    const float* ray_transformation,

    // grad
    float* v_ray_transformations,
    glm::vec3& v_mean3D,
    glm::vec2& v_scale,
    glm::vec4& v_quat,
    float3& v_mean2D
);
