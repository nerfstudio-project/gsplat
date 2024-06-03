#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <glm/glm.hpp>

// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. VT J -> R(n)
__global__ void project_gaussians_backward_kernel(
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

__global__ void rasterize_backward_kernel(
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

__device__ void build_H(
    const glm::vec3 & p_world,
    const float4 & quat,
    const float2 & scale,
    const float* viewmat,
    const float4 & intrins,
    float tan_fovx,
    float tan_fovy,
    const float* ray_transformation,

    // grad input
    const float* dL_dray_transformation,
    // const float* dL_dnormal3D,

    // grad output
    glm::vec3 & v_mean3D,
    glm::vec2 & v_scale,
    glm::vec4 & v_quat
);

__device__ void build_AABB(
    int idx,
    const int * radii,
    const float W,
    const float H,
    const float * ray_transformations,

    // grad output
    float3 * v_mean2Ds,
    float *dL_ray_transformations
);
