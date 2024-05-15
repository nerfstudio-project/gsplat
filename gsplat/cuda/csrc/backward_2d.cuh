#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. VT J -> R(n)
__global__ void project_gaussians_backward_kernel(
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
    const float* __restrict__ transMats,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_transMats,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
);

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float* __restrict__ transMats,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float* __restrict__ v_transMats,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);

__device__ void build_H(
    const glm::vec3 & p_world,
    const glm::vec4 & quat,
    const glm::vec2 & scale,
    const float* viewmat,
    const float* projmat,
    const int W,
    const int H,
    const float* transMat,
    const float* v_transMat,
    const float* v_normal3D,
    glm::vec3 & v_mean3D,
    glm::vec2 & v_scale,
    glm::vec4 & v_rot
);

__device__ void build_AABB(
    int P,
    const int * radii,
    const float W,
    const float H,
    const float * transMats,
    float3 * v_mean2D,
    float *v_transMat
);
