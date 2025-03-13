/*
* Copyright (C) 2023, Inria
* GRAPHDECO research group, https://team.inria.fr/graphdeco
* All rights reserved.
*
* This software is free for non-commercial, research and evaluation use 
* under the terms of the LICENSE.md file.
*
* For inquiries contact  george.drettakis@inria.fr
*/

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include <stdio.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define GENERALIZED_GAUSSIAN_KERNEL 0 // use standard Gaussian

#define CHECK_CUDA(call, debug) {                              \
    call;                                                      \
    if(debug) {                                                \
        auto ret = cudaDeviceSynchronize();                    \
        if (ret != cudaSuccess) {                              \
            std::cerr << "\n[CUDA ERROR] in " << __FILE__      \
                    << ":" << __LINE__                       \
                    << ": " << cudaGetErrorString(ret);      \
            throw std::runtime_error(cudaGetErrorString(ret)); \
        }                                                      \
    }                                                          \
}

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// #define DEBUG_PRINT_CUDA // uncomment to enable cuda debug printfs
#ifdef DEBUG_PRINT_CUDA
    #define DEBUG_PRINTF_CUDA(...) if(cg::this_grid().thread_rank() == 0) printf(__VA_ARGS__)
#else
    #define DEBUG_PRINTF_CUDA(...) ((void)0)
#endif

constexpr uint32_t WARP_SIZE = 32U;
constexpr uint32_t WARP_MASK = 0xFFFFFFFFU;

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

#ifndef DONT_DEFINE_SWAP
template<typename T>
__forceinline__ __device__ void swap(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}
#endif

__forceinline__ __device__ float3 make_float3(const float4& f4)
{
    return { f4.x, f4.y, f4.z };
}

__forceinline__ __device__ float sq(float x) 
{ 
    return x * x;
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
    return ((v + 1.0) * S) * 0.5;
}

__forceinline__ __device__ glm::vec3 pix2world(const glm::vec2 pix, const int W, const int H, glm::vec4 inverse_vp0, glm::vec4 inverse_vp1, glm::vec4 inverse_vp3)
{
    const glm::vec2 pix_ndc = pix * glm::vec2(2.0f / W, 2.0f / H) - 1.0f;
    glm::vec4 p_world = inverse_vp0 * pix_ndc.x + inverse_vp1 * pix_ndc.y + inverse_vp3;
    float rcp_w = __frcp_rn(p_world.w);
    return glm::vec3(p_world) * rcp_w;
}

__forceinline__ __device__ glm::vec3 pix2world(const glm::vec2 pix, const int W, const int H, const glm::mat4 inverse_vp)
{
    return pix2world(pix, W, H, inverse_vp[0], inverse_vp[1], inverse_vp[3]);
}

__forceinline__ __device__ glm::vec3 world2ndc(const glm::vec3 p_world, const glm::mat4 viewproj_matrix)
{
    glm::vec4 p_hom = viewproj_matrix * glm::vec4(p_world, 1.0f);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    glm::vec3 p_ndc = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    return p_ndc;
}

__forceinline__ __device__ void getRect(const glm::vec2 c, const glm::vec2 rect_extent, uint2& rect_min, uint2& rect_max, dim3 grid)
{
    // tile bounds covering pixel centers computed from rect centers (c ~ image points) and rect extents
    rect_min = {
        min(grid.x, max((int)0, (int) floorf((c.x - 0.5f - rect_extent.x) / BLOCK_X))),
        min(grid.y, max((int)0, (int) floorf((c.y - 0.5f - rect_extent.y) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max((int)0, (int) ceilf((c.x - 0.5f + rect_extent.x) / BLOCK_X))),
        min(grid.y, max((int)0, (int) ceilf((c.y - 0.5f + rect_extent.y) / BLOCK_Y)))
    };
}

__forceinline__ __device__ glm::mat4x4 loadMatrix4x4(const float* matrix)
{
    glm::mat4x4 mat;
    for (int i = 0; i < 4; i++)
    {
        float4 tmp = *((float4*) (matrix + i * 4));
        mat[i][0] = tmp.x;
        mat[i][1] = tmp.y;
        mat[i][2] = tmp.z;
        mat[i][3] = tmp.w;
    }
    return mat;
}

__forceinline__ __device__ glm::mat4x3 loadMatrix4x3(const float* matrix)
{
    glm::mat4x3 mat;
    for (int i = 0; i < 4; i++)
    {
        float4 tmp = *((float4*) (matrix + i * 4));
        mat[i][0] = tmp.x;
        mat[i][1] = tmp.y;
        mat[i][2] = tmp.z;
    }
    return mat;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    };
    return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
    float4 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    };
    return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
    };
    return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
    };
    return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
    float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

    float3 dnormvdv;
    dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
    dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
    dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

    float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
    float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
    float4 dnormvdv;
    dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
    dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
    dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
    dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
    return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
    const glm::vec3 mean3D,
    const glm::mat4x3 viewmatrix,
    bool prefiltered,
    glm::vec3& p_view)
{
    // Bring points to screen space
    // float4 p_hom = transformPoint4x4(mean3D, projmatrix);
    // float p_w = 1.0f / (p_hom.w + 0.0000001f);
    // float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
    // p_view = transformPoint4x3(p_orig, viewmatrix);

    glm::vec4 p_world(mean3D, 1.0f);
    p_view = viewmatrix * p_world;

    if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
    {
        if (prefiltered)
        {
            printf("Point is filtered although prefiltered is set. This shouldn't happen!");
            __trap();
        }
        return false;
    }
    return true;
}

__forceinline__ __device__ uint64_t constructSortKey(uint32_t tile_id, float depth)
{
    uint64_t key = tile_id;
    key <<= 32;
    key |= *((uint32_t*)&depth);
    return key;
}

// Forward version of 2D covariance matrix computation
__forceinline__ __device__ glm::mat3 computeCov2D(const glm::vec3 p_view, 
    float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, 
    const glm::mat4 viewmatrix, bool is_perfect_equidistant_fisheye)
{
    // The following models the steps outlined by equations 29
    // and 31 in "EWA Splatting" (Zwicker et al., 2002). 
    // Additionally considers aspect / scaling of viewport.
    // Transposes used to account for row-/column-major conventions.
    glm::vec3 t = p_view;

    glm::mat3 J;

    if (!is_perfect_equidistant_fisheye) {
        const float limx = 1.3f * tan_fovx;
        const float limy = 1.3f * tan_fovy;
        const float txtz = t.x / t.z;
        const float tytz = t.y / t.z;
        t.x = min(limx, max(-limx, txtz)) * t.z;
        t.y = min(limy, max(-limy, tytz)) * t.z;

        J = glm::mat3(
            focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
            0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0, 0, 0);

    } else {
        // From FisheyGS (https://github.com/zmliao/Fisheye-GS/blob/master/submodules/fisheye_gs_rasterizer/cuda_rasterizer/forward.cu)
        const float limx = 9982444353.f;
        const float limy = 9982444353.f;
        const float txtz = t.x / t.z;
        const float tytz = t.y / t.z;
        t.x = min(limx, max(-limx, txtz)) * t.z;
        t.y = min(limy, max(-limy, tytz)) * t.z;

        float eps        = 0.0000001f;
        float x2         = t.x * t.x + eps;
        float y2         = t.y * t.y;
        float xy         = t.x * t.y;
        float x2y2       = x2 + y2;
        float len_xy     = length(glm::vec2({t.x, t.y})) + eps;
        float x2y2z2_inv = 1.f / (x2y2 + t.z * t.z);

        float b = glm::atan(len_xy, t.z) / len_xy / x2y2;
        float a = t.z * x2y2z2_inv / (x2y2);
        J = glm::mat3(
            focal_x * (x2 * a + y2 * b), focal_x * xy * (a - b), -focal_x * t.x * x2y2z2_inv,
            focal_y * xy * (a - b), focal_y * (y2 * a + x2 * b), -focal_y * t.y * x2y2z2_inv,
            0, 0, 0);
    }

    glm::mat3 W = glm::transpose(glm::mat3(viewmatrix));
    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    return cov;
}

__device__ inline glm::vec3 dilateCov2D(const glm::mat3 cov, const bool proper_ewa_scaling, float& det_dilated, float& convolution_scaling_factor)
{
    // Apply low-pass filter: every Gaussian should be at least one pixel wide/high. Discard 3rd row and column.
    glm::vec3 cov2D = { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };

    constexpr float h_var = 0.3f;
    cov2D.x += h_var;
    cov2D.z += h_var;
    det_dilated = cov2D.x * cov2D.z - cov2D.y * cov2D.y;

    if (proper_ewa_scaling)
    {
        // As employed by Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting"
        // https://github.com/autonomousvision/mip-splatting
        const float det_orig = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
        // convolution_scaling_factor = sqrt(max(0.000025f, det_orig / det_dilated)); // max for numerical stability
        convolution_scaling_factor = sqrt(det_orig / det_dilated);
    }
    else
    {
        convolution_scaling_factor = 1.0f;
    }

    return cov2D;
}

__forceinline__ __device__ float4 computeConicOpacity(glm::vec3 cov2D, float opacity, float det, float convolution_scaling_factor)
{
    float4 conic_opacity;

    float det_inv = 1.f / det;
    conic_opacity.x = cov2D.z * det_inv;
    conic_opacity.y = -cov2D.y * det_inv;
    conic_opacity.z = cov2D.x * det_inv;

    conic_opacity.w = opacity * convolution_scaling_factor;
    return conic_opacity;
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__forceinline__ __device__ void computeCov3D(const glm::vec3 scale, /*float mod,*/ const glm::vec4 rot, float* cov3D)
{
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = /*mod * */ scale.x;
    S[1][1] = /*mod * */ scale.y;
    S[2][2] = /*mod * */ scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot;// / glm::length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    // const float ql = glm::length(rot);
    // glm::vec4 q = rot;
    // float x = q.y / ql;
    // float y = q.z / ql;
    // float z = q.w / ql;
    // float r = sqrtf(1.f - x * x - y * y - z * z);

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

__forceinline__ __device__ glm::vec3 safe_normalize(glm::vec3 v)
{
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    return l > 0.0f ? (v * rsqrtf(l)) : v;
}

__forceinline__ __device__ void bwd_safe_normalize(const glm::vec3 v, glm::vec3& d_v, glm::vec3 d_out)
{
    float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (l > 0.0f)
    {
        float fac = 1.0 / powf(v.x * v.x + v.y * v.y + v.z * v.z, 1.5f);
        d_v.x += (d_out.x * (v.y * v.y + v.z * v.z) - d_out.y * (v.x * v.y) - d_out.z * (v.x * v.z)) * fac;
        d_v.y += (d_out.y * (v.x * v.x + v.z * v.z) - d_out.x * (v.y * v.x) - d_out.z * (v.y * v.z)) * fac;
        d_v.z += (d_out.z * (v.x * v.x + v.y * v.y) - d_out.x * (v.z * v.x) - d_out.y * (v.z * v.y)) * fac;
    }
}

__forceinline__ __device__ glm::vec3 matmul_bw_vec(const glm::mat3& m, const glm::vec3& gdt)
{
    return glm::vec3(
        gdt.x*m[0].x+gdt.y*m[0].y+gdt.z*m[0].z,
        gdt.x*m[1].x+gdt.y*m[1].y+gdt.z*m[1].z,
        gdt.x*m[2].x+gdt.y*m[2].y+gdt.z*m[2].z
    );
}

__forceinline__ __device__ glm::vec4 matmul_bw_quat(const glm::vec3& p, const glm::vec3& g, const glm::vec4& q)
{
    glm::vec3 dmat[3];
    dmat[0] = g.x*p;
    dmat[1] = g.y*p;
    dmat[2] = g.z*p;
    
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;
    
    float dr = 0;
    float dx = 0;
    float dy = 0;
    float dz = 0;
    
    // m[0] = make_float3((1.f - 2.f * (y * y + z * z)), 2.f * (x * y + r * z), 2.f * (x * z - r * y));
    
    // m[0].x = (1.f - 2.f * (y * y + z * z))
    dy += -4 * y * dmat[0].x;
    dz += -4 * z * dmat[0].x;
    // m[0].y = 2.f * (x * y + r * z)
    dr += 2 * z * dmat[0].y;
    dx += 2 * y * dmat[0].y;
    dy += 2 * x * dmat[0].y;
    dz += 2 * r * dmat[0].y;  
    // m[0].z = 2.f * (x * z - r * y)
    dr += -2 * y * dmat[0].z;
    dx +=  2 * z * dmat[0].z;
    dy += -2 * r * dmat[0].z;
    dz +=  2 * x * dmat[0].z;  
    
    // m[1] = make_float3(2.f * (x * y - r * z), (1.f - 2.f * (x * x + z * z)), 2.f * (y * z + r * x));
    
    // m[1].x = 2.f * (x * y - r * z)
    dr += -2 * z * dmat[1].x;
    dx += 2 * y * dmat[1].x;
    dy += 2 * x * dmat[1].x;
    dz += -2 * r * dmat[1].x;  
    // m[1].y = (1.f - 2.f * (x * x + z * z))
    dx += -4 * x * dmat[1].y;
    dz += -4 * z * dmat[1].y;  
    // m[1].z = 2.f * (y * z + r * x))
    dr += 2 * x * dmat[1].z;
    dx += 2 * r * dmat[1].z;
    dy += 2 * z * dmat[1].z;
    dz += 2 * y * dmat[1].z;  
    
    // m[2] = make_float3(2.f * (x * z + r * y), 2.f * (y * z - r * x), (1.f - 2.f * (x * x + y * y)));
    
    // m[2].x = 2.f * (x * z + r * y)
    dr += 2 * y * dmat[2].x;
    dx += 2 * z * dmat[2].x;
    dy += 2 * r * dmat[2].x;
    dz += 2 * x * dmat[2].x;  
    // m[2].y = 2.f * (y * z - r * x)
    dr += -2 * x * dmat[2].y;
    dx += -2 * r * dmat[2].y;
    dy += 2 * z * dmat[2].y;
    dz += 2 * y * dmat[2].y;  
    // m[2].z = (1.f - 2.f * (x * x + y * y))
    dx += -4 * x * dmat[2].z;
    dy += -4 * y * dmat[2].z;
    
    return glm::vec4(dr,dx,dy,dz);
}

__forceinline__ __device__ glm::vec3 safe_normalize_bw(const glm::vec3& v, const glm::vec3& d_out)
{
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    if (l > 0.0f) {
        const float il = rsqrtf(l);
        const float il3 = (il * il * il);
        return il * d_out - il3 * glm::vec3(d_out.x * (v.x * v.x) + d_out.y * (v.y * v.x) + d_out.z * (v.z * v.x),
                                            d_out.x * (v.x * v.y) + d_out.y * (v.y * v.y) + d_out.z * (v.z * v.y),
                                            d_out.x * (v.x * v.z) + d_out.y * (v.y * v.z) + d_out.z * (v.z * v.z));
    }
    return glm::vec3(0);
}

// ------------------------------------------------------------------------
// Geometric Formula to Compute Gaussian Response
// ------------------------------------------------------------------------

__forceinline__ __device__ float evaluate_opacity_factor3D_geometric(
    const glm::vec3& o_minus_mu, const glm::vec3& raydir, 
    const glm::vec4& grot, const glm::vec3& gscl, 
    glm::vec3& grd, glm::vec3& gro)
{
    const glm::vec3 giscl = 1.f / gscl;

    const float r = grot.x;
    const float x = grot.y;
    const float y = grot.z;
    const float z = grot.w;
    // column major !!!
    const glm::mat3 grot_mat = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    const glm::vec3& gposc = o_minus_mu;
    gro = giscl * (grot_mat * gposc);
    grd = safe_normalize(giscl * (grot_mat * raydir));

    const glm::vec3 gcrod = glm::cross(grd, gro);
    const float grayDist = glm::dot(gcrod, gcrod);

/// generalized gaussian of degree n : scaling is s = -4.5/3^n
#if GENERALIZED_GAUSSIAN_KERNEL == 8 // just because I love the Zenzizenzizenzic kernel
    constexpr float s = -0.000685871056241;
    const float grayDistSq = grayDist * grayDist;
    return s * grayDistSq * grayDistSq;
#elif GENERALIZED_GAUSSIAN_KERNEL == 5 // quintic ?
    constexpr float s = -0.0185185185185;
    return s * grayDist * grayDist * sqrtf(grayDist);
#elif GENERALIZED_GAUSSIAN_KERNEL == 4 // tesseractic
    constexpr float s = -0.0555555555556;
    return s * grayDist * grayDist;
#elif GENERALIZED_GAUSSIAN_KERNEL == 3 // cubic
    constexpr float s = -0.166666666667;
    return  s * grayDist * sqrtf(grayDist);
#else // default to quadratic
    return -0.5f * grayDist;
#endif

}

__forceinline__ __device__ 
float evaluate_opacity_factor3D_and_depth_geometric_bwd(
    const int global_id,
    const glm::vec3& o_minus_mu, 
    const glm::vec3& raydir, 
    const glm::vec4& grot, 
    const glm::vec3& gscl, 
    const float gres, 
    const float gresGrd,
#if ENABLE_DEPTH_GRADIENTS 
    const float gdist,
    const float rayHitGrdWeight, ///< dL_ddistance * weight
#endif
    glm::vec3* __restrict__ dL_dscales,
    glm::vec4* __restrict__ dL_drotations,
    glm::vec3* __restrict__ dL_dmeans3D)
{
    const float r = grot.x;
    const float x = grot.y;
    const float y = grot.z;
    const float z = grot.w;
    // column major !!!
    const glm::mat3 grot_mat = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    const glm::vec3 giscl = 1.f / gscl;

    const glm::vec3& gposc = o_minus_mu;
    const glm::vec3  gposcr = (grot_mat * gposc);
    const glm::vec3  gro = giscl * gposcr;
    const glm::vec3  rayDirR = (grot_mat * raydir);
    const glm::vec3  grdu = giscl * rayDirR;
    const glm::vec3  grd = safe_normalize(grdu);
    const glm::vec3  gcrod = glm::cross(grd, gro);

#if ENABLE_DEPTH_GRADIENTS
    // gradient vs gaussian particle distance (aka gdist)
    const glm::vec3 grdd = grd * glm::dot(grd, -1.0f * gro);
    const glm::vec3 grds = gscl * grdd;
    const glm::vec3 grdsRayHitGrd = gdist > 0.0f ? ((/*2.0f **/ grds) / (/*2.0f **/ gdist)) * rayHitGrdWeight : glm::vec3(0.0f);
    const glm::vec3 gsclRayHitGrd = grdd * grdsRayHitGrd;
    const glm::vec3 grdRayHitGrd = -gscl * glm::vec3(
        2 * grd.x * gro.x + grd.y * gro.y + grd.z * gro.z, 
        grd.x * gro.x + 2 * grd.y * gro.y + grd.z * gro.z, 
        grd.x * gro.x + grd.y * gro.y + 2 * grd.z * gro.z) * grdsRayHitGrd;
    const glm::vec3 groRayHitGrd = -gscl * grd * grd * grdsRayHitGrd;
#else
    const glm::vec3 gsclRayHitGrd = glm::vec3(0.0f);
    const glm::vec3 grdRayHitGrd = glm::vec3(0.0f);
    const glm::vec3 groRayHitGrd = glm::vec3(0.0f);
#endif

/// generalized gaussian of degree b : scaling a = -4.5/3^b
/// d_e^{-a*|x|^b}/d_x^2 = -a*(0.5*b)*x^{b-2}*e^{-a*|x|^b}
#if GENERALIZED_GAUSSIAN_KERNEL == 8 // just because I love the Zenzizenzizenzic kernel
    constexpr float s = -0.000685871056241 * (0.5f * 8);
    const float grayDist = glm::dot(gcrod, gcrod);
    const float grayDistSq = grayDist * grayDist;
    const float grayDistGrd = s * grayDistSq * grayDist * gres * gresGrd;
#elif GENERALIZED_GAUSSIAN_KERNEL == 5 // quintic
    constexpr float s = -0.0185185185185 * (0.5f * 5);
    const float grayDist = glm::dot(gcrod, gcrod);
    const float grayDistGrd = s * grayDist * sqrtf(grayDist) * gres * gresGrd;
#elif GENERALIZED_GAUSSIAN_KERNEL == 4 // tesseractic
    constexpr float s = -0.0555555555556 * (0.5f * 4);
    const float grayDist = glm::dot(gcrod, gcrod);
    const float grayDistGrd = s * grayDist * gres * gresGrd;
#elif GENERALIZED_GAUSSIAN_KERNEL == 3 // cubic
    constexpr float s = -0.166666666667 * (0.5f * 3);
    const float grayDist = glm::dot(gcrod, gcrod);
    const float grayDistGrd =  s * sqrtf(grayDist) * gres * gresGrd;
#else // default to quadratic
    const float grayDistGrd = -0.5f * gres * gresGrd;
#endif
    const auto gcrodGrd = 2.f * gcrod * grayDistGrd;

    const auto grdGrd = glm::vec3(gcrodGrd.z * gro.y - gcrodGrd.y * gro.z,
                                gcrodGrd.x * gro.z - gcrodGrd.z * gro.x,
                                gcrodGrd.y * gro.x - gcrodGrd.x * gro.y);
    const auto groGrd = glm::vec3(gcrodGrd.y * grd.z - gcrodGrd.z * grd.y,
                                gcrodGrd.z * grd.x - gcrodGrd.x * grd.z,
                                gcrodGrd.x * grd.y - gcrodGrd.y * grd.x);
    const auto gsclGrdGro = glm::vec3((-gposcr.x / (gscl.x * gscl.x)),
                                    (-gposcr.y / (gscl.y * gscl.y)),
                                    (-gposcr.z / (gscl.z * gscl.z))) *
                            (groGrd + groRayHitGrd);
    const glm::vec3 gposcrGrd = giscl * (groGrd + groRayHitGrd);
    const glm::vec3 gposcGrd = matmul_bw_vec(grot_mat, gposcrGrd);
    const glm::vec4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);
    const glm::vec3 rayMoGPosGrd = -gposcGrd;

    atomicAdd(&dL_dmeans3D[global_id].x, rayMoGPosGrd.x);
    atomicAdd(&dL_dmeans3D[global_id].y, rayMoGPosGrd.y);
    atomicAdd(&dL_dmeans3D[global_id].z, rayMoGPosGrd.z);

    const auto grduGrd = safe_normalize_bw(grdu, grdGrd + grdRayHitGrd);
    atomicAdd(&dL_dscales[global_id].x, gsclRayHitGrd.x + gsclGrdGro.x + (-rayDirR.x / (gscl.x * gscl.x)) * grduGrd.x);
    atomicAdd(&dL_dscales[global_id].y, gsclRayHitGrd.y + gsclGrdGro.y + (-rayDirR.y / (gscl.y * gscl.y)) * grduGrd.y);
    atomicAdd(&dL_dscales[global_id].z, gsclRayHitGrd.z + gsclGrdGro.z + (-rayDirR.z / (gscl.z * gscl.z)) * grduGrd.z);

    const auto rayDirRGrd = giscl * grduGrd;
    const auto grotGrdRayDirR = matmul_bw_quat(raydir, rayDirRGrd, grot);
    atomicAdd(&dL_drotations[global_id].x, grotGrdPoscr.x + grotGrdRayDirR.x);
    atomicAdd(&dL_drotations[global_id].y, grotGrdPoscr.y + grotGrdRayDirR.y);
    atomicAdd(&dL_drotations[global_id].z, grotGrdPoscr.z + grotGrdRayDirR.z);
    atomicAdd(&dL_drotations[global_id].w, grotGrdPoscr.w + grotGrdRayDirR.w);
}

__forceinline__ __device__ float depth_along_ray_geometric(const glm::vec3& grd, const glm::vec3& gro, const glm::vec3& gscl)
{
    const glm::vec3 grds = gscl * grd * glm::dot(grd, -gro);
    return sqrtf(glm::dot(grds, grds));
}

__forceinline__ __device__ float depth_along_ray_geometric(const glm::vec3& o_minus_mu, const glm::vec3& raydir, const glm::vec4& grot, const glm::vec3& gscl)
{
    const glm::vec3  giscl = 1.f / gscl;

    const float r = grot.x;
    const float x = grot.y;
    const float y = grot.z;
    const float z = grot.w;
    // column major !!!
    const glm::mat3 grot_mat = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    const glm::vec3& gposc = o_minus_mu;
    const glm::vec3 gro = giscl * (grot_mat * gposc);
    const glm::vec3 grd = safe_normalize(giscl * (grot_mat * raydir));

    const glm::vec3 grds = gscl * grd * glm::dot(grd, -gro);
    return sqrtf(glm::dot(grds, grds));;
}

__forceinline__ __device__ glm::vec3 computeViewRay(const glm::mat4 inverse_vp, const glm::vec3 campos, const float2 pix, const int W, const int H)
{
    const glm::vec3 p_world = pix2world(glm::vec2(pix.x, pix.y), W, H, inverse_vp);
    const glm::vec3 viewdir = glm::normalize(p_world - glm::vec3(campos.x, campos.y, campos.z));
    return { viewdir.x, viewdir.y, viewdir.z };
}

__forceinline__ __device__ glm::vec3 colormapMagma(float x)
{
    const glm::vec3 c_magma[] = {
        {-0.002136485053939582f, -0.000749655052795221f, -0.005386127855323933f},
        {0.2516605407371642f, 0.6775232436837668f, 2.494026599312351f},
        {8.353717279216625f, -3.577719514958484f, 0.3144679030132573f},
        {-27.66873308576866f, 14.26473078096533f, -13.64921318813922f},
        {52.17613981234068f, -27.94360607168351f, 12.94416944238394f},
        {-50.76852536473588f, 29.04658282127291f, 4.23415299384598f},
        {18.65570506591883f, -11.48977351997711f, -5.601961508734096f}
    };
    x = glm::clamp(x, 0.f, 1.f);
    glm::vec3 res = (c_magma[0]+x*(c_magma[1]+x*(c_magma[2]+x*(c_magma[3]+x*(c_magma[4]+x*(c_magma[5]+c_magma[6]*x))))));
    return glm::vec3(
        glm::clamp(res[0], 0.f, 1.f),
        glm::clamp(res[1], 0.f, 1.f),
        glm::clamp(res[2], 0.f, 1.f)
    );
}

// supporting the TURBO depth colormap of google (https://blog.research.google/2019/08/turbo-improved-rainbow-colormap-for.html?m=1)
// somewhat adapted from https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
__forceinline__ __device__ glm::vec3 colormapTurbo(float x) {
    float turbo_srgb_floats[256][3] = {{0.18995,0.07176,0.23217},{0.19483,0.08339,0.26149},{0.19956,0.09498,0.29024},{0.20415,0.10652,0.31844},{0.20860,0.11802,0.34607},{0.21291,0.12947,0.37314},{0.21708,0.14087,0.39964},{0.22111,0.15223,0.42558},{0.22500,0.16354,0.45096},{0.22875,0.17481,0.47578},{0.23236,0.18603,0.50004},{0.23582,0.19720,0.52373},{0.23915,0.20833,0.54686},{0.24234,0.21941,0.56942},{0.24539,0.23044,0.59142},{0.24830,0.24143,0.61286},{0.25107,0.25237,0.63374},{0.25369,0.26327,0.65406},{0.25618,0.27412,0.67381},{0.25853,0.28492,0.69300},{0.26074,0.29568,0.71162},{0.26280,0.30639,0.72968},{0.26473,0.31706,0.74718},{0.26652,0.32768,0.76412},{0.26816,0.33825,0.78050},{0.26967,0.34878,0.79631},{0.27103,0.35926,0.81156},{0.27226,0.36970,0.82624},{0.27334,0.38008,0.84037},{0.27429,0.39043,0.85393},{0.27509,0.40072,0.86692},{0.27576,0.41097,0.87936},{0.27628,0.42118,0.89123},{0.27667,0.43134,0.90254},{0.27691,0.44145,0.91328},{0.27701,0.45152,0.92347},{0.27698,0.46153,0.93309},{0.27680,0.47151,0.94214},{0.27648,0.48144,0.95064},{0.27603,0.49132,0.95857},{0.27543,0.50115,0.96594},{0.27469,0.51094,0.97275},{0.27381,0.52069,0.97899},{0.27273,0.53040,0.98461},{0.27106,0.54015,0.98930},{0.26878,0.54995,0.99303},{0.26592,0.55979,0.99583},{0.26252,0.56967,0.99773},{0.25862,0.57958,0.99876},{0.25425,0.58950,0.99896},{0.24946,0.59943,0.99835},{0.24427,0.60937,0.99697},{0.23874,0.61931,0.99485},{0.23288,0.62923,0.99202},{0.22676,0.63913,0.98851},{0.22039,0.64901,0.98436},{0.21382,0.65886,0.97959},{0.20708,0.66866,0.97423},{0.20021,0.67842,0.96833},{0.19326,0.68812,0.96190},{0.18625,0.69775,0.95498},{0.17923,0.70732,0.94761},{0.17223,0.71680,0.93981},{0.16529,0.72620,0.93161},{0.15844,0.73551,0.92305},{0.15173,0.74472,0.91416},{0.14519,0.75381,0.90496},{0.13886,0.76279,0.89550},{0.13278,0.77165,0.88580},{0.12698,0.78037,0.87590},{0.12151,0.78896,0.86581},{0.11639,0.79740,0.85559},{0.11167,0.80569,0.84525},{0.10738,0.81381,0.83484},{0.10357,0.82177,0.82437},{0.10026,0.82955,0.81389},{0.09750,0.83714,0.80342},{0.09532,0.84455,0.79299},{0.09377,0.85175,0.78264},{0.09287,0.85875,0.77240},{0.09267,0.86554,0.76230},{0.09320,0.87211,0.75237},{0.09451,0.87844,0.74265},{0.09662,0.88454,0.73316},{0.09958,0.89040,0.72393},{0.10342,0.89600,0.71500},{0.10815,0.90142,0.70599},{0.11374,0.90673,0.69651},{0.12014,0.91193,0.68660},{0.12733,0.91701,0.67627},{0.13526,0.92197,0.66556},{0.14391,0.92680,0.65448},{0.15323,0.93151,0.64308},{0.16319,0.93609,0.63137},{0.17377,0.94053,0.61938},{0.18491,0.94484,0.60713},{0.19659,0.94901,0.59466},{0.20877,0.95304,0.58199},{0.22142,0.95692,0.56914},{0.23449,0.96065,0.55614},{0.24797,0.96423,0.54303},{0.26180,0.96765,0.52981},{0.27597,0.97092,0.51653},{0.29042,0.97403,0.50321},{0.30513,0.97697,0.48987},{0.32006,0.97974,0.47654},{0.33517,0.98234,0.46325},{0.35043,0.98477,0.45002},{0.36581,0.98702,0.43688},{0.38127,0.98909,0.42386},{0.39678,0.99098,0.41098},{0.41229,0.99268,0.39826},{0.42778,0.99419,0.38575},{0.44321,0.99551,0.37345},{0.45854,0.99663,0.36140},{0.47375,0.99755,0.34963},{0.48879,0.99828,0.33816},{0.50362,0.99879,0.32701},{0.51822,0.99910,0.31622},{0.53255,0.99919,0.30581},{0.54658,0.99907,0.29581},{0.56026,0.99873,0.28623},{0.57357,0.99817,0.27712},{0.58646,0.99739,0.26849},{0.59891,0.99638,0.26038},{0.61088,0.99514,0.25280},{0.62233,0.99366,0.24579},{0.63323,0.99195,0.23937},{0.64362,0.98999,0.23356},{0.65394,0.98775,0.22835},{0.66428,0.98524,0.22370},{0.67462,0.98246,0.21960},{0.68494,0.97941,0.21602},{0.69525,0.97610,0.21294},{0.70553,0.97255,0.21032},{0.71577,0.96875,0.20815},{0.72596,0.96470,0.20640},{0.73610,0.96043,0.20504},{0.74617,0.95593,0.20406},{0.75617,0.95121,0.20343},{0.76608,0.94627,0.20311},{0.77591,0.94113,0.20310},{0.78563,0.93579,0.20336},{0.79524,0.93025,0.20386},{0.80473,0.92452,0.20459},{0.81410,0.91861,0.20552},{0.82333,0.91253,0.20663},{0.83241,0.90627,0.20788},{0.84133,0.89986,0.20926},{0.85010,0.89328,0.21074},{0.85868,0.88655,0.21230},{0.86709,0.87968,0.21391},{0.87530,0.87267,0.21555},{0.88331,0.86553,0.21719},{0.89112,0.85826,0.21880},{0.89870,0.85087,0.22038},{0.90605,0.84337,0.22188},{0.91317,0.83576,0.22328},{0.92004,0.82806,0.22456},{0.92666,0.82025,0.22570},{0.93301,0.81236,0.22667},{0.93909,0.80439,0.22744},{0.94489,0.79634,0.22800},{0.95039,0.78823,0.22831},{0.95560,0.78005,0.22836},{0.96049,0.77181,0.22811},{0.96507,0.76352,0.22754},{0.96931,0.75519,0.22663},{0.97323,0.74682,0.22536},{0.97679,0.73842,0.22369},{0.98000,0.73000,0.22161},{0.98289,0.72140,0.21918},{0.98549,0.71250,0.21650},{0.98781,0.70330,0.21358},{0.98986,0.69382,0.21043},{0.99163,0.68408,0.20706},{0.99314,0.67408,0.20348},{0.99438,0.66386,0.19971},{0.99535,0.65341,0.19577},{0.99607,0.64277,0.19165},{0.99654,0.63193,0.18738},{0.99675,0.62093,0.18297},{0.99672,0.60977,0.17842},{0.99644,0.59846,0.17376},{0.99593,0.58703,0.16899},{0.99517,0.57549,0.16412},{0.99419,0.56386,0.15918},{0.99297,0.55214,0.15417},{0.99153,0.54036,0.14910},{0.98987,0.52854,0.14398},{0.98799,0.51667,0.13883},{0.98590,0.50479,0.13367},{0.98360,0.49291,0.12849},{0.98108,0.48104,0.12332},{0.97837,0.46920,0.11817},{0.97545,0.45740,0.11305},{0.97234,0.44565,0.10797},{0.96904,0.43399,0.10294},{0.96555,0.42241,0.09798},{0.96187,0.41093,0.09310},{0.95801,0.39958,0.08831},{0.95398,0.38836,0.08362},{0.94977,0.37729,0.07905},{0.94538,0.36638,0.07461},{0.94084,0.35566,0.07031},{0.93612,0.34513,0.06616},{0.93125,0.33482,0.06218},{0.92623,0.32473,0.05837},{0.92105,0.31489,0.05475},{0.91572,0.30530,0.05134},{0.91024,0.29599,0.04814},{0.90463,0.28696,0.04516},{0.89888,0.27824,0.04243},{0.89298,0.26981,0.03993},{0.88691,0.26152,0.03753},{0.88066,0.25334,0.03521},{0.87422,0.24526,0.03297},{0.86760,0.23730,0.03082},{0.86079,0.22945,0.02875},{0.85380,0.22170,0.02677},{0.84662,0.21407,0.02487},{0.83926,0.20654,0.02305},{0.83172,0.19912,0.02131},{0.82399,0.19182,0.01966},{0.81608,0.18462,0.01809},{0.80799,0.17753,0.01660},{0.79971,0.17055,0.01520},{0.79125,0.16368,0.01387},{0.78260,0.15693,0.01264},{0.77377,0.15028,0.01148},{0.76476,0.14374,0.01041},{0.75556,0.13731,0.00942},{0.74617,0.13098,0.00851},{0.73661,0.12477,0.00769},{0.72686,0.11867,0.00695},{0.71692,0.11268,0.00629},{0.70680,0.10680,0.00571},{0.69650,0.10102,0.00522},{0.68602,0.09536,0.00481},{0.67535,0.08980,0.00449},{0.66449,0.08436,0.00424},{0.65345,0.07902,0.00408},{0.64223,0.07380,0.00401},{0.63082,0.06868,0.00401},{0.61923,0.06367,0.00410},{0.60746,0.05878,0.00427},{0.59550,0.05399,0.00453},{0.58336,0.04931,0.00486},{0.57103,0.04474,0.00529},{0.55852,0.04028,0.00579},{0.54583,0.03593,0.00638},{0.53295,0.03169,0.00705},{0.51989,0.02756,0.00780},{0.50664,0.02354,0.00863},{0.49321,0.01963,0.00955},{0.47960,0.01583,0.01055}};
    
    float interp = glm::clamp(x * 255.f, 0.f, 255.f);
    int floor = x > 0 ? (int)interp : 0;
    int ceil = floor >= 255 ? 255 : floor + 1;
    float diff = interp - floor;

    return glm::vec3(
        glm::clamp(turbo_srgb_floats[floor][0] + (turbo_srgb_floats[ceil][0] - turbo_srgb_floats[floor][0]) * diff, 0.f, 1.f),
        glm::clamp(turbo_srgb_floats[floor][1] + (turbo_srgb_floats[ceil][1] - turbo_srgb_floats[floor][1]) * diff, 0.f, 1.f),
        glm::clamp(turbo_srgb_floats[floor][2] + (turbo_srgb_floats[ceil][2] - turbo_srgb_floats[floor][2]) * diff, 0.f, 1.f)
    );
}

#ifdef __CUDACC__

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeysCUDA(
    const dim3 grid, const int P,
    const glm::vec2* __restrict__ rects,
    const glm::vec2* __restrict__ points_xy,
    const float*     __restrict__ depths,
    const uint32_t*  __restrict__ offsets,
    const int*       __restrict__ radii,
    uint64_t* __restrict__ gaussian_keys_unsorted,
    uint32_t* __restrict__ gaussian_values_unsorted);

#endif

#endif
