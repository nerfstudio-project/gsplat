#include "config.h"
#include "cuda_runtime.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


inline __device__ float ndc2pix(const float x, const float W) {
    return 0.5 * ((1.f + x) * W - 1.f);
}

inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    uint2& bb_min,
    uint2& bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int) (center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int) (center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int) (center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int) (center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(
    const float2 pix_center,
    const float pix_radius,
    const dim3 tile_bounds,
    uint2 &tile_min,
    uint2 &tile_max
) {
    float2 tile_center = {pix_center.x / (float) BLOCK_X, pix_center.y / (float) BLOCK_Y};
    float2 tile_radius = {pix_radius / (float) BLOCK_X, pix_radius / (float) BLOCK_Y};
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

inline __device__ bool compute_cov2d_bounds(const float3 cov2d, float3 &conic, float& radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

	float b = 0.5f * (cov2d.x + cov2d.z);
	float v1 = b + sqrt(max(0.1f, b * b - det));
	float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
	radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

// compute vjp from df/d_conic to df/c_cov2d
inline __device__ bool cov2d_to_conic_vjp(const float3 conic, const float3 v_conic, float3 &v_cov2d)
{
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline __device__ float3 transform_4x3(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline __device__ float4 transform_4x4(const float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.w * s;
    float x = quat.x * s;
    float y = quat.y * s;
    float z = quat.z * s;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y)
    );
}

inline __device__ float4 quat_to_rotmat_vjp(const float4 quat, const glm::mat3 v_Rmat) {
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.w * s;
    float x = quat.x * s;
    float y = quat.y * s;
    float z = quat.z * s;

    float4 v_quat;
    float *v_R = (float *) &v_Rmat[0][0];
    v_quat.w = (
        2.f * z * (v_R[1] - v_R[3])
        + 2.f * y * (v_R[6] - v_R[2])
        + 2.f * x * (v_R[5] - v_R[7])
    );
    v_quat.x = (
        2.f * y * (v_R[1] + v_R[3])
        + 2.f * z * (v_R[6] - v_R[2])
        + 2.f * w * (v_R[5] - v_R[7])
        - 4.f * x * (v_R[4] + v_R[8])
    );
    v_quat.y = (
        -4.f * y * (v_R[0] + v_R[8])
        + 2.f * x * (v_R[1] + v_R[3])
        + 2.f * z * (v_R[5] + v_R[7])
        + 2.f * w * (v_R[6] - v_R[2])
    );
    v_quat.z = (
        -4.f * z * (v_R[0] + v_R[4])
        + 2.f * x * (v_R[6] + v_R[2])
        + 2.f * y * (v_R[5] + v_R[7])
        + 2.f * w * (v_R[1] - v_R[3])
    );
    return v_quat;
}

inline __device__ glm::mat3 scale_to_mat(const float3 scale, const float glob_scale) {
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// device helper for culling near points
inline __device__ bool clip_near_plane(
    const float3 p, const float *viewmat, float3& p_view
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= 0.1f) {
        return true;
    }
    return false;
}
