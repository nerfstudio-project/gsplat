#include "cuda_runtime.h"

#define BLOCK_X 16
#define BLOCK_Y 16

inline __device__ float ndc2pix(float x, float W) {
    return 0.5 * ((1.f + x) * W - 1.f);
}

inline __device__ void get_bbox(
    const float2 center,
    const float radius,
    const dim3 tile_bounds,
    uint2& bb_min,
    uint2& bb_max
) {
    // get bounding box with center and radius, within bounds
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)((center.x - radius) / BLOCK_X)), tile_bounds.x);
    bb_max.x = min(max(0, (int)((center.x + radius + BLOCK_X - 1) / BLOCK_X)), tile_bounds.x);
    bb_min.y = min(max(0, (int)((center.y - radius) / BLOCK_Y)), tile_bounds.y);
    bb_max.y = min(max(0, (int)((center.y + radius + BLOCK_Y - 1) / BLOCK_Y)), tile_bounds.y);
}

inline __device__ bool compute_cov2d_bounds(float3 cov2d, float3 &conic, float& radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
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
	radius = 3.f * sqrt(max(v1, v2));
    return true;
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

