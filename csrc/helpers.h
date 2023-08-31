#include "cuda_runtime.h"

inline __device__ float ndc2pix(float x, float W) {
    return 0.5 * ((1.f + x) * W - 1.f);
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
inline __device__ bool in_frustum(
    const float3 p,
    const float *viewmat,
    const float *projmat,
    const float tan_fovx,
    const float tan_fovy,
    float3& p_view,
    float3& p_proj
) {
    float4 p_homo = transform_4x4(projmat, p);
    float p_w = 1.f / (p_homo.w + 1e-5f);
    p_proj = {p_homo.x * p_w, p_homo.y * p_w, p_homo.z * p_w};
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= 0.1f) {
        return false;
    }

    const float xlim = 1.5f * tan_fovx;
    const float ylim = 1.5f * tan_fovy;
    if ((p_proj.x < -xlim) || (p_proj.x > xlim) || (p_proj.y < -ylim) || (p_proj.y > ylim)) {
        return false;
    }
    return true;
}

