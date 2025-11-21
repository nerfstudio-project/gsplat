#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define CHANNELS 3
#define MAX_REGISTER_CHANNELS 3

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
constant float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

inline uint num_sh_bases(const uint degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

inline float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5;
}

inline void get_bbox(
    const float2 center,
    const float2 dims,
    const int3 img_size,
    thread uint2 &bb_min,
    thread uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline void get_tile_bbox(
    const float2 pix_center,
    const float pix_radius,
    const int3 tile_bounds,
    thread uint2 &tile_min,
    thread uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius / (float)BLOCK_X, pix_radius / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline float3 transform_4x3(constant float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline float4 transform_4x4(constant float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline float3x3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float s = rsqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    // metal matrices are column-major
    return float3x3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// device helper for culling near points
inline bool clip_near_plane(
    const float3 p,
    constant float *viewmat,
    thread float3 &p_view,
    float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

inline float3x3 scale_to_mat(const float3 scale, const float glob_scale) {
    float3x3 S = float3x3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// device helper to get 3D covariance from scale and quat parameters
inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, device float *cov3d
) {
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.x, quat.y, quat.z, quat.w);
    float3x3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    float3x3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// device helper to approximate projected 2d cov from 3d mean and cov
float3 project_cov3d_ewa(
    thread float3& mean3d,
    device float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy
) {
    // clip the
    // we expect row major matrices as input, metal uses column major
    // upper 3x3 submatrix
    float3x3 W = float3x3(
        viewmat[0],
        viewmat[4],
        viewmat[8],
        viewmat[1],
        viewmat[5],
        viewmat[9],
        viewmat[2],
        viewmat[6],
        viewmat[10]
    );
    float3 p = float3(viewmat[3], viewmat[7], viewmat[11]);
    float3 t = W * float3(mean3d.x, mean3d.y, mean3d.z) + p;

    // clip so that the covariance
    float lim_x = 1.3 * tan_fovx;
    float lim_y = 1.3 * tan_fovy;
    t.x = t.z * min(lim_x, max(-lim_x, t.x / t.z));
    t.y = t.z * min(lim_y, max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    float3x3 J = float3x3(
        fx * rz,
        0.f,
        0.f,
        0.f,
        fy * rz,
        0.f,
        -fx * t.x * rz2,
        -fy * t.y * rz2,
        0.f
    );
    float3x3 T = J * W;

    float3x3 V = float3x3(
        cov3d[0],
        cov3d[1],
        cov3d[2],
        cov3d[1],
        cov3d[3],
        cov3d[4],
        cov3d[2],
        cov3d[4],
        cov3d[5]
    );

    float3x3 cov = T * V * transpose(T);

    // add a little blur along axes and save upper triangular elements
    return float3(float(cov[0][0]) + 0.3f, float(cov[0][1]), float(cov[1][1]) + 0.3f);
}

inline bool compute_cov2d_bounds(
    const float3 cov2d,
    thread float3 &conic,
    thread float &radius
) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
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

inline float2 project_pix(
    constant float *mat, const float3 p, const uint2 img_size, const float2 pp
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, (int)img_size.x, pp.x), ndc2pix(p_proj.y, (int)img_size.y, pp.y)
    };
}

/*
    !!!!IMPORTANT!!!
    Metal does not support packed arrays of vectorized types like int2, float2, float3, etc.
    and instead pads the elements of arrays of these types to fixed alignments.
    Use the below functions to read and write from packed arrays of these types.
*/

inline int2 read_packed_int2(constant int* arr, int idx) {
    return int2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_int2(device int* arr, int idx, int2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline void write_packed_int2x(device int* arr, int idx, int x) {
    arr[2*idx] = x;
}

inline void write_packed_int2y(device int* arr, int idx, int y) {
    arr[2*idx+1] = y;
}

inline float2 read_packed_float2(constant float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline float2 read_packed_float2(device float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_float2(device float* arr, int idx, float2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline int3 read_packed_int3(constant int* arr, int idx) {
    return int3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_int3(device int* arr, int idx, int3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float3 read_packed_float3(constant float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline float3 read_packed_float3(device float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_float3(device float* arr, int idx, float3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float4 read_packed_float4(constant float* arr, int idx) {
    return float4(arr[4*idx], arr[4*idx+1], arr[4*idx+2], arr[4*idx+3]);
}

inline void write_packed_float4(device float* arr, int idx, float4 val) {
    arr[4*idx] = val.x;
    arr[4*idx+1] = val.y;
    arr[4*idx+2] = val.z;
    arr[4*idx+3] = val.w;
}

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
kernel void project_gaussians_forward_kernel(
    constant int& num_points,
    constant float* means3d, // float3
    constant float* scales, // float3
    constant float& glob_scale,
    constant float* quats, // float4
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant uint3& tile_bounds,
    constant float& clip_thresh,
    device float* covs3d,
    device float* xys, // float2
    device float* depths,
    device int* radii,
    device float* conics, // float3
    device int32_t* num_tiles_hit,
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = read_packed_float3(means3d, idx);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        return;
    }

    // compute the projected covariance
    float3 scale = read_packed_float3(scales, idx);
    float4 quat = read_packed_float4(quats, idx);
    device float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        p_world, cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy
    );

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) {
        return; // zero determinant
    }
    write_packed_float3(conics, idx, conic);

    // compute the projected mean
    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, (int3)tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    write_packed_float2(xys, idx, center);
}

kernel void nd_rasterize_forward_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins, // int2
    constant float* xys, // float2
    constant float* conics, // float3
    constant float* colors,
    constant float* opacities,
    device float* final_Ts,
    device int* final_index,
    device float* out_img,
    constant float* background,
    constant uint2& blockDim,
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 threadIdx [[thread_position_in_threadgroup]]
) {
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * (int)img_size.x + j;

    // return if out of bounds
    if (i >= (int)img_size.y || j >= (int)img_size.x) {
        return;
    }

    // which gaussians to look through in this tile
    int2 range = read_packed_int2(tile_bins, tile_id);
    float T = 1.f;

    // iterate over all gaussians and apply rendering EWA equation (e.q. 2 from
    // paper)
    int idx;
    for (idx = range.x; idx < range.y; ++idx) {
        const int32_t g = gaussian_ids_sorted[idx];
        const float3 conic = read_packed_float3(conics, g);
        const float2 center = read_packed_float2(xys, g);
        const float2 delta = {center.x - px, center.y - py};

        // Mahalanobis distance (here referred to as sigma) measures how many
        // standard deviations away distance delta is. sigma = -0.5(d.T * conic
        // * d)
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];

        const float alpha = min(0.999f, opac * exp(-sigma));

        // break out conditions
        if (alpha < 1.f / 255.f) {
            continue;
        }
        const float next_T = T * (1.f - alpha);
        if (next_T <= 1e-4f) {
            // we want to render the last gaussian that contributes and note
            // that here idx > range.x so we don't underflow
            idx -= 1;
            break;
        }
        const float vis = alpha * T;
        for (int c = 0; c < channels; ++c) {
            out_img[channels * pix_id + c] += colors[channels * g + c] * vis;
        }
        T = next_T;
    }
    final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
    final_index[pix_id] =
        (idx == range.y)
            ? idx - 1
            : idx; // index of in bin of last gaussian in this pixel
    for (int c = 0; c < channels; ++c) {
        out_img[channels * pix_id + c] += T * background[c];
    }
}

void sh_coeffs_to_color(
    const uint degree,
    const float3 viewdir,
    constant float *coeffs,
    device float *colors
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    // expects CHANNELS * num_bases coefficients
    // supports up to num_bases = 25
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * coeffs[1 * CHANNELS + c] +
                              z * coeffs[2 * CHANNELS + c] -
                              x * coeffs[3 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * coeffs[4 * CHANNELS + c] +
             SH_C2[1] * yz * coeffs[5 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * coeffs[6 * CHANNELS + c] +
             SH_C2[3] * xz * coeffs[7 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * coeffs[8 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * coeffs[9 * CHANNELS + c] +
             SH_C3[1] * xy * z * coeffs[10 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * coeffs[11 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 coeffs[12 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * coeffs[13 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * coeffs[14 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * coeffs[15 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * coeffs[16 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * coeffs[17 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * coeffs[18 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * coeffs[19 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 coeffs[20 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * coeffs[21 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 coeffs[22 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * coeffs[23 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 coeffs[24 * CHANNELS + c]);
    }
}

void sh_coeffs_to_color_vjp(
    const uint degree,
    const float3 viewdir,
    constant float *v_colors,
    device float *v_coeffs
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[c] = SH_C0 * v_colors[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;

    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_coeffs[1 * CHANNELS + c] = v1 * v_colors[c];
        v_coeffs[2 * CHANNELS + c] = v2 * v_colors[c];
        v_coeffs[3 * CHANNELS + c] = v3 * v_colors[c];
        if (degree < 2) {
            continue;
        }
        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * zz - xx - yy);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (xx - yy);
        v_coeffs[4 * CHANNELS + c] = v4 * v_colors[c];
        v_coeffs[5 * CHANNELS + c] = v5 * v_colors[c];
        v_coeffs[6 * CHANNELS + c] = v6 * v_colors[c];
        v_coeffs[7 * CHANNELS + c] = v7 * v_colors[c];
        v_coeffs[8 * CHANNELS + c] = v8 * v_colors[c];
        if (degree < 3) {
            continue;
        }
        float v9 = SH_C3[0] * y * (3.f * xx - yy);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float v12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float v13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float v14 = SH_C3[5] * z * (xx - yy);
        float v15 = SH_C3[6] * x * (xx - 3.f * yy);
        v_coeffs[9 * CHANNELS + c] = v9 * v_colors[c];
        v_coeffs[10 * CHANNELS + c] = v10 * v_colors[c];
        v_coeffs[11 * CHANNELS + c] = v11 * v_colors[c];
        v_coeffs[12 * CHANNELS + c] = v12 * v_colors[c];
        v_coeffs[13 * CHANNELS + c] = v13 * v_colors[c];
        v_coeffs[14 * CHANNELS + c] = v14 * v_colors[c];
        v_coeffs[15 * CHANNELS + c] = v15 * v_colors[c];
        if (degree < 4) {
            continue;
        }
        float v16 = SH_C4[0] * xy * (xx - yy);
        float v17 = SH_C4[1] * yz * (3.f * xx - yy);
        float v18 = SH_C4[2] * xy * (7.f * zz - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * zz - 3.f);
        float v20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * zz - 3.f);
        float v22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
        float v23 = SH_C4[7] * xz * (xx - 3.f * yy);
        float v24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
        v_coeffs[16 * CHANNELS + c] = v16 * v_colors[c];
        v_coeffs[17 * CHANNELS + c] = v17 * v_colors[c];
        v_coeffs[18 * CHANNELS + c] = v18 * v_colors[c];
        v_coeffs[19 * CHANNELS + c] = v19 * v_colors[c];
        v_coeffs[20 * CHANNELS + c] = v20 * v_colors[c];
        v_coeffs[21 * CHANNELS + c] = v21 * v_colors[c];
        v_coeffs[22 * CHANNELS + c] = v22 * v_colors[c];
        v_coeffs[23 * CHANNELS + c] = v23 * v_colors[c];
        v_coeffs[24 * CHANNELS + c] = v24 * v_colors[c];
    }
}

kernel void compute_sh_forward_kernel(
    constant uint& num_points,
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float* viewdirs, // float3
    constant float* coeffs,
    device float* colors,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) {
        return;
    }
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint idx_sh = num_bases * num_channels * idx;
    uint idx_col = num_channels * idx;

    sh_coeffs_to_color(
        degrees_to_use, read_packed_float3(viewdirs, idx), &(coeffs[idx_sh]), &(colors[idx_col])
    );
}

kernel void compute_sh_backward_kernel(
    constant uint& num_points,
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float* viewdirs, // float3
    constant float* v_colors,
    device float* v_coeffs,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) {
        return;
    }
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint idx_sh = num_bases * num_channels * idx;
    uint idx_col = num_channels * idx;

    sh_coeffs_to_color_vjp(
        degrees_to_use, read_packed_float3(viewdirs, idx), &(v_colors[idx_col]), &(v_coeffs[idx_sh])
    );
}

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
kernel void map_gaussian_to_intersects_kernel(
    constant int& num_points,
    constant float* xys, // float2
    constant float* depths,
    constant int* radii,
    constant int32_t* num_tiles_hit,
    constant uint3& tile_bounds,
    device int64_t* isect_ids,
    device int32_t* gaussian_ids,
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the tile bbox for gaussian
    uint2 tile_min, tile_max;
    float2 center = read_packed_float2(xys, idx);
    get_tile_bbox(center, radii[idx], (int3)tile_bounds, tile_min, tile_max);
    // printf("point %d, %d radius, min %d %d, max %d %d\n", idx, radii[idx],
    // tile_min.x, tile_min.y, tile_max.x, tile_max.y);

    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : num_tiles_hit[idx - 1];
    // printf("point %d starting at %d\n", idx, cur_idx);
    int64_t depth_id = (int64_t) * (constant int32_t *)&(depths[idx]);
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            // isect_id is tile ID and depth as int32
            int64_t tile_id = i * tile_bounds.x + j; // tile within image
            isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
    // printf("point %d ending at %d\n", idx, cur_idx);
}

// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
kernel void get_tile_bin_edges_kernel(
    constant int& num_intersects,
    constant int64_t* isect_ids_sorted,
    device int* tile_bins, // int2
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            write_packed_int2x(tile_bins, cur_tile_idx, 0);
        if (idx == num_intersects - 1)
            write_packed_int2y(tile_bins, cur_tile_idx, num_intersects);
        return;
    }
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        write_packed_int2y(tile_bins, prev_tile_idx, idx);
        write_packed_int2x(tile_bins, cur_tile_idx, idx);
        return;
    }
}

inline int warp_reduce_all_max(int val, const int warp_size) {
    // This uses an xor so that all threads in a warp get the same result
    for ( int mask = warp_size / 2; mask > 0; mask /= 2 )
        val = max(val, simd_shuffle_xor(val, mask));

    return val;
}

inline int warp_reduce_all_or(int val, const int warp_size) {
    // This uses an xor so that all threads in a warp get the same result
    for ( int mask = warp_size / 2; mask > 0; mask /= 2 )
        val = val | simd_shuffle_xor(val, mask);

    return val;
}

inline float warp_reduce_sum(float val, const int warp_size) {
    for ( int offset = warp_size / 2; offset > 0; offset /= 2 )
        val += simd_shuffle_and_fill_down(val, 0., offset);

    return val;
}

inline float3 warpSum3(float3 val, uint warp_size){
    val.x = warp_reduce_sum(val.x, warp_size);
    val.y = warp_reduce_sum(val.y, warp_size);
    val.z = warp_reduce_sum(val.z, warp_size);
    return val;
}

inline float2 warpSum2(float2 val, uint warp_size){
    val.x = warp_reduce_sum(val.x, warp_size);
    val.y = warp_reduce_sum(val.y, warp_size);
    return val;
}

inline float warpSum(float val, uint warp_size){
    val = warp_reduce_sum(val, warp_size);
    return val;
}

kernel void rasterize_backward_kernel(
    constant uint3& tile_bounds,
    constant uint2& img_size,
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins, // int2
    constant float* xys, // float2
    constant float* conics, // float3
    constant float* rgbs, // float3
    constant float* opacities,
    constant float* background, // single float3
    constant float* final_Ts,
    constant int* final_index,
    constant float* v_output, // float3
    constant float* v_output_alpha,
    device atomic_float* v_xy, // float2
    device atomic_float* v_conic, // float3
    device atomic_float* v_rgb, // float3
    device atomic_float* v_opacity,
    uint3 gp [[thread_position_in_grid]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint tr [[thread_index_in_threadgroup]],
    uint warp_size [[threads_per_simdgroup]],
    uint wr [[thread_index_in_simdgroup]]
) {
    int32_t tile_id =
        blockIdx.y * tile_bounds.x + blockIdx.x;
    uint i = gp.y;
    uint j = gp.x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min((int32_t)(i * img_size.x + j), (int32_t)(img_size.x * img_size.y - 1));

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = read_packed_int2(tile_bins, tile_id);
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    threadgroup int32_t id_batch[BLOCK_SIZE];
    threadgroup float3 xy_opacity_batch[BLOCK_SIZE];
    threadgroup float3 conic_batch[BLOCK_SIZE];
    threadgroup float3 rgbs_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = read_packed_float3(v_output, pix_id);
    const float v_out_alpha = v_output_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int warp_bin_final = warp_reduce_all_max(bin_final, warp_size);
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = read_packed_float2(xys, g_id);
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = read_packed_float3(conics, g_id);
            rgbs_batch[tr] = read_packed_float3(rgbs, g_id);
        }
        // wait for other threads to collect the gaussians in batch
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float3 conic;
            float vis;
            if(valid){
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                vis = exp(-sigma);
                alpha = min(0.99f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if (!warp_reduce_all_or(valid, warp_size)) {
                continue;
            }

            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += (rgb.x * T - buffer.x * ra) * v_out.x;
                v_alpha += (rgb.y * T - buffer.y * ra) * v_out.y;
                v_alpha += (rgb.z * T - buffer.z * ra) * v_out.z;

                v_alpha += T_final * ra * v_out_alpha;
                // contribution from background pixel
                v_alpha += -T_final * ra * background[0] * v_out.x;
                v_alpha += -T_final * ra * background[1] * v_out.y;
                v_alpha += -T_final * ra * background[2] * v_out.z;
                // update the running sum
                buffer.x += rgb.x * fac;
                buffer.y += rgb.y * fac;
                buffer.z += rgb.z * fac;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                        0.5f * v_sigma * delta.x * delta.y,
                                        0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_opacity_local = vis * v_alpha;
            }

            v_rgb_local = warpSum3(v_rgb_local, warp_size);
            v_conic_local = warpSum3(v_conic_local, warp_size);
            v_xy_local = warpSum2(v_xy_local, warp_size);
            v_opacity_local = warpSum(v_opacity_local, warp_size);

            if (wr == 0) {
                int32_t g = id_batch[t];

                atomic_fetch_add_explicit(v_rgb + 3*g + 0, v_rgb_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_rgb + 3*g + 1, v_rgb_local.y, memory_order_relaxed);
                atomic_fetch_add_explicit(v_rgb + 3*g + 2, v_rgb_local.z, memory_order_relaxed);

                atomic_fetch_add_explicit(v_conic + 3*g + 0, v_conic_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*g + 1, v_conic_local.y, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*g + 2, v_conic_local.z, memory_order_relaxed);

                atomic_fetch_add_explicit(v_xy + 2*g + 0, v_xy_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_xy + 2*g + 1, v_xy_local.y, memory_order_relaxed);

                atomic_fetch_add_explicit(v_opacity + g, v_opacity_local, memory_order_relaxed);
            }
        }
    }
}

kernel void nd_rasterize_backward_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    constant int32_t* gaussians_ids_sorted,
    constant int* tile_bins, // int2
    constant float* xys, // float2
    constant float* conics, // float3
    constant float* rgbs,
    constant float* opacities,
    constant float* background,
    constant float* final_Ts,
    constant int* final_index,
    constant float* v_output,
    constant float* v_output_alpha,
    device atomic_float* v_xy, // float2
    device atomic_float* v_conic, // float3
    device atomic_float* v_rgb,
    device atomic_float* v_opacity,
    device float* workspace,
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 threadIdx [[thread_position_in_threadgroup]]
) {
    if (channels > MAX_REGISTER_CHANNELS && workspace == nullptr) {
        return;
    }
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    if (i >= img_size.y || j >= img_size.x) {
        return;
    }

    // which gaussians get gradients for this pixel
    int2 range = read_packed_int2(tile_bins, tile_id);
    // df/d_out for this pixel
    constant float *v_out = &(v_output[channels * pix_id]);
    const float v_out_alpha = v_output_alpha[pix_id];
    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    device float *S = &workspace[channels * pix_id];
    int bin_final = final_index[pix_id];

    // iterate backward to compute the jacobians wrt rgb, opacity, mean2d, and
    // conic recursively compute T_{n-1} from T_n, where T_i = prod(j < i) (1 -
    // alpha_j), and S_{n-1} from S_n, where S_j = sum_{i > j}(rgb_i * alpha_i *
    // T_i) df/dalpha_i = rgb_i * T_i - S_{i+1| / (1 - alpha_i)
    for (int idx = bin_final - 1; idx >= range.x; --idx) {
        const int32_t g = gaussians_ids_sorted[idx];
        const float3 conic = read_packed_float3(conics, g);
        const float2 center = read_packed_float2(xys, g);
        const float2 delta = {center.x - px, center.y - py};
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];
        const float vis = exp(-sigma);
        const float alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }

        // compute the current T for this gaussian
        const float ra = 1.f / (1.f - alpha);
        T *= ra;
        // rgb = rgbs[g];
        // update v_rgb for this gaussian
        const float fac = alpha * T;
        float v_alpha = 0.f;
        for (int c = 0; c < channels; ++c) {
            // gradient wrt rgb
            atomic_fetch_add_explicit(v_rgb + channels * g + c, fac * v_out[c], memory_order_relaxed);
            // contribution from this pixel
            v_alpha += (rgbs[channels * g + c] * T - S[c] * ra) * v_out[c];
            // contribution from background pixel
            v_alpha += -T_final * ra * background[c] * v_out[c];
            // update the running sum
            S[c] += rgbs[channels * g + c] * fac;
        }
        v_alpha += T_final * ra * v_out_alpha;
        // update v_opacity for this gaussian
        atomic_fetch_add_explicit(v_opacity + g, vis * v_alpha, memory_order_relaxed);

        // compute vjps for conics and means
        // d_sigma / d_delta = conic * delta
        // d_sigma / d_conic = delta * delta.T
        const float v_sigma = -opac * vis * v_alpha;

        atomic_fetch_add_explicit(v_conic + 3*g + 0, 0.5f * v_sigma * delta.x * delta.x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_conic + 3*g + 1, 0.5f * v_sigma * delta.x * delta.y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_conic + 3*g + 2, 0.5f * v_sigma * delta.y * delta.y, memory_order_relaxed);
        atomic_fetch_add_explicit(
            v_xy + 2*g + 0, v_sigma * (conic.x * delta.x + conic.y * delta.y), memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            v_xy + 2*g + 1, v_sigma * (conic.y * delta.x + conic.z * delta.y), memory_order_relaxed
        );
    }
}

// given v_xy_pix, get v_xyz
inline float3 project_pix_vjp(
    constant float *mat, const float3 p, const uint2 img_size, const float2 v_xy
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);

    float3 v_ndc = {0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y, 0.0f};
    float4 v_proj = {
        v_ndc.x * rw, v_ndc.y * rw, 0., -(v_ndc.x + v_ndc.y) * rw * rw
    };
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    return {
        mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z,
        mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z,
        mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z
    };
}

// compute vjp from df/d_conic to df/c_cov2d
inline void cov2d_to_conic_vjp(
    float3 conic,
    float3 v_conic,
    device float* v_cov2d // float3
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    float2x2 X = float2x2(conic.x, conic.y, conic.y, conic.z);
    float2x2 G = float2x2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1. * X * G * X;
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

// output space: 2D covariance, input space: cov3d
void project_cov3d_ewa_vjp(
    const float3 mean3d,
    constant float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    float3 v_cov2d,
    device float* v_mean3d, // float3
    device float* v_cov3d
) {
    // viewmat is row major, float3x3 is column major
    // upper 3x3 submatrix
    // clang-format off
    float3x3 W = float3x3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    // clang-format on
    float3 p = float3(viewmat[3], viewmat[7], viewmat[11]);
    float3 t = W * float3(mean3d.x, mean3d.y, mean3d.z) + p;
    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    // clang-format off
    float3x3 J = float3x3(
        fx * rz,         0.f,             0.f,
        0.f,             fy * rz,         0.f,
        -fx * t.x * rz2, -fy * t.y * rz2, 0.f
    );
    float3x3 V = float3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    // cov = T * V * Tt; G = df/dcov = v_cov
    // -> d/dV = Tt * G * T
    // -> df/dT = G * T * Vt + Gt * T * V
    float3x3 v_cov = float3x3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );
    // clang-format on

    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    // vjp of cov3d parameters
    // v_cov3d_i = v_V : dV/d_cov3d_i
    // where : is frobenius inner product
    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    // compute df/d_mean3d
    // T = J * W
    float3x3 v_J = v_T * transpose(W);
    float rz3 = rz2 * rz;
    float3 v_t = float3(
        -fx * rz2 * v_J[2][0],
        -fy * rz2 * v_J[2][1],
        -fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[2][0] -
            fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[2][1]
    );
    // printf("v_t %.2f %.2f %.2f\n", v_t[0], v_t[1], v_t[2]);
    // printf("W %.2f %.2f %.2f\n", W[0][0], W[0][1], W[0][2]);
    v_mean3d[0] += (float)dot(v_t, W[0]);
    v_mean3d[1] += (float)dot(v_t, W[1]);
    v_mean3d[2] += (float)dot(v_t, W[2]);
}

inline float4 quat_to_rotmat_vjp(const float4 quat, const float3x3 v_R) {
    float s = rsqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const device float* v_cov3d,
    device float* v_scale, // float3
    device float* v_quat // float4
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    float3x3 v_V = float3x3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    float3x3 v_M = 2.f * v_V * M;
    v_scale[0] = (float)dot(R[0], v_M[0]);
    v_scale[1] = (float)dot(R[1], v_M[1]);
    v_scale[2] = (float)dot(R[2], v_M[2]);

    float3x3 v_R = v_M * S;
    float4 out_v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = out_v_quat.x;
    v_quat[1] = out_v_quat.y;
    v_quat[2] = out_v_quat.z;
    v_quat[3] = out_v_quat.w;
}

kernel void project_gaussians_backward_kernel(
    constant int& num_points,
    constant float* means3d, // float3
    constant float* scales, // float3
    constant float& glob_scale,
    constant float* quats, // float4
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant float* cov3d,
    constant int* radii,
    constant float* conics, // float3
    constant float* v_xy, // float2
    constant float* v_depth,
    constant float* v_conic, // float3
    device float* v_cov2d, // float3
    device float* v_cov3d,
    device float* v_mean3d, // float3
    device float* v_scale, // float3
    device float* v_quat, // float4
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = read_packed_float3(means3d, idx);
    float fx = intrins.x;
    float fy = intrins.y;
    // get v_mean3d from v_xy
    write_packed_float3(
        v_mean3d, idx,
        project_pix_vjp(projmat, p_world, img_size, read_packed_float2(v_xy, idx))
    );

    // get z gradient contribution to mean3d gradient
    // z = viemwat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] *
    // mean3d.z + viewmat[11]
    float v_z = v_depth[idx];
    write_packed_float3(
        v_mean3d, idx,
        read_packed_float3(v_mean3d, idx) + float3(viewmat[8], viewmat[9], viewmat[10]) * v_z
    );

    // get v_cov2d
    cov2d_to_conic_vjp(
        read_packed_float3(conics, idx),
        read_packed_float3(v_conic, idx),
        &(v_cov2d[3*idx])
    );
    // get v_cov3d (and v_mean3d contribution)
    project_cov3d_ewa_vjp(
        p_world,
        &(cov3d[6 * idx]),
        viewmat,
        fx,
        fy,
        read_packed_float3(v_cov2d, idx),
        &(v_mean3d[3*idx]),
        &(v_cov3d[6 * idx])
    );
    // get v_scale and v_quat
    scale_rot_to_cov3d_vjp(
        read_packed_float3(scales, idx),
        glob_scale,
        read_packed_float4(quats, idx),
        &(v_cov3d[6 * idx]),
        &(v_scale[3*idx]),
        &(v_quat[4*idx])
    );
}

kernel void compute_cov2d_bounds_kernel(
    constant uint& num_pts,
    constant float* covs2d,
    device float* conics,
    device float* radii,
    uint row [[thread_index_in_threadgroup]]
) {
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}