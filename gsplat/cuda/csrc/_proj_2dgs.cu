#include "bindings.h"
#include "reduce.cuh"
#include "quaternion.cuh"
#include "transform.cuh"

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <ATen/cuda/Atomic.cuh>

namespace gsplat {

namespace cg = cooperative_groups;

inline __device__ float sum(vec3 a) {
    return a.x + a.y + a.z;
}

__device__ void compute_ray_transforms_aabb_vjp(
    const float *ray_transforms,
    const float *v_means2d,
    const vec3 v_normals,
    const mat3 W,
    const mat3 P,
    const vec3 cam_pos,
    const vec3 mean_c,
    const vec4 quat,
    const vec2 scale,
    mat3 &_v_ray_transforms,
    vec4 &v_quat,
    vec2 &v_scale,
    vec3 &v_mean
) {
    if (v_means2d[0] != 0 || v_means2d[1] != 0) {
        const float distance = ray_transforms[6] * ray_transforms[6] + ray_transforms[7] * ray_transforms[7] -
                            ray_transforms[8] * ray_transforms[8];
        const float f = 1 / (distance);
        const float dpx_dT00 = f * ray_transforms[6];
        const float dpx_dT01 = f * ray_transforms[7];
        const float dpx_dT02 = -f * ray_transforms[8];
        const float dpy_dT10 = f * ray_transforms[6];
        const float dpy_dT11 = f * ray_transforms[7];
        const float dpy_dT12 = -f * ray_transforms[8];
        const float dpx_dT30 = ray_transforms[0] * (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const float dpx_dT31 = ray_transforms[1] * (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const float dpx_dT32 = -ray_transforms[2] * (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);
        const float dpy_dT30 = ray_transforms[3] * (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const float dpy_dT31 = ray_transforms[4] * (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const float dpy_dT32 = -ray_transforms[5] * (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);

        _v_ray_transforms[0][0] += v_means2d[0] * dpx_dT00;
        _v_ray_transforms[0][1] += v_means2d[0] * dpx_dT01;
        _v_ray_transforms[0][2] += v_means2d[0] * dpx_dT02;
        _v_ray_transforms[1][0] += v_means2d[1] * dpy_dT10;
        _v_ray_transforms[1][1] += v_means2d[1] * dpy_dT11;
        _v_ray_transforms[1][2] += v_means2d[1] * dpy_dT12;
        _v_ray_transforms[2][0] += v_means2d[0] * dpx_dT30 + v_means2d[1] * dpy_dT30;
        _v_ray_transforms[2][1] += v_means2d[0] * dpx_dT31 + v_means2d[1] * dpy_dT31;
        _v_ray_transforms[2][2] += v_means2d[0] * dpx_dT32 + v_means2d[1] * dpy_dT32;
    }

    mat3 R = quat_to_rotmat(quat);
    mat3 v_M = P * glm::transpose(_v_ray_transforms);
    mat3 W_t = glm::transpose(W);
    mat3 v_RS = W_t * v_M;
    vec3 v_tn = W_t * v_normals;

    // dual visible
    vec3 tn = W * R[2];
    float cos = glm::dot(-tn, mean_c);
    float multiplier = cos > 0 ? 1 : -1;
    v_tn *= multiplier;

    mat3 v_R = mat3(v_RS[0] * scale[0], v_RS[1] * scale[1], v_tn);

    quat_to_rotmat_vjp(quat, v_R, v_quat);
    v_scale[0] += glm::dot(v_RS[0], R[0]);
    v_scale[1] += glm::dot(v_RS[1], R[1]);

    v_mean += v_RS[2];
}


/****************************************************************************
 * Projection of Gaussians (Single Batch) Forward Pass 2DGS
 ****************************************************************************/


 __global__ void fully_fused_projection_fwd_2dgs_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]:  Gaussian means. (i.e. source points)
    const float *__restrict__ quats,    // [N, 4]:  Quaternions (No need to be normalized): This is the rotation component (for 2D)
    const float *__restrict__ scales,   // [N, 3]:  Scales. [N, 3] scales for x, y, z
    const float *__restrict__ viewmats, // [C, 4, 4]:  Camera-to-World coordinate mat
                                    // [R t]
                                    // [0 1]
    const float *__restrict__ Ks,       // [C, 3, 3]:  Projective transformation matrix
                                    // [f_x 0  c_x]
                                    // [0  f_y c_y]
                                    // [0   0   1]  : f_x, f_y are focal lengths, c_x, c_y is coords for camera center on screen space
    const int32_t image_width,       // Image width  pixels
    const int32_t image_height,      // Image height pixels
    const float near_plane,              // Near clipping plane (for finite range used in z sorting)
    const float far_plane,               // Far clipping plane (for finite range used in z sorting)
    const float radius_clip,             // Radius clipping threshold (through away small primitives)
    // outputs
    int32_t *__restrict__ radii, // [C, N]   The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
    float *__restrict__ means2d,     // [C, N, 2] 2D means of the projected Gaussians.
    float *__restrict__ depths,      // [C, N] The z-depth of the projected Gaussians.
    float *__restrict__ ray_transforms,      // [C, N, 3, 3] Transformation matrices that transform xy-planes in pixel spaces into splat coordinates (WH)^T in equation (9) in paper
    float *__restrict__ normals      // [C, N, 3] The normals in camera spaces.
) {

    /**
     * ===============================================
     * Initialize execution and threading variables:
     * idx: global thread index
     * cid: camera id (N is the total number of primitives, C is the number of cameras)
     * gid: gaussian id (N is the total number of primitives, C is the number of cameras)

     * THIS KERNEL LAUNCHES PER PRIMITIVE PER CAMERA i.e. C*N THREADS IN TOTAL
     * ===============================================
    */

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();  // get the thread index from grid
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    /**
     * ===============================================
     * Load data and put together camera rotation / translation
     * ===============================================
    */

    // shift pointers to the current camera and gaussian
    means += gid * 3;      // find the mean of the primitive this thread is responsible for
    viewmats += cid * 16;  // step 4x4 camera matrix
    Ks += cid * 9;         // step 3x3 intrinsic matrix

    // glm is column-major but input is row-major
    // rotation component of the camera. Explicit Transpose
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    // translation component of the camera
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    /**
     * ===============================================
     * Build ray transformation matrix from Primitive to Camera
     * in the original paper, q_ray [xz, yz, z, 1] = WH * q_uv : [u,v,1,1]
     * 
     * Thus: RS_camera = R * H(P->W)

     * Since H matrix (4x4) is defined as:
     * [v_x v_y 0_vec3  t]
     * [0   0   0       1]
     * 
     * thus RS_Camera defined as R * [v_x v_y 0], which gives
     * [R⋅v_x R⋅v_y 0]
     * Thus the only non zero terms will be the first two columns of R
     *
     * This gives the "affine rotation component" from uv to camera space as RS_camera
     *
     * the final addition component will be mean_c, which is the center of primitive in camera space, as
     * q_cam = RS_camera * q_uv + mean_c
     *
     * Like with homogeneous coordinates. if we encode incoming 2d points as [u,v,1], we can have:
     * q_cam = [RS_camera[0,1] | mean_c] * [u,v,1] 
     * ===============================================
    */

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);

    // return this thread for overly small primitives
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    quats += gid * 4;
    scales += gid * 3;

    mat3 RS_camera =
        R * quat_to_rotmat(glm::make_vec4(quats)) *
        mat3(scales[0], 0.0      , 0.0,
                0.0      , scales[1], 0.0,
                0.0      , 0.0      , 1.0);

    mat3 WH = mat3(RS_camera[0], RS_camera[1], mean_c);

    // projective transformation matrix: Camera -> Screen
    // when write in this order, the matrix is actually K^T as glm will read it in column major order
    // [Ks[0],  0,  0]
    // [0,   Ks[4],  0]
    // [Ks[2], Ks[5],  1]
    mat3 world_2_pix =
        mat3(Ks[0], 0.0  , Ks[2],
                0.0  , Ks[4], Ks[5],
                0.0  , 0.0  , 1.0);

    // WH is defined as [R⋅v_x, R⋅v_y, mean_c]: q_uv = [u,v,-1] -> q_cam = [c1,c2,c3]
    // here is the issue, world_2_pix is actually K^T
    // M is thus (KWH)^T = (WH)^T * K^T = (WH)^T * world_2_pix
    // thus M stores the "row majored" version of KWH, or column major version of (KWH)^T
    mat3 M = glm::transpose(WH) * world_2_pix;
    /**
     * ===============================================
     * Compute AABB
     * ===============================================
     */

    // compute AABB
    const vec3 M0 = vec3(M[0][0], M[0][1], M[0][2]);  // the first column of KWH^T, thus first row of KWH
    const vec3 M1 = vec3(M[1][0], M[1][1], M[1][2]);  // the second column of KWH^T, thus second row of KWH
    const vec3 M2 = vec3(M[2][0], M[2][1], M[2][2]);  // the third column of KWH^T, thus third row of KWH

    // we know that KWH brings [u,v,-1] to ray1, ray2, ray3] = [xz, yz, z]
    // temp_point is [1,1,-1], which is a "corner" of the UV space.
    const vec3 temp_point = vec3(1.0f, 1.0f, -1.0f);

    // ==============================================
    // trivial implementation to find mean and 1 sigma radius
    // ==============================================
    // const vec3 mean_ray = glm::transpose(M) * vec3(0.0f, 0.0f, -1.0f);
    // const vec3 temp_point_ray = glm::transpose(M) * temp_point;

    // const vec2 mean2d = vec2(mean_ray.x / mean_ray.z, mean_ray.y / mean_ray.z);
    // const vec2 half_extend_p = vec2(temp_point_ray.x / temp_point_ray.z, temp_point_ray.y / temp_point_ray.z) - mean2d;
    // const vec2 half_extend = vec2(half_extend_p.x * half_extend_p.x, half_extend_p.y * half_extend_p.y);

    // ==============================================
    // pro implementation
    // ==============================================
    // this is purely resulted from algebraic manipulation
    // check here for details: https://github.com/hbb1/diff-surfel-rasterization/issues/8#issuecomment-2138069016
    const float distance = sum(temp_point * M2 * M2);

    // ill-conditioned primitives will have distance = 0.0f, we ignore them
    if (distance == 0.0f)
        return;

    const vec3 f = (1 / distance) * temp_point;
    const vec2 mean2d = vec2(sum(f * M0 * M2), sum(f * M1 * M2));

    const vec2 temp = {sum(f * M0 * M0), sum(f * M1 * M1)};
    const vec2 half_extend = mean2d * mean2d - temp;

    // ==============================================
    const float radius =
        ceil(3.f * sqrt(max(1e-4, max(half_extend.x, half_extend.y))));

    if (radius <= radius_clip) {
        radii[idx] = 0;
        return;
    }

    // CULLING STEP:
    // mask out gaussians outside the image region
    if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
        mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
        radii[idx] = 0;  
        return;
    }

    // normals dual visible
    vec3 normal = RS_camera[2];
    // flip normal if it is pointing away from the camera
    float multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1;
    normal *= multipler;

    // write to outputs
    radii[idx] = (int32_t)radius;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;

    // row major storing (KWH)
    ray_transforms[idx * 9] = M0.x;
    ray_transforms[idx * 9 + 1] = M0.y;
    ray_transforms[idx * 9 + 2] = M0.z;
    ray_transforms[idx * 9 + 3] = M1.x;
    ray_transforms[idx * 9 + 4] = M1.y;
    ray_transforms[idx * 9 + 5] = M1.z;
    ray_transforms[idx * 9 + 6] = M2.x;
    ray_transforms[idx * 9 + 7] = M2.y;
    ray_transforms[idx * 9 + 8] = M2.z;

    // primitive normals
    normals[idx * 3] = normal.x;
    normals[idx * 3 + 1] = normal.y;
    normals[idx * 3 + 2] = normal.z;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_fwd_2dgs_tensor(
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 3]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii =
        torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::zeros({C, N}, means.options());
    torch::Tensor ray_transforms = torch::empty({C, N, 3, 3}, means.options());
    torch::Tensor normals = torch::zeros({C, N, 3}, means.options());

    if (C && N) {
        fully_fused_projection_fwd_2dgs_kernel
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                near_plane,
                far_plane,
                radius_clip,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                ray_transforms.data_ptr<float>(),
                normals.data_ptr<float>()
            );
    }
    return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
}

/****************************************************************************
 * Projection of Gaussians (Batched) Backward Pass
 ****************************************************************************/

__global__ void fully_fused_projection_bwd_2dgs_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ quats,    // [N, 4]
    const float *__restrict__ scales,   // [N, 3]
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    // fwd outputs
    const int32_t *__restrict__ radii, // [C, N]
    const float *__restrict__ ray_transforms,      // [C, N, 3, 3]
    // grad outputs
    const float *__restrict__ v_means2d, // [C, N, 2]
    const float *__restrict__ v_depths,  // [C, N]
    const float *__restrict__ v_normals, // [C, N, 3]
    // grad inputs
    float *__restrict__ v_ray_transforms,  // [C, N, 3, 3]
    float *__restrict__ v_means,   // [N, 3]
    float *__restrict__ v_quats,   // [N, 4]
    float *__restrict__ v_scales,  // [N, 3]
    float *__restrict__ v_viewmats // [C, 4, 4]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    ray_transforms += idx * 9;

    v_means2d += idx * 2;
    v_depths += idx;
    v_normals += idx * 3;
    v_ray_transforms += idx * 9;

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);

    vec4 quat = glm::make_vec4(quats + gid * 4);
    vec2 scale = glm::make_vec2(scales + gid * 3);

    mat3 P = mat3(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);

    mat3 _v_ray_transforms = mat3(
        v_ray_transforms[0],
        v_ray_transforms[1],
        v_ray_transforms[2],
        v_ray_transforms[3],
        v_ray_transforms[4],
        v_ray_transforms[5],
        v_ray_transforms[6],
        v_ray_transforms[7],
        v_ray_transforms[8]
    );

    _v_ray_transforms[2][2] += v_depths[0];

    vec3 v_normal = glm::make_vec3(v_normals);

    vec3 v_mean(0.f);
    vec2 v_scale(0.f);
    vec4 v_quat(0.f);
    compute_ray_transforms_aabb_vjp(
        ray_transforms,
        v_means2d,
        v_normal,
        R,
        P,
        t,
        mean_c,
        quat,
        scale,
        _v_ray_transforms,
        v_quat,
        v_scale,
        v_mean
    );

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }

    // Directly output gradients w.r.t. the quaternion and scale
    warpSum(v_quat, warp_group_g);
    warpSum(v_scale, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        v_quats += gid * 4;
        v_scales += gid * 3;
        gpuAtomicAdd(v_quats, v_quat[0]);
        gpuAtomicAdd(v_quats + 1, v_quat[1]);
        gpuAtomicAdd(v_quats + 2, v_quat[2]);
        gpuAtomicAdd(v_quats + 3, v_quat[3]);
        gpuAtomicAdd(v_scales, v_scale[0]);
        gpuAtomicAdd(v_scales + 1, v_scale[1]);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_bwd_2dgs_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 2]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const torch::Tensor &radii,  // [C, N]
    const torch::Tensor &ray_transforms, // [C, N, 3, 3]
    // grad outputs
    const torch::Tensor &v_means2d, // [C, N, 2]
    const torch::Tensor &v_depths,  // [C, N]
    const torch::Tensor &v_normals, // [C, N, 3]
    const torch::Tensor &v_ray_transforms,  // [C, N, 3, 3]
    const bool viewmats_requires_grad
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(ray_transforms);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_normals);
    GSPLAT_CHECK_INPUT(v_ray_transforms);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_quats = torch::zeros_like(quats);
    torch::Tensor v_scales = torch::zeros_like(scales);
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        fully_fused_projection_bwd_2dgs_kernel
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                radii.data_ptr<int32_t>(),
                ray_transforms.data_ptr<float>(),
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_normals.data_ptr<float>(),
                v_ray_transforms.data_ptr<float>(),
                v_means.data_ptr<float>(),
                v_quats.data_ptr<float>(),
                v_scales.data_ptr<float>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}


/****************************************************************************
 * Projection of Gaussians (Batched) Backward Pass 2DGS
 ****************************************************************************/


 __global__ void fully_fused_projection_packed_bwd_2dgs_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ quats,    // [N, 4]
    const float *__restrict__ scales,   // [N, 3]
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    // fwd outputs
    const int64_t *__restrict__ camera_ids,   // [nnz]
    const int64_t *__restrict__ gaussian_ids, // [nnz]
    const float *__restrict__ ray_transforms,             // [nnz, 3]
    // grad outputs
    const float *__restrict__ v_means2d, // [nnz, 2]
    const float *__restrict__ v_depths,  // [nnz]
    const float *__restrict__ v_normals, // [nnz, 3]
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    // grad inputs
    float *__restrict__ v_ray_transforms,
    float *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    float *__restrict__ v_quats,   // [N, 4] or [nnz, 4] Optional
    float *__restrict__ v_scales,  // [N, 3] or [nnz, 3] Optional
    float *__restrict__ v_viewmats // [C, 4, 4] Optional
) {
    // parallelize over nnz.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= nnz) {
        return;
    }
    const int64_t cid = camera_ids[idx];   // camera id
    const int64_t gid = gaussian_ids[idx]; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    ray_transforms += idx * 9;

    v_means2d += idx * 2;
    v_normals += idx * 3;
    v_depths += idx;
    v_ray_transforms += idx * 9;

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);

    vec4 quat = glm::make_vec4(quats + gid * 4);
    vec2 scale = glm::make_vec2(scales + gid * 3);
    mat3 P = mat3(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);

    mat3 _v_ray_transforms = mat3(
        v_ray_transforms[0],
        v_ray_transforms[1],
        v_ray_transforms[2],
        v_ray_transforms[3],
        v_ray_transforms[4],
        v_ray_transforms[5],
        v_ray_transforms[6],
        v_ray_transforms[7],
        v_ray_transforms[8]
    );

    _v_ray_transforms[2][2] += v_depths[0];

    vec3 v_normal = glm::make_vec3(v_normals);

    vec3 v_mean(0.f);
    vec2 v_scale(0.f);
    vec4 v_quat(0.f);
    compute_ray_transforms_aabb_vjp(
        ray_transforms,
        v_means2d,
        v_normal,
        R,
        P,
        t,
        mean_c,
        quat,
        scale,
        _v_ray_transforms,
        v_quat,
        v_scale,
        v_mean
    );

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    if (sparse_grad) {
        // write out results with sparse layout
        if (v_means != nullptr) {
            v_means += idx * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                v_means[i] = v_mean[i];
            }
        }
        v_quats += idx * 4;
        v_scales += idx * 3;
        v_quats[0] = v_quat[0];
        v_quats[1] = v_quat[1];
        v_quats[2] = v_quat[2];
        v_quats[3] = v_quat[3];
        v_scales[0] = v_scale[0];
        v_scales[1] = v_scale[1];
    } else {
        // write out results with dense layout
        // #if __CUDA_ARCH__ >= 700
        // write out results with warp-level reduction
        auto warp_group_g = cg::labeled_partition(warp, gid);
        if (v_means != nullptr) {
            warpSum(v_mean, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_means += gid * 3;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t i = 0; i < 3; i++) {
                    gpuAtomicAdd(v_means + i, v_mean[i]);
                }
            }
        }
        // Directly output gradients w.r.t. the quaternion and scale
        warpSum(v_quat, warp_group_g);
        warpSum(v_scale, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_quats += gid * 4;
            v_scales += gid * 3;
            gpuAtomicAdd(v_quats, v_quat[0]);
            gpuAtomicAdd(v_quats + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + 3, v_quat[3]);
            gpuAtomicAdd(v_scales, v_scale[0]);
            gpuAtomicAdd(v_scales + 1, v_scale[1]);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_packed_bwd_2dgs_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 3]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const torch::Tensor &camera_ids,   // [nnz]
    const torch::Tensor &gaussian_ids, // [nnz]
    const torch::Tensor &ray_transforms,       // [nnz, 3, 3]
    // grad outputs
    const torch::Tensor &v_means2d, // [nnz, 2]
    const torch::Tensor &v_depths,  // [nnz]
    const torch::Tensor &v_ray_transforms,  // [nnz, 3, 3]
    const torch::Tensor &v_normals, // [nnz, 3]
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {

    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(camera_ids);
    GSPLAT_CHECK_INPUT(gaussian_ids);
    GSPLAT_CHECK_INPUT(ray_transforms);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_normals);
    GSPLAT_CHECK_INPUT(v_ray_transforms);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    uint32_t nnz = camera_ids.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means, v_quats, v_scales, v_viewmats;
    if (sparse_grad) {
        v_means = torch::zeros({nnz, 3}, means.options());

        v_quats = torch::zeros({nnz, 4}, quats.options());
        v_scales = torch::zeros({nnz, 3}, scales.options());

        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros({C, 4, 4}, viewmats.options());
        }

    } else {
        v_means = torch::zeros_like(means);

        v_quats = torch::zeros_like(quats);
        v_scales = torch::zeros_like(scales);

        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros_like(viewmats);
        }
    }
    if (nnz) {

        fully_fused_projection_packed_bwd_2dgs_kernel
            <<<(nnz + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                nnz,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                ray_transforms.data_ptr<float>(),
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_normals.data_ptr<float>(),
                sparse_grad,
                v_ray_transforms.data_ptr<float>(),
                v_means.data_ptr<float>(),
                v_quats.data_ptr<float>(),
                v_scales.data_ptr<float>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}


/****************************************************************************
 * Projection of Gaussians (Batched) Forward Pass 2DGS
 ****************************************************************************/


 __global__ void fully_fused_projection_packed_fwd_2dgs_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ quats,    // [N, 4]
    const float *__restrict__ scales,   // [N, 3]
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const int32_t
        *__restrict__ block_accum,    // [C * blocks_per_row] packing helper
    int32_t *__restrict__ block_cnts, // [C * blocks_per_row] packing helper
    // outputs
    int32_t *__restrict__ indptr,       // [C + 1]
    int64_t *__restrict__ camera_ids,   // [nnz]
    int64_t *__restrict__ gaussian_ids, // [nnz]
    int32_t *__restrict__ radii,        // [nnz]
    float *__restrict__ means2d,            // [nnz, 2]
    float *__restrict__ depths,             // [nnz]
    float *__restrict__ ray_transforms,             // [nnz, 3, 3]
    float *__restrict__ normals             // [nnz, 3]
) {
    int32_t blocks_per_row = gridDim.x;

    int32_t row_idx = blockIdx.y; // cid
    int32_t block_col_idx = blockIdx.x;
    int32_t block_idx = row_idx * blocks_per_row + block_col_idx;

    int32_t col_idx = block_col_idx * blockDim.x + threadIdx.x; // gid

    bool valid = (row_idx < C) && (col_idx < N);

    // check if points are with camera near and far plane
    vec3 mean_c;
    mat3 R;
    if (valid) {
        // shift pointers to the current camera and gaussian
        means += col_idx * 3;
        viewmats += row_idx * 16;

        // glm is column-major but input is row-major
        R = mat3(
            viewmats[0],
            viewmats[4],
            viewmats[8], // 1st column
            viewmats[1],
            viewmats[5],
            viewmats[9], // 2nd column
            viewmats[2],
            viewmats[6],
            viewmats[10] // 3rd column
        );
        vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

        // transform Gaussian center to camera space
        posW2C(R, t, glm::make_vec3(means), mean_c);
        if (mean_c.z < near_plane || mean_c.z > far_plane) {
            valid = false;
        }
    }

    vec2 mean2d;
    mat3 M;
    float radius;
    vec3 normal;
    if (valid) {
        // build ray transformation matrix and transform from world space to
        // camera space
        quats += col_idx * 4;
        scales += col_idx * 3;

        mat3 RS_camera =
            R * quat_to_rotmat(glm::make_vec4(quats)) *
            mat3(scales[0], 0.0, 0.0, 0.0, scales[1], 0.0, 0.0, 0.0, 1.0);
        ;
        mat3 WH = mat3(RS_camera[0], RS_camera[1], mean_c);

        mat3 world_2_pix =
            mat3(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);
        M = glm::transpose(WH) * world_2_pix;

        // compute AABB
        const vec3 M0 = vec3(M[0][0], M[0][1], M[0][2]);
        const vec3 M1 = vec3(M[1][0], M[1][1], M[1][2]);
        const vec3 M2 = vec3(M[2][0], M[2][1], M[2][2]);

        const vec3 temp_point = vec3(1.0f, 1.0f, -1.0f);
        const float distance = sum(temp_point * M2 * M2);

        if (distance == 0.0f)
            valid = false;

        const vec3 f = (1 / distance) * temp_point;
        mean2d = vec2(sum(f * M0 * M2), sum(f * M1 * M2));

        const vec2 temp = {sum(f * M0 * M0), sum(f * M1 * M1)};
        const vec2 half_extend = mean2d * mean2d - temp;
        radius = ceil(3.f * sqrt(max(1e-4, max(half_extend.x, half_extend.y))));

        if (radius <= radius_clip) {
            valid = false;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
            mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
            valid = false;
        }

        // normal dual visible
        normal = RS_camera[2];
        float multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1;
        normal *= multipler;
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        // First pass: compute the block-wide sum
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, GSPLAT_N_THREADS> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        } else {
            aggregate = 0;
        }
        if (threadIdx.x == 0) {
            block_cnts[block_idx] = aggregate;
        }
    } else {
        // Second pass: write out the indices of the non zero elements
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, GSPLAT_N_THREADS> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if (valid) {
            if (block_idx > 0) {
                int32_t offset = block_accum[block_idx - 1];
                thread_data += offset;
            }
            // write to outputs
            camera_ids[thread_data] = row_idx;   // cid
            gaussian_ids[thread_data] = col_idx; // gid
            radii[thread_data] = (int32_t)radius;
            means2d[thread_data * 2] = mean2d.x;
            means2d[thread_data * 2 + 1] = mean2d.y;
            depths[thread_data] = mean_c.z;
            ray_transforms[thread_data * 9] = M[0][0];
            ray_transforms[thread_data * 9 + 1] = M[0][1];
            ray_transforms[thread_data * 9 + 2] = M[0][2];
            ray_transforms[thread_data * 9 + 3] = M[1][0];
            ray_transforms[thread_data * 9 + 4] = M[1][1];
            ray_transforms[thread_data * 9 + 5] = M[1][2];
            ray_transforms[thread_data * 9 + 6] = M[2][0];
            ray_transforms[thread_data * 9 + 7] = M[2][1];
            ray_transforms[thread_data * 9 + 8] = M[2][2];
            normals[thread_data * 3] = normal.x;
            normals[thread_data * 3 + 1] = normal.y;
            normals[thread_data * 3 + 2] = normal.z;
        }
        // lane 0 of the first block in each row writes the indptr
        if (threadIdx.x == 0 && block_col_idx == 0) {
            if (row_idx == 0) {
                indptr[0] = 0;
                indptr[C] = block_accum[C * blocks_per_row - 1];
            } else {
                indptr[row_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_packed_fwd_2dgs_tensor(
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 3]
    const torch::Tensor &scales,   // [N, 3]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    auto opt = means.options().dtype(torch::kInt32);

    uint32_t nrows = C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;

    dim3 threads = {GSPLAT_N_THREADS, 1, 1};
    // limit on the number of blocks: [2**31 - 1, 65535, 65535]
    dim3 blocks = {blocks_per_row, nrows, 1};

    // first pass
    int32_t nnz;
    torch::Tensor block_accum;
    if (C && N) {
        torch::Tensor block_cnts = torch::empty({nrows * blocks_per_row}, opt);
        fully_fused_projection_packed_fwd_2dgs_kernel
            <<<blocks, threads, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                near_plane,
                far_plane,
                radius_clip,
                nullptr,
                block_cnts.data_ptr<int32_t>(),
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );
        block_accum = torch::cumsum(block_cnts, 0, torch::kInt32);
        nnz = block_accum[-1].item<int32_t>();
    } else {
        nnz = 0;
    }

    // second pass
    torch::Tensor indptr = torch::empty({C + 1}, opt);
    torch::Tensor camera_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor gaussian_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor radii =
        torch::empty({nnz}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({nnz, 2}, means.options());
    torch::Tensor depths = torch::empty({nnz}, means.options());
    torch::Tensor ray_transforms = torch::empty({nnz, 3, 3}, means.options());
    torch::Tensor normals = torch::empty({nnz, 3}, means.options());

    if (nnz) {
        fully_fused_projection_packed_fwd_2dgs_kernel
            <<<blocks, threads, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                near_plane,
                far_plane,
                radius_clip,
                block_accum.data_ptr<int32_t>(),
                nullptr,
                indptr.data_ptr<int32_t>(),
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                ray_transforms.data_ptr<float>(),
                normals.data_ptr<float>()
            );
    } else {
        indptr.fill_(0);
    }

    return std::make_tuple(
        indptr,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        ray_transforms,
        normals
    );
}

} // namespace gsplat