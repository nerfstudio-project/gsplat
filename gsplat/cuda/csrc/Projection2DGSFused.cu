#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Projection2DGS.cuh" // Utils for 2DGS Projection
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_2dgs_fused_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t
        *__restrict__ means, // [N, 3]:  Gaussian means. (i.e. source points)
    const scalar_t
        *__restrict__ quats, // [N, 4]:  Quaternions (No need to be normalized):
                             // This is the rotation component (for 2D)
    const scalar_t
        *__restrict__ scales, // [N, 3]:  Scales. [N, 3] scales for x, y, z
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]:  World-to-Camera
                                           // coordinate mat [R t] [0 1]
    const scalar_t
        *__restrict__ Ks, // [C, 3, 3]:  Projective transformation matrix
                          // [f_x 0  c_x]
                          // [0  f_y c_y]
                          // [0   0   1]  : f_x, f_y are focal lengths, c_x, c_y
                          // is coords for camera center on screen space
    const int32_t image_width,  // Image width  pixels
    const int32_t image_height, // Image height pixels
    const scalar_t
        near_plane, // Near clipping plane (for finite range used in z sorting)
    const scalar_t
        far_plane, // Far clipping plane (for finite range used in z sorting)
    const scalar_t radius_clip, // Radius clipping threshold (through away small
                                // primitives)
    // outputs
    int32_t *__restrict__ radii, // [C, N, 2]   The maximum radius of the projected
                                 // Gaussians in pixel unit. Int32 tensor.
    scalar_t
        *__restrict__ means2d, // [C, N, 2] 2D means of the projected Gaussians.
    scalar_t
        *__restrict__ depths, // [C, N] The z-depth of the projected Gaussians.
    scalar_t
        *__restrict__ ray_transforms, // [C, N, 3, 3] Transformation matrices
                                      // that transform xy-planes in pixel
                                      // spaces into splat coordinates (WH)^T in
                                      // equation (9) in paper
    scalar_t *__restrict__ normals    // [C, N, 3] The normals in camera spaces.
) {

    /**
     * ===============================================
     * Initialize execution and threading variables:
     * idx: global thread index
     * cid: camera id (N is the total number of primitives, C is the number of
     cameras)
     * gid: gaussian id (N is the total number of primitives, C is the number of
     cameras)

     * THIS KERNEL LAUNCHES PER PRIMITIVE PER CAMERA i.e. C*N THREADS IN TOTAL
     * ===============================================
    */

    // parallelize over C * N.
    uint32_t idx =
        cg::this_grid().thread_rank(); // get the thread index from grid
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
    means += gid *
             3; // find the mean of the primitive this thread is responsible for
    viewmats += cid * 16; // step 4x4 camera matrix
    Ks += cid * 9;        // step 3x3 intrinsic matrix

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
     * This gives the "affine rotation component" from uv to camera space as
     RS_camera
     *
     * the final addition component will be mean_c, which is the center of
     primitive in camera space, as
     * q_cam = RS_camera * q_uv + mean_c
     *
     * Like with homogeneous coordinates. if we encode incoming 2d points as
     [u,v,1], we can have:
     * q_cam = [RS_camera[0,1] | mean_c] * [u,v,1]
     * ===============================================
    */

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);

    // return this thread for overly small primitives
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    quats += gid * 4;
    scales += gid * 3;

    mat3 RS_camera =
        R * quat_to_rotmat(glm::make_vec4(quats)) *
        mat3(scales[0], 0.0, 0.0, 0.0, scales[1], 0.0, 0.0, 0.0, 1.0);

    mat3 WH = mat3(RS_camera[0], RS_camera[1], mean_c);

    // projective transformation matrix: Camera -> Screen
    // when write in this order, the matrix is actually K^T as glm will read it
    // in column major order [Ks[0],  0,  0] [0,   Ks[4],  0] [Ks[2], Ks[5],  1]
    mat3 world_2_pix =
        mat3(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);

    // WH is defined as [R⋅v_x, R⋅v_y, mean_c]: q_uv = [u,v,-1] -> q_cam =
    // [c1,c2,c3] here is the issue, world_2_pix is actually K^T M is thus
    // (KWH)^T = (WH)^T * K^T = (WH)^T * world_2_pix thus M stores the "row
    // majored" version of KWH, or column major version of (KWH)^T
    mat3 M = glm::transpose(WH) * world_2_pix;
    /**
     * ===============================================
     * Compute AABB
     * ===============================================
     */

    // compute AABB
    const vec3 M0 = vec3(
        M[0][0], M[0][1], M[0][2]
    ); // the first column of KWH^T, thus first row of KWH
    const vec3 M1 = vec3(
        M[1][0], M[1][1], M[1][2]
    ); // the second column of KWH^T, thus second row of KWH
    const vec3 M2 = vec3(
        M[2][0], M[2][1], M[2][2]
    ); // the third column of KWH^T, thus third row of KWH

    // we know that KWH brings [u,v,-1] to ray1, ray2, ray3] = [xz, yz, z]
    // temp_point is [1,1,-1], which is a "corner" of the UV space.
    const vec3 temp_point = vec3(1.0f, 1.0f, -1.0f);

    // ==============================================
    // trivial implementation to find mean and 1 sigma radius
    // ==============================================
    // const vec3 mean_ray = glm::transpose(M) * vec3(0.0f, 0.0f, -1.0f);
    // const vec3 temp_point_ray = glm::transpose(M) * temp_point;

    // const vec2 mean2d = vec2(mean_ray.x / mean_ray.z, mean_ray.y /
    // mean_ray.z); const vec2 half_extend_p = vec2(temp_point_ray.x /
    // temp_point_ray.z, temp_point_ray.y / temp_point_ray.z) - mean2d; const
    // vec2 half_extend = vec2(half_extend_p.x * half_extend_p.x,
    // half_extend_p.y * half_extend_p.y);

    // ==============================================
    // pro implementation
    // ==============================================
    // this is purely resulted from algebraic manipulation
    // check here for details:
    // https://github.com/hbb1/diff-surfel-rasterization/issues/8#issuecomment-2138069016
    const float distance = sum(temp_point * M2 * M2);

    // ill-conditioned primitives will have distance = 0.0f, we ignore them
    if (distance == 0.0f)
        return;

    const vec3 f = (1 / distance) * temp_point;
    const vec2 mean2d = vec2(sum(f * M0 * M2), sum(f * M1 * M2));

    const vec2 temp = {sum(f * M0 * M0), sum(f * M1 * M1)};
    const vec2 half_extend = mean2d * mean2d - temp;

    // ==============================================
    const float radius_x = ceil(3.33f * sqrt(max(1e-4, half_extend.x)));
    const float radius_y = ceil(3.33f * sqrt(max(1e-4, half_extend.y)));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // CULLING STEP:
    // mask out gaussians outside the image region
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // normals dual visible
    vec3 normal = RS_camera[2];
    // flip normal if it is pointing away from the camera
    float multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1;
    normal *= multipler;

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
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

void launch_projection_2dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    // outputs
    at::Tensor radii,          // [C, N, 2]
    at::Tensor means2d,        // [C, N, 2]
    at::Tensor depths,         // [C, N]
    at::Tensor ray_transforms, // [C, N, 3, 3]
    at::Tensor normals         // [C, N, 3]
) {
    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    projection_2dgs_fused_fwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
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

template <typename scalar_t>
__global__ void projection_2dgs_fused_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ quats,    // [N, 4]
    const scalar_t *__restrict__ scales,   // [N, 3]
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    // fwd outputs
    const int32_t *__restrict__ radii,           // [C, N, 2]
    const scalar_t *__restrict__ ray_transforms, // [C, N, 3, 3]
    // grad outputs
    const scalar_t *__restrict__ v_means2d, // [C, N, 2]
    const scalar_t *__restrict__ v_depths,  // [C, N]
    const scalar_t *__restrict__ v_normals, // [C, N, 3]
    // grad inputs
    scalar_t *__restrict__ v_ray_transforms, // [C, N, 3, 3]
    scalar_t *__restrict__ v_means,          // [N, 3]
    scalar_t *__restrict__ v_quats,          // [N, 4]
    scalar_t *__restrict__ v_scales,         // [N, 3]
    scalar_t *__restrict__ v_viewmats        // [C, 4, 4]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
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
#pragma unroll
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

void launch_projection_2dgs_fused_bwd_kernel(
    // fwd inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor radii,          // [C, N, 2]
    const at::Tensor ray_transforms, // [C, N, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [C, N, 2]
    const at::Tensor v_depths,         // [C, N]
    const at::Tensor v_normals,        // [C, N, 3]
    const at::Tensor v_ray_transforms, // [C, N, 3, 3]
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means,   // [C, N, 3]
    at::Tensor v_quats,   // [C, N, 4]
    at::Tensor v_scales,  // [C, N, 3]
    at::Tensor v_viewmats // [C, 4, 4]
) {
    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    projection_2dgs_fused_bwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
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

} // namespace gsplat
