#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Gaussians (Single Batch) Forward Pass 2DGS
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_fwd_2dgs_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ quats,    // [N, 4]
    const T *__restrict__ scales,   // [N, 3]
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T near_plane,
    const T far_plane,
    const T radius_clip,
    // outputs
    int32_t *__restrict__ radii, // [C, N]
    T *__restrict__ means2d,     // [C, N, 2]
    T *__restrict__ depths,      // [C, N]
    T *__restrict__ ray_transforms,      // [C, N, 3, 3]
    T *__restrict__ normals      // [C, N, 3]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    mat3<T> R = mat3<T>(
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
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    // build ray transformation matrix and transform from world space to camera
    // space
    quats += gid * 4;
    scales += gid * 3;

    mat3<T> RS_camera =
        R * quat_to_rotmat<T>(glm::make_vec4(quats)) *
        mat3<T>(scales[0], 0.0, 0.0, 0.0, scales[1], 0.0, 0.0, 0.0, 1.0);

    mat3<T> WH = mat3<T>(RS_camera[0], RS_camera[1], mean_c);

    mat3<T> world_2_pix =
        mat3<T>(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);
    mat3<T> M = glm::transpose(WH) * world_2_pix;

    // compute AABB
    const vec3<T> M0 = vec3<T>(M[0][0], M[0][1], M[0][2]);
    const vec3<T> M1 = vec3<T>(M[1][0], M[1][1], M[1][2]);
    const vec3<T> M2 = vec3<T>(M[2][0], M[2][1], M[2][2]);

    const vec3<T> temp_point = vec3<T>(1.0f, 1.0f, -1.0f);
    const T distance = sum(temp_point * M2 * M2);

    if (distance == 0.0f)
        return;

    const vec3<T> f = (1 / distance) * temp_point;
    const vec2<T> mean2d = vec2<T>(sum(f * M0 * M2), sum(f * M1 * M2));

    const vec2<T> temp = {sum(f * M0 * M0), sum(f * M1 * M1)};
    const vec2<T> half_extend = mean2d * mean2d - temp;
    const T radius =
        ceil(3.f * sqrt(max(1e-4, max(half_extend.x, half_extend.y))));

    if (radius <= radius_clip) {
        radii[idx] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
        mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
        radii[idx] = 0;
        return;
    }

    // normals dual visible
    vec3<T> normal = RS_camera[2];
    T multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1;
    normal *= multipler;

    // write to outputs
    radii[idx] = (int32_t)radius;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    ray_transforms[idx * 9] = M0.x;
    ray_transforms[idx * 9 + 1] = M0.y;
    ray_transforms[idx * 9 + 2] = M0.z;
    ray_transforms[idx * 9 + 3] = M1.x;
    ray_transforms[idx * 9 + 4] = M1.y;
    ray_transforms[idx * 9 + 5] = M1.z;
    ray_transforms[idx * 9 + 6] = M2.x;
    ray_transforms[idx * 9 + 7] = M2.y;
    ray_transforms[idx * 9 + 8] = M2.z;
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
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor ray_transforms = torch::empty({C, N, 3, 3}, means.options());
    torch::Tensor normals = torch::empty({C, N, 3}, means.options());

    if (C && N) {
        fully_fused_projection_fwd_2dgs_kernel<float>
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

} // namespace gsplat