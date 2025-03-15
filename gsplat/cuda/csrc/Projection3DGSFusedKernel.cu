#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h> 

#include "Common.h"
#include "Utils.cuh"
#include "ProjectionKernels.h"

namespace gsplat{

template <typename scalar_t>
__global__ void projection_3dgs_fused_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ covars,   // [N, 6] optional
    const scalar_t *__restrict__ quats,    // [N, 4] optional
    const scalar_t *__restrict__ scales,   // [N, 3] optional
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,  // [C, N]
    scalar_t *__restrict__ means2d,      // [C, N, 2]
    scalar_t *__restrict__ depths,       // [C, N]
    scalar_t *__restrict__ conics,       // [C, N, 3]
    scalar_t *__restrict__ compensations // [C, N] optional
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

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
        quat_scale_to_covar_preci(
            glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
        );
    }
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    // take 3 sigma as the radius (non differentiable)
    float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    float v1 = b + sqrt(max(0.01f, b * b - det));
    float radius = ceil(3.f * sqrt(v1));
    // float v2 = b - sqrt(max(0.1f, b * b - det));
    // float radius = ceil(3.f * sqrt(max(v1, v2)));

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

    // write to outputs
    radii[idx] = (int32_t)radius;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}


void launch_projection_3dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> &covars, // [N, 6] optional
    const at::optional<at::Tensor> &quats,  // [N, 4] optional
    const at::optional<at::Tensor> &scales, // [N, 3] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,          // [C, N]
    at::Tensor means2d,       // [C, N, 2]
    at::Tensor depths,        // [C, N]
    at::Tensor conics,        // [C, N, 3]
    at::Tensor compensations  // [C, N] optional
){
    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    AT_DISPATCH_ALL_TYPES(
        means.scalar_type(), "projection_3dgs_fused_fwd_kernel",
        [&]() {
            projection_3dgs_fused_fwd_kernel<scalar_t>
                <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                C,
                N,
                means.data_ptr<scalar_t>(),
                covars.has_value() ? covars.value().data_ptr<scalar_t>() : nullptr,
                quats.has_value() ? quats.value().data_ptr<scalar_t>() : nullptr,
                scales.has_value() ? scales.value().data_ptr<scalar_t>() : nullptr,
                viewmats.data_ptr<scalar_t>(),
                Ks.data_ptr<scalar_t>(),
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                camera_model,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<scalar_t>(),
                depths.data_ptr<scalar_t>(),
                conics.data_ptr<scalar_t>(),
                calc_compensations ? compensations.data_ptr<scalar_t>() : nullptr
            );
        });
}


// template <typename scalar_t>
// __global__ void projection_3dgs_bwd_kernel(
//     const uint32_t C,
//     const uint32_t N,
//     const scalar_t *__restrict__ means,  // [C, N, 3]
//     const scalar_t *__restrict__ covars, // [C, N, 3, 3]
//     const scalar_t *__restrict__ Ks,     // [C, 3, 3]
//     const uint32_t width,
//     const uint32_t height,
//     const CameraModelType camera_model,
//     const scalar_t *__restrict__ v_means2d,  // [C, N, 2]
//     const scalar_t *__restrict__ v_covars2d, // [C, N, 2, 2]
//     scalar_t *__restrict__ v_means,          // [C, N, 3]
//     scalar_t *__restrict__ v_covars          // [C, N, 3, 3]
// ) {

//     // parallelize over C * N.
//     uint32_t idx = cg::this_grid().thread_rank();
//     if (idx >= C * N) {
//         return;
//     }
//     const uint32_t cid = idx / N; // camera id
//     // const uint32_t gid = idx % N; // gaussian id

//     // shift pointers to the current camera and gaussian
//     means += idx * 3;
//     covars += idx * 9;
//     v_means += idx * 3;
//     v_covars += idx * 9;
//     Ks += cid * 9;
//     v_means2d += idx * 2;
//     v_covars2d += idx * 4;

//     float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
//     mat3 v_covar(0.f);
//     vec3 v_mean(0.f);
//     const vec3 mean = glm::make_vec3(means);
//     const mat3 covar = glm::make_mat3(covars);
//     const vec2 v_mean2d = glm::make_vec2(v_means2d);
//     const mat2 v_covar2d = glm::make_mat2(v_covars2d);

//     switch (camera_model) {
//         case CameraModelType::PINHOLE: // perspective projection
//             persp_proj_vjp(
//                 mean,
//                 covar,
//                 fx,
//                 fy,
//                 cx,
//                 cy,
//                 width,
//                 height,
//                 glm::transpose(v_covar2d),
//                 v_mean2d,
//                 v_mean,
//                 v_covar
//             );
//             break;
//         case CameraModelType::ORTHO: // orthographic projection
//             ortho_proj_vjp(
//                 mean,
//                 covar,
//                 fx,
//                 fy,
//                 cx,
//                 cy,
//                 width,
//                 height,
//                 glm::transpose(v_covar2d),
//                 v_mean2d,
//                 v_mean,
//                 v_covar
//             );
//             break;
//         case CameraModelType::FISHEYE: // fisheye projection
//             fisheye_proj_vjp(
//                 mean,
//                 covar,
//                 fx,
//                 fy,
//                 cx,
//                 cy,
//                 width,
//                 height,
//                 glm::transpose(v_covar2d),
//                 v_mean2d,
//                 v_mean,
//                 v_covar
//             );
//             break;
//     }

//     // write to outputs: glm is column-major but we want row-major
//     #pragma unroll
//     for (uint32_t i = 0; i < 3; i++) { // rows
//         #pragma unroll
//         for (uint32_t j = 0; j < 3; j++) { // cols
//             v_covars[i * 3 + j] = v_covar[j][i];
//         }
//     }

//     #pragma unroll
//     for (uint32_t i = 0; i < 3; i++) {
//         v_means[i] = v_mean[i];
//     }
// }

// void launch_projection_3dgs_bwd_kernel(
//     // inputs
//     const at::Tensor means,  // [C, N, 3]
//     const at::Tensor covars, // [C, N, 3, 3]
//     const at::Tensor Ks,     // [C, 3, 3]
//     const uint32_t width,
//     const uint32_t height,
//     const CameraModelType camera_model,
//     const at::Tensor v_means2d, // [C, N, 2]
//     const at::Tensor v_covars2d, // [C, N, 2, 2]
//     // outputs
//     at::Tensor v_means, // [C, N, 3]
//     at::Tensor v_covars // [C, N, 3, 3]
// ){
//     uint32_t C = means.size(0);
//     uint32_t N = means.size(1);

//     int64_t n_elements = C * N;
//     dim3 threads(256);
//     dim3 grid((n_elements + threads.x - 1) / threads.x);
//     int64_t shmem_size = 0; // No shared memory used in this kernel

//     AT_DISPATCH_ALL_TYPES(
//         means.scalar_type(), "projection_3dgs_fwd_kernel",
//         [&]() {
//             projection_3dgs_bwd_kernel<scalar_t>
//                 <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
//                 C, N, 
//                 means.data_ptr<scalar_t>(),
//                 covars.data_ptr<scalar_t>(),
//                 Ks.data_ptr<scalar_t>(),
//                 width,
//                 height,
//                 camera_model,
//                 v_means2d.data_ptr<scalar_t>(),
//                 v_covars2d.data_ptr<scalar_t>(),
//                 v_means.data_ptr<scalar_t>(),
//                 v_covars.data_ptr<scalar_t>()
//             );
//         });
// }


} // namespace gsplat
