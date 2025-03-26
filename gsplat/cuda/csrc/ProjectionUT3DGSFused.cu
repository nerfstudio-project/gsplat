#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"
#include "Cameras.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t, class CameraModel>
__global__ void projection_ut_3dgs_fused_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ quats,    // [N, 4]
    const scalar_t *__restrict__ scales,   // [N, 3]
    const scalar_t *__restrict__ opacities, // [N] optional
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModel camera_model,
    const RollingShutterParameters rs_params, 
    const UnscentedTransformParameters ut_params,
    // outputs
    int32_t *__restrict__ radii,         // [C, N, 2]
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
    // const uint32_t cid = idx / N; // camera id (single camera for now)
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    const glm::fvec3 mean = glm::make_vec3(means + gid * 3);
    const glm::fvec3 scale = glm::make_vec3(scales + gid * 3);
    glm::fquat quat = glm::fquat{
        quats[gid * 4 + 0],
        quats[gid * 4 + 1],
        quats[gid * 4 + 2],
        quats[gid * 4 + 3]};  // w,x,y,z quaternion
    quat = glm::normalize(quat);

    // transform Gaussian center to camera space
	// Interpolate to *center* shutter pose as single per-Gaussian camera pose
	const auto shutter_pose = interpolate_shutter_pose(0.5f, rs_params);
    const vec3 mean_c = apply_quaternion(shutter_pose.q, mean) + shutter_pose.t;
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // fixed number of rolling-shutter iterations
    auto constexpr N_ROLLING_SHUTTER_ITERATIONS = 10; 

    // projection using uncented transform
    auto const image_gaussian_return =
        world_gaussian_to_image_gaussian_unscented_transform_shutter_pose<N_ROLLING_SHUTTER_ITERATIONS>(
            camera_model, rs_params, ut_params, mean, scale, quat
        );
    const glm::fvec2 mean2D_ut = image_gaussian_return.mean;
    const glm::fvec3 cov2D_ut = image_gaussian_return.covariance; 
    const bool valid_ut = image_gaussian_return.valid; 
    if (!valid_ut) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    mat2 covar2d = mat2(cov2D_ut[0], cov2D_ut[1], cov2D_ut[1], cov2D_ut[2]);
    vec2 mean2d = mean2D_ut;

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = 3.33f;
    if (opacities != nullptr) {
        float opacity = opacities[gid];
        opacity *= compensation;
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    float tmp = sqrtf(max(0.01f, b * b - det));
    float v1 = b + tmp; // larger eigenvalue
    float r1 = extend * sqrtf(v1);
    float radius_x = ceilf(min(extend * sqrtf(covar2d[0][0]), r1));
    float radius_y = ceilf(min(extend * sqrtf(covar2d[1][1]), r1));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // mask out gaussians outside the image region
    auto image_width = camera_model.parameters.resolution[0];
    auto image_height = camera_model.parameters.resolution[1];
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
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

void launch_projection_ut_3dgs_fused_kernel(
    // inputs
    const at::Tensor means,                // [N, 3]
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const at::optional<at::Tensor> opacities, // [N] optional
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelParametersVariant camera_model_params,
    const RollingShutterParameters rs_params,
    const UnscentedTransformParameters ut_params,
    // outputs
    at::Tensor radii,                      // [C, N, 2]
    at::Tensor means2d,                    // [C, N, 2]
    at::Tensor depths,                     // [C, N]
    at::Tensor conics,                     // [C, N, 3]
    at::optional<at::Tensor> compensations // [C, N] optional
) {
    // Note: quats need to be normalized before passing in.

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = 1; // number of cameras, only support single camera for now

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    auto launchKernel = [&](auto const& camera_model) {
        projection_ut_3dgs_fused_kernel<float>
            <<<grid,
            threads,
            shmem_size,
            at::cuda::getCurrentCUDAStream()>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                opacities.has_value() ? opacities.value().data_ptr<float>() : nullptr,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                camera_model,
                rs_params,
                ut_params,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                conics.data_ptr<float>(),
                compensations.has_value()
                    ? compensations.value().data_ptr<float>()
                    : nullptr
            );
    };

    std::visit(OverloadVisitor{
        [&](OpenCVPinholeCameraModelParameters const& params) {
            // check for perfect-pinhole special case (none of the distortion coefficients is non-zero)
            if (params.is_perfect_pinhole()) {
                // instantiate perfect pinhole camera model instance by discarding all zero distortion parameters
                auto const camera_model = PerfectPinholeCameraModel({
                    params.resolution, params.shutter_type, params.principal_point, params.focal_length
                });
                launchKernel(camera_model);
            } else {
                launchKernel(OpenCVPinholeCameraModel(params));
            }
        },
        [&](OpenCVFisheyeCameraModelParameters const& params) {
            launchKernel(OpenCVFisheyeCameraModel<>(params));
        },
    }, camera_model_params);
}


} // namespace gsplat
