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

template <typename scalar_t>
__global__ void projection_ut_3dgs_fused_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,     // [B, N, 3]
    const scalar_t *__restrict__ quats,     // [B, N, 4]
    const scalar_t *__restrict__ scales,    // [B, N, 3]
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats0, // [B, C, 4, 4]
    const scalar_t *__restrict__ viewmats1, // [B, C, 4, 4] optional for rolling shutter
    const scalar_t *__restrict__ Ks,        // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model_type,
    // uncented transform
    const UnscentedTransformParameters ut_params,    
    const ShutterType rs_type,
    const scalar_t *__restrict__ radial_coeffs,     // [B, C, 6] or [B, C, 4] optional
    const scalar_t *__restrict__ tangential_coeffs, // [B, C, 2] optional
    const scalar_t *__restrict__ thin_prism_coeffs, // [B, C, 4] optional
    // outputs
    int32_t *__restrict__ radii,         // [B, C, N, 2]
    scalar_t *__restrict__ means2d,      // [B, C, N, 2]
    scalar_t *__restrict__ depths,       // [B, C, N]
    scalar_t *__restrict__ conics,       // [B, C, N, 3]
    scalar_t *__restrict__ compensations // [B, C, N] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) {
        return;
    }
    const uint32_t bid = idx / (C * N);  // batch id
    const uint32_t cid = (idx / N) % C;  // camera id 
    const uint32_t gid = idx % N;        // gaussian id

    // shift pointers to the current gaussian
    const glm::fvec3 mean = glm::make_vec3(means + bid * N * 3 + gid * 3);
    const glm::fvec3 scale = glm::make_vec3(scales + bid * N * 3 + gid * 3);
    glm::fquat quat = glm::fquat{
        quats[bid * N * 4 + gid * 4 + 0],
        quats[bid * N * 4 + gid * 4 + 1],
        quats[bid * N * 4 + gid * 4 + 2],
        quats[bid * N * 4 + gid * 4 + 3]};  // w,x,y,z quaternion
    quat = glm::normalize(quat);

    // shift pointers to the current camera. note that glm is colume-major.
    const vec2 focal_length = {Ks[bid * C * 9 + cid * 9 + 0], Ks[bid * C * 9 + cid * 9 + 4]};
    const vec2 principal_point = {Ks[bid * C * 9 + cid * 9 + 2], Ks[bid * C * 9 + cid * 9 + 5]};

    // Create rolling shutter parameter
    auto rs_params = RollingShutterParameters(
        viewmats0 + bid * C * 16 + cid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + bid * C * 16 + cid * 16
    );

    // transform Gaussian center to camera space
	// Interpolate to *center* shutter pose as single per-Gaussian camera pose
	const auto shutter_pose = interpolate_shutter_pose(0.5f, rs_params);
    const vec3 mean_c = glm::rotate(shutter_pose.q, mean) + shutter_pose.t;
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // projection using uncented transform
    ImageGaussianReturn image_gaussian_return;
    if (camera_model_type == CameraModelType::PINHOLE) {
        if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
            PerfectPinholeCameraModel::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            PerfectPinholeCameraModel camera_model(cm_params);
            image_gaussian_return =
                world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                    camera_model, rs_params, ut_params, mean, scale, quat);
        } else {
            OpenCVPinholeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 6>(radial_coeffs + bid * C * 6 + cid * 6);
            }
            if (tangential_coeffs != nullptr) {
                cm_params.tangential_coeffs = make_array<float, 2>(tangential_coeffs + bid * C * 2 + cid * 2);
            }
            if (thin_prism_coeffs != nullptr) {
                cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + bid * C * 4 + cid * 4);
            }
            OpenCVPinholeCameraModel camera_model(cm_params);
            image_gaussian_return =
                world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                    camera_model, rs_params, ut_params, mean, scale, quat);
        }
    } else if (camera_model_type == CameraModelType::FISHEYE) {
        OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = rs_type;
        cm_params.principal_point = { principal_point.x, principal_point.y };
        cm_params.focal_length = { focal_length.x, focal_length.y };
        if (radial_coeffs != nullptr) {
            cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + bid * C * 4 + cid * 4);
        }
        OpenCVFisheyeCameraModel camera_model(cm_params);
        image_gaussian_return =
            world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                camera_model, rs_params, ut_params, mean, scale, quat);
    } else {
        // should never reach here
        assert(false);
        return;
    }

    auto [mean2d, covar2d, valid_ut] = image_gaussian_return;
    if (!valid_ut) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

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
        float opacity = opacities[bid * N + gid];
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
    const at::Tensor means,                   // [..., N, 3]
    const at::Tensor quats,                   // [..., N, 4]
    const at::Tensor scales,                  // [..., N, 3]
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    // outputs
    at::Tensor radii,                      // [..., C, N, 2]
    at::Tensor means2d,                    // [..., C, N, 2]
    at::Tensor depths,                     // [..., C, N]
    at::Tensor conics,                     // [..., C, N, 3]
    at::optional<at::Tensor> compensations // [..., C, N] optional
) {
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t B = means.numel() / (N * 3); // number of batches
    uint32_t C = Ks.size(-3);             // number of cameras

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    projection_ut_3dgs_fused_kernel<float>
        <<<grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
            B,
            C,
            N,
            means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            opacities.has_value() ? opacities.value().data_ptr<float>() : nullptr,
            viewmats0.data_ptr<float>(),
            viewmats1.has_value() ? viewmats1.value().data_ptr<float>() : nullptr,
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            camera_model,
            // uncented transform
            ut_params,
            rs_type,
            radial_coeffs.has_value()
                ? radial_coeffs.value().data_ptr<float>()
                : nullptr,
            tangential_coeffs.has_value()
                ? tangential_coeffs.value().data_ptr<float>()
                : nullptr,
            thin_prism_coeffs.has_value()
                ? thin_prism_coeffs.value().data_ptr<float>()
                : nullptr,
            radii.data_ptr<int32_t>(),
            means2d.data_ptr<float>(),
            depths.data_ptr<float>(),
            conics.data_ptr<float>(),
            compensations.has_value()
                ? compensations.value().data_ptr<float>()
                : nullptr
        );
}


} // namespace gsplat
