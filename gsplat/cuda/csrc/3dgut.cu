#include "bindings.h"
#include "quaternion.cuh"
#include "transform.cuh"
#include "cameras.cuh"

#include <cooperative_groups.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Gaussians (Single Batch) Forward Pass
 ****************************************************************************/

template<class CameraModel>
__global__ void fully_fused_projection_3dgut_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const CameraModel camera_model,
    const RollingShutterParameters rs_params, 
    const UnscentedTransformParameters ut_params,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ quats,    // [N, 4]
    const float *__restrict__ scales,   // [N, 3]
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    // outputs
    int32_t *__restrict__ radii,      // [C, N]
    float *__restrict__ means2d,      // [C, N, 2]
    float *__restrict__ depths,       // [C, N]
    float *__restrict__ conics        // [C, N, 3]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    auto const mean = glm::make_vec3(means + gid * 3);
    auto const scale = glm::make_vec3(scales + gid * 3);
    // w,x,y,z quaternion
    auto const rotation_quat = glm::fquat{
        quats[gid * 4 + 0],
        quats[gid * 4 + 1],
        quats[gid * 4 + 2],
        quats[gid * 4 + 3]}; 


	// Interpolate to *center* shutter pose as single per-Gaussian camera pose
	const auto shutter_pose = interpolate_shutter_pose(0.5f, rs_params);
    const vec3 mean_c = apply_quaternion(shutter_pose.q, mean) + shutter_pose.t;

    // Near and far plane culling
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    // fixed number of rolling-shutter iterations - same as in NCore
    auto constexpr N_ROLLING_SHUTTER_ITERATIONS = 10; 

    // uncented transform
    auto const image_gaussian_return =
        world_gaussian_to_image_gaussian_unscented_transform_shutter_pose<N_ROLLING_SHUTTER_ITERATIONS>(
            camera_model, rs_params, ut_params, mean, scale, rotation_quat
        );
    auto const mean2D_ut = image_gaussian_return.mean; // vec2
    auto const cov2D_ut = image_gaussian_return.covariance; // vec3
    auto const valid_ut = image_gaussian_return.valid; // bool
    if (!valid_ut) {
        radii[idx] = 0;
        return;
    }

    const mat2 covar2d = mat2(
        cov2D_ut[0],
        cov2D_ut[1], // 1st column
        cov2D_ut[1],
        cov2D_ut[2] // 2nd column
    );
    const vec2 mean2d = mean2D_ut;

    float det = covar2d[0][0] * covar2d[1][1] - covar2d[0][1] * covar2d[1][0];
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
    auto image_width = camera_model.parameters.resolution[0];
    auto image_height = camera_model.parameters.resolution[1];
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
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_3dgut_fwd_tensor(
    const OpenCVPinholeCameraModelParameters &camera_params,   
    const RollingShutterParameters &rs_params, 
    const torch::Tensor &means,                // [N, 3]
    const torch::Tensor &quats,                // [N, 4]
    const torch::Tensor &scales,               // [N, 3]
    const float near_plane,
    const float far_plane,
    const float radius_clip
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = 1;                // number of cameras, only support 1 for now.

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii =
        torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());

    auto const camera_model = PerfectPinholeCameraModel({
        camera_params.resolution, 
        camera_params.shutter_type, 
        camera_params.principal_point, 
        camera_params.focal_length
    });

    // Use default parameters for unscented transform
    UnscentedTransformParameters ut_params;
    if (C && N) {
        fully_fused_projection_3dgut_fwd_kernel
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                camera_model,
                rs_params,
                ut_params,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                near_plane,
                far_plane,
                radius_clip,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                conics.data_ptr<float>()
            );
    }
    return std::make_tuple(radii, means2d, depths, conics);
}




} // namespace gsplat