#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;


/****************************************************************************
 * Projection of Gaussians (Single Batch) Backward Pass
 ****************************************************************************/

template <typename T>
__global__ void view_to_gaussians_bwd_kernel(
    // fwd inputs
    const uint32_t C, const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ quats,    // [N, 4] 
    const T *__restrict__ scales,   // [N, 3] 
    const T *__restrict__ camtoworlds, // [C, 4, 4] TODO world-to-cam
    const int32_t *__restrict__ radii,       // [C, N]
    // fwd outputs
    const T *__restrict__ view2gaussians,                      // [C, N, 10]
    // grad outputs
    const T *__restrict__ v_view2gaussians,                    // [C, N, 10]
    // grad inputs
    T *__restrict__ v_means,   // [N, 3]
    T *__restrict__ v_quats,   // [N, 4] 
    T *__restrict__ v_scales,  // [N, 3]
    T *__restrict__ v_camtoworlds // [C, 4, 4] optional
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
    camtoworlds += cid * 16;
    quats += gid * 4;
    scales += gid * 3;    
    view2gaussians += idx * 10;
    v_view2gaussians += idx * 10;
    
    // glm is column-major but input is row-major
    mat3<T> camtoworlds_R = mat3<T>(
        camtoworlds[0], camtoworlds[4], camtoworlds[8], // 1st column
        camtoworlds[1], camtoworlds[5], camtoworlds[9], // 2nd column
        camtoworlds[2], camtoworlds[6], camtoworlds[10] // 3rd column
    );
    vec3<T> camtoworlds_t = vec3<T>(camtoworlds[3], camtoworlds[7], camtoworlds[11]);

    vec3<T> mean = glm::make_vec3(means);
    vec4<T> quat = glm::make_vec4(quats);
    vec3<T> scale = glm::make_vec3(scales);
    mat3<T> rotmat = quat_to_rotmat<T>(quat);
    
    mat3<T> worldtogaussian_R = glm::transpose(rotmat);
    vec3<T> worldtogaussian_t = -worldtogaussian_R * mean;

    mat3<T> view2gaussian_R =  worldtogaussian_R * camtoworlds_R;
    vec3<T> view2gaussian_t = worldtogaussian_R * camtoworlds_t + worldtogaussian_t;

    vec3<T> scales_inv_square = {1.0f / (scale.x * scale.x + 1e-10f), 1.0f / (scale.y * scale.y + 1e-10f), 1.0f / (scale.z * scale.z + 1e-10f)};
	T CC = view2gaussian_t.x * view2gaussian_t.x * scales_inv_square.x + \
           view2gaussian_t.y * view2gaussian_t.y * scales_inv_square.y + \
           view2gaussian_t.z * view2gaussian_t.z * scales_inv_square.z;
          
	mat3<T> scales_inv_square_R = mat3<T>(
		scales_inv_square.x * view2gaussian_R[0][0], scales_inv_square.y * view2gaussian_R[0][1], scales_inv_square.z * view2gaussian_R[0][2],
		scales_inv_square.x * view2gaussian_R[1][0], scales_inv_square.y * view2gaussian_R[1][1], scales_inv_square.z * view2gaussian_R[1][2],
		scales_inv_square.x * view2gaussian_R[2][0], scales_inv_square.y * view2gaussian_R[2][1], scales_inv_square.z * view2gaussian_R[2][2]
	); 

    // gradient
    mat3<T> dL_dSigma = mat3<T>(
		v_view2gaussians[0], 0.5f * v_view2gaussians[1], 0.5f * v_view2gaussians[2],
		0.5f * v_view2gaussians[1], v_view2gaussians[3], 0.5f * v_view2gaussians[4],
		0.5f * v_view2gaussians[2], 0.5f * v_view2gaussians[4], v_view2gaussians[5]
	);
	vec3<T> dL_dB = vec3<T>(v_view2gaussians[6], v_view2gaussians[7], v_view2gaussians[8]);
	T dL_dCC = v_view2gaussians[9];

    // TODO
    // vec3<T> BB = view2gaussian_t * scales_inv_square_R;
	// mat3<T> Sigma = glm::transpose(view2gaussian_R) * scales_inv_square_R;
    mat3<T> dL_dS_inv_square_R = view2gaussian_R * dL_dSigma + glm::outerProduct(view2gaussian_t, dL_dB); 
	mat3<T> v_view2gaussian_R = glm::transpose(dL_dSigma * glm::transpose(scales_inv_square_R));
	
	// mat3<T> scales_inv_square_R = mat3<T>(
	// 	scales_inv_square.x * view2gaussian_R[0][0], scales_inv_square.y * view2gaussian_R[0][1], scales_inv_square.z * view2gaussian_R[0][2],
	// 	scales_inv_square.x * view2gaussian_R[1][0], scales_inv_square.y * view2gaussian_R[1][1], scales_inv_square.z * view2gaussian_R[1][2],
	// 	scales_inv_square.x * view2gaussian_R[2][0], scales_inv_square.y * view2gaussian_R[2][1], scales_inv_square.z * view2gaussian_R[2][2]
	// ); 
	v_view2gaussian_R += mat3<T>(
		scales_inv_square.x * dL_dS_inv_square_R[0][0], scales_inv_square.y * dL_dS_inv_square_R[0][1], scales_inv_square.z * dL_dS_inv_square_R[0][2],
		scales_inv_square.x * dL_dS_inv_square_R[1][0], scales_inv_square.y * dL_dS_inv_square_R[1][1], scales_inv_square.z * dL_dS_inv_square_R[1][2],
		scales_inv_square.x * dL_dS_inv_square_R[2][0], scales_inv_square.y * dL_dS_inv_square_R[2][1], scales_inv_square.z * dL_dS_inv_square_R[2][2]
	); 
	vec3<T> dL_dS_inv_square = vec3<T>(
		dL_dS_inv_square_R[0][0] * view2gaussian_R[0][0] + dL_dS_inv_square_R[1][0] * view2gaussian_R[1][0] + dL_dS_inv_square_R[2][0] * view2gaussian_R[2][0],
		dL_dS_inv_square_R[0][1] * view2gaussian_R[0][1] + dL_dS_inv_square_R[1][1] * view2gaussian_R[1][1] + dL_dS_inv_square_R[2][1] * view2gaussian_R[2][1],
		dL_dS_inv_square_R[0][2] * view2gaussian_R[0][2] + dL_dS_inv_square_R[1][2] * view2gaussian_R[1][2] + dL_dS_inv_square_R[2][2] * view2gaussian_R[2][2]
	);
    // T CC = view2gaussian_t.x * view2gaussian_t.x * scales_inv_square.x + \
    //        view2gaussian_t.y * view2gaussian_t.y * scales_inv_square.y + \
    //        view2gaussian_t.z * view2gaussian_t.z * scales_inv_square.z;
    vec3<T> v_view2gaussian_t = vec3<T>(
		2.f * view2gaussian_t.x * scales_inv_square.x * dL_dCC + dL_dB.x * scales_inv_square_R[0][0] + dL_dB.y * scales_inv_square_R[1][0] + dL_dB.z * scales_inv_square_R[2][0],
		2.f * view2gaussian_t.y * scales_inv_square.y * dL_dCC + dL_dB.x * scales_inv_square_R[0][1] + dL_dB.y * scales_inv_square_R[1][1] + dL_dB.z * scales_inv_square_R[2][1],
		2.f * view2gaussian_t.z * scales_inv_square.z * dL_dCC + dL_dB.x * scales_inv_square_R[0][2] + dL_dB.y * scales_inv_square_R[1][2] + dL_dB.z * scales_inv_square_R[2][2]
    );
	dL_dS_inv_square.x += dL_dCC * view2gaussian_t.x * view2gaussian_t.x;
	dL_dS_inv_square.y += dL_dCC * view2gaussian_t.y * view2gaussian_t.y;
	dL_dS_inv_square.z += dL_dCC * view2gaussian_t.z * view2gaussian_t.z;
	// vec3<T> scales_inv_square = {1.0f / (scale.x * scale.x + 1e-7), 1.0f / (scale.y * scale.y + 1e-7), 1.0f / (scale.z * scale.z + 1e-7)};
	vec3<T> v_scale = vec3<T>(
		-2.f / scale.x * scales_inv_square.x * dL_dS_inv_square.x,
		-2.f / scale.y * scales_inv_square.y * dL_dS_inv_square.y,
		-2.f / scale.z * scales_inv_square.z * dL_dS_inv_square.z
    );

    // mat3<T> view2gaussian_R =  worldtogaussian_R * camtoworlds_R;
    // vec3<T> view2gaussian_t = worldtogaussian_R * camtoworlds_t + worldtogaussian_t;
    vec3<T> v_worldtogaussian_t = v_view2gaussian_t;
    mat3<T> v_worldtogaussian_R = v_view2gaussian_R * glm::transpose(camtoworlds_R) + glm::outerProduct(v_view2gaussian_t, camtoworlds_t);
    // vec3<T> worldtogaussian_t = -worldtogaussian_R * mean;
    v_worldtogaussian_R -= glm::outerProduct(v_worldtogaussian_t, mean);
    vec3<T> v_mean = -glm::transpose(worldtogaussian_R) * v_worldtogaussian_t;
    // mat3<T> worldtogaussian_R = glm::transpose(rotmat);
    mat3<T> v_rotmat = glm::transpose(v_worldtogaussian_R);
    
    // grad for quat rotmat
    vec4<T> v_quat(0.f);
    quat_to_rotmat_vjp<T>(quat, v_rotmat, v_quat);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    warpSum(v_mean, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        v_means += gid * 3;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 3; i++) {
            gpuAtomicAdd(v_means + i, v_mean[i]);
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
        gpuAtomicAdd(v_scales + 2, v_scale[2]);
    }
    // not supported yet for viewmats
    // if (v_viewmats != nullptr) {
    //     auto warp_group_c = cg::labeled_partition(warp, cid);
    //     warpSum(v_R, warp_group_c);
    //     warpSum(v_t, warp_group_c);
    //     if (warp_group_c.thread_rank() == 0) {
    //         v_viewmats += cid * 16;
    //         PRAGMA_UNROLL
    //         for (uint32_t i = 0; i < 3; i++) { // rows
    //             PRAGMA_UNROLL
    //             for (uint32_t j = 0; j < 3; j++) { // cols
    //                 gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
    //             }
    //             gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
    //         }
    //     }
    // }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
view_to_gaussians_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const torch::Tensor &quats,                // [N, 4] 
    const torch::Tensor &scales,               // [N, 3] 
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &radii,                // [C, N]
    // fwd outputs
    const torch::Tensor &view2gaussians,                      // [C, N, 10]
    // grad outputs
    const torch::Tensor &v_view2gaussians,                    // [C, N, 10]
    const bool viewmats_requires_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(radii);
    CHECK_INPUT(view2gaussians);
    CHECK_INPUT(v_view2gaussians);

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
        view_to_gaussians_bwd_kernel<float><<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
            C, N, 
            means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            radii.data_ptr<int32_t>(), 
            view2gaussians.data_ptr<float>(),
            v_view2gaussians.data_ptr<float>(),
            v_means.data_ptr<float>(),
            v_quats.data_ptr<float>(),
            v_scales.data_ptr<float>(),
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
        // view_to_gaussians_bwd_kernel<double><<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
        //     C, N, 
        //     means.data_ptr<double>(),
        //     quats.data_ptr<double>(),
        //     scales.data_ptr<double>(),
        //     viewmats.data_ptr<double>(),
        //     radii.data_ptr<int32_t>(), 
        //     view2gaussians.data_ptr<double>(),
        //     v_view2gaussians.data_ptr<double>(),
        //     v_means.data_ptr<double>(),
        //     v_quats.data_ptr<double>(),
        //     v_scales.data_ptr<double>(),
        //     viewmats_requires_grad ? v_viewmats.data_ptr<double>() : nullptr);
    }
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}
