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
 * Projection of Gaussians (Single Batch) Forward Pass
 ****************************************************************************/

template <typename T>
__global__ void
view_to_gaussians_fwd_kernel(const uint32_t C, const uint32_t N,
                                  const T *__restrict__ means,       // [N, 3]
                                  const T *__restrict__ quats,       // [N, 4] 
                                  const T *__restrict__ scales,      // [N, 3] 
                                  const T *__restrict__ camtoworlds,    // [C, 4, 4]
                                  const int32_t *__restrict__ radii, // [C, N]
                                  // outputs
                                  T *__restrict__ view2gaussians    // [C, N, 10]
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

    // precompute the value here to avoid repeated computations also reduce IO
	// v is the viewdirection and v^T is the transpose of v
	// t = position of the camera in the gaussian coordinate system
	// A = v^T @ R^T @ S^-1 @ S^-1 @ R @ v
	// B = t^T @ S^-1 @ S^-1 @ R @ v
	// C = t^T @ S^-1 @ S^-1 @ t
	// For the given caemra, t is fix and v depends on the pixel
	// therefore we can precompute A, B, C and use them in the forward pass
	// For A, we can precompute R^T @ S^-1 @ S^-1 @ R, which is a symmetric matrix and only store the upper triangle in 6 values
	// For B, we can precompute S^-1 @ S^-1 @ R @ v, which is a vector and store it in 3 values
	// and C is fixed, so we only need to store 1 value
	// Therefore, we only need to store 10 values in the view2gaussian matrix
    vec3<T> scales_inv_square = {1.0f / (scale.x * scale.x + 1e-7), 1.0f / (scale.y * scale.y + 1e-7), 1.0f / (scale.z * scale.z + 1e-7)};
	T CC = view2gaussian_t.x * view2gaussian_t.x * scales_inv_square.x + \
           view2gaussian_t.y * view2gaussian_t.y * scales_inv_square.y + \
           view2gaussian_t.z * view2gaussian_t.z * scales_inv_square.z;
          
	mat3<T> scales_inv_square_R = mat3<T>(
		scales_inv_square.x * view2gaussian_R[0][0], scales_inv_square.y * view2gaussian_R[0][1], scales_inv_square.z * view2gaussian_R[0][2],
		scales_inv_square.x * view2gaussian_R[1][0], scales_inv_square.y * view2gaussian_R[1][1], scales_inv_square.z * view2gaussian_R[1][2],
		scales_inv_square.x * view2gaussian_R[2][0], scales_inv_square.y * view2gaussian_R[2][1], scales_inv_square.z * view2gaussian_R[2][2]
	); 

	vec3<T> BB = view2gaussian_t * scales_inv_square_R;
	mat3<T> Sigma = glm::transpose(view2gaussian_R) * scales_inv_square_R;

	// write to view2gaussian
	view2gaussians[0] = Sigma[0][0];
	view2gaussians[1] = Sigma[0][1];
	view2gaussians[2] = Sigma[0][2];
	view2gaussians[3] = Sigma[1][1];
	view2gaussians[4] = Sigma[1][2];
	view2gaussians[5] = Sigma[2][2];
	view2gaussians[6] = BB.x;
	view2gaussians[7] = BB.y;
	view2gaussians[8] = BB.z;
	view2gaussians[9] = CC;	    
}


torch::Tensor view_to_gaussians_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const torch::Tensor &quats,                // [N, 4] 
    const torch::Tensor &scales,               // [N, 3] 
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &radii                // [C, N]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(radii);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor view2gaussians = torch::empty({C, N, 10}, means.options());
    
    if (C && N) {
        view_to_gaussians_fwd_kernel<float><<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            radii.data_ptr<int32_t>(),
            view2gaussians.data_ptr<float>());
    }
    return view2gaussians;
}
