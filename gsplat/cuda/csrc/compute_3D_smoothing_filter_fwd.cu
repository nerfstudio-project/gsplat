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
 * Compute the 3D smoothing filter size of 3D Gaussians Forward Pass
 ****************************************************************************/

template <typename T>
__global__ void
compute_3D_smoothing_filter_fwd_kernel(const uint32_t C, const uint32_t N,
                                  const T *__restrict__ means,    // [N, 3]
                                  const T *__restrict__ viewmats, // [C, 4, 4]
                                  const T *__restrict__ Ks,       // [C, 3, 3]
                                  const int32_t image_width, const int32_t image_height,
                                  const T near_plane,
                                  // outputs
                                  T *__restrict__ filter // [N, ]
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
    mat3<T> R = mat3<T>(viewmats[0], viewmats[4], viewmats[8], // 1st column
                        viewmats[1], viewmats[5], viewmats[9], // 2nd column
                        viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane ) {
        return;
    }

    // project the point to image plane
    vec2<T> mean2d;
    
    const T fx = Ks[0];
    const T fy = Ks[4];
    const T cx = Ks[2];
    const T cy = Ks[5];

    T x = mean_c[0], y = mean_c[1], z = mean_c[2];
    T rz = 1.f / z;
    mean2d = vec2<T>({fx * x * rz + cx, fy * y * rz + cy});

    // mask out gaussians outside the image region
    if (mean2d.x <= 0 || mean2d.x >= image_width ||
        mean2d.y <= 0 || mean2d.y >= image_height) {
        return;
    }
    
    T filter_size = z / fx;
    
    // write to outputs
    // atomicMin(&filter[gid], filter_size);

    // atomicMin is not supported for float, so we use __float_as_int
    // refer to https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250
    atomicMin((int *)&filter[gid], __float_as_int(filter_size));
}


torch::Tensor compute_3D_smoothing_filter_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width, const uint32_t image_height,
    const float near_plane) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor filter = torch::full({N}, 1000000, means.options());
    
    if (C && N) {
        compute_3D_smoothing_filter_fwd_kernel<float><<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            viewmats.data_ptr<float>(), Ks.data_ptr<float>(), image_width, image_height,
            near_plane,
            filter.data_ptr<float>());
    }
    return filter;
}
