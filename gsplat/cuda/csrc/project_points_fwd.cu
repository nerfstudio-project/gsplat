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
 * Projection of Points (Single Batch) Forward Pass
 ****************************************************************************/

template <typename T>
__global__ void
project_points_fwd_kernel(const uint32_t C, const uint32_t N,
                                  const T *__restrict__ means,    // [N, 3]
                                  const T *__restrict__ viewmats, // [C, 4, 4]
                                  const T *__restrict__ Ks,       // [C, 3, 3]
                                  const int32_t image_width, const int32_t image_height,
                                  const T near_plane, const T far_plane,
                                  // outputs
                                  int32_t *__restrict__ radii,      // [C, N]
                                  T *__restrict__ means2d,      // [C, N, 2]
                                  T *__restrict__ depths       // [C, N]
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
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
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
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx] = 1;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
project_points_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width, const uint32_t image_height,
    const float near_plane, const float far_plane) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii = torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    
    if (C && N) {
        project_points_fwd_kernel<float><<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            viewmats.data_ptr<float>(), 
            Ks.data_ptr<float>(), 
            image_width, image_height,
            near_plane, far_plane, 
            radii.data_ptr<int32_t>(),
            means2d.data_ptr<float>(), 
            depths.data_ptr<float>());
    }
    return std::make_tuple(radii, means2d, depths);
}
