#include "bindings.h"
#include "types.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "tetra.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename T>
__global__ void ray_tetra_intersection_bwd_kernel(
    // fwd inputs
    const uint32_t N,
    const T *__restrict__ rays_o,      // [N, 3]
    const T *__restrict__ rays_d,      // [N, 3]
    const T *__restrict__ vertices,    // [N, 4, 3]
    // grad outputs (only backpropagate to tetrahedron vertices)
    const T *__restrict__ v_t_entrys,  // [N]
    const T *__restrict__ v_t_exits,   // [N]
    // grad inputs (only backpropagate to tetrahedron vertices)
    T *__restrict__ v_vertices         // [N, 4, 3]
) {
    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) return;

    // shift pointers to the current ray and tetra
    rays_o += idx * 3;
    rays_d += idx * 3;
    vertices += idx * 12;

    v_t_entrys += idx;
    v_t_exits += idx;
    v_vertices += idx * 12;

    const vec3<T> ray_o = glm::make_vec3(rays_o);
    const vec3<T> ray_d = glm::make_vec3(rays_d);
    const vec3<T> v0 = glm::make_vec3(vertices);
    const vec3<T> v1 = glm::make_vec3(vertices + 3);
    const vec3<T> v2 = glm::make_vec3(vertices + 6);
    const vec3<T> v3 = glm::make_vec3(vertices + 9);
    
    const T v_t_entry = v_t_entrys[0];
    const T v_t_exit = v_t_exits[0];
    vec3<T> v_v[4] = {vec3<T>(0.f), vec3<T>(0.f), vec3<T>(0.f), vec3<T>(0.f)};

    // backpropagate
    ray_tetra_intersection_vjp(
        // fwd inputs
        ray_o,
        ray_d,
        v0,
        v1,
        v2,
        v3,
        // grad outputs
        v_t_entry,
        v_t_exit,
        // grad inputs
        v_v[0],
        v_v[1],
        v_v[2],
        v_v[3]
    );

    // write to outputs
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
        v_vertices[i * 3] = v_v[i].x;
        v_vertices[i * 3 + 1] = v_v[i].y;
        v_vertices[i * 3 + 2] = v_v[i].z;
    }
}

std::tuple<torch::Tensor> ray_tetra_intersection_bwd_tensor(
    const torch::Tensor &rays_o,     // [N, 3]
    const torch::Tensor &rays_d,     // [N, 3]
    const torch::Tensor &vertices,   // [N, 4, 3]
    const torch::Tensor &v_t_entrys, // [N]
    const torch::Tensor &v_t_exits   // [N]
) {
    GSPLAT_DEVICE_GUARD(rays_o);
    GSPLAT_CHECK_INPUT(rays_o);
    GSPLAT_CHECK_INPUT(rays_d);
    GSPLAT_CHECK_INPUT(vertices);
    GSPLAT_CHECK_INPUT(v_t_entrys);
    GSPLAT_CHECK_INPUT(v_t_exits);

    uint32_t N = rays_o.size(0);
    torch::Tensor v_vertices = torch::empty({N, 4, 3}, rays_o.options());

    if (N > 0) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        ray_tetra_intersection_bwd_kernel<float>
            <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS,
                0,
                stream>>>(
                N,
                rays_o.data_ptr<float>(),
                rays_d.data_ptr<float>(),
                vertices.data_ptr<float>(),
                v_t_entrys.data_ptr<float>(),
                v_t_exits.data_ptr<float>(),
                v_vertices.data_ptr<float>()
            );
    }
    return std::make_tuple(v_vertices);
} 

} // namespace gsplat
