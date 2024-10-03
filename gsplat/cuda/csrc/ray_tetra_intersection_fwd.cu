#include "bindings.h"
#include "types.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "tetra.cuh" // ray_tetra_intersection

namespace gsplat {

namespace cg = cooperative_groups;

template <typename T>
__global__ void ray_tetra_intersection_fwd_kernel(
    // inputs
    const uint32_t N,
    const T *__restrict__ rays_o,         // [N, 3]
    const T *__restrict__ rays_d,         // [N, 3]
    const T *__restrict__ vertices,       // [N, 4, 3]
    // outputs
    int32_t *__restrict__ entry_face_ids, // [N]
    int32_t *__restrict__ exit_face_ids,  // [N]
    T *__restrict__ t_entrys,             // [N]
    T *__restrict__ t_exits               // [N]
) {
    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) return;

    // shift pointers to the current ray and tetra
    rays_o += idx * 3;
    rays_d += idx * 3;
    vertices += idx * 12;

    entry_face_ids += idx;
    exit_face_ids += idx;
    t_entrys += idx;
    t_exits += idx;

    // calculate intersection
    T t_entry, t_exit;
    int32_t entry_face_idx, exit_face_idx;
    bool hit = ray_tetra_intersection(
        // inputs
        glm::make_vec3(rays_o),
        glm::make_vec3(rays_d),
        glm::make_vec3(vertices),
        glm::make_vec3(vertices + 3),
        glm::make_vec3(vertices + 6),
        glm::make_vec3(vertices + 9),
        // outputs
        entry_face_idx,
        exit_face_idx,
        t_entry,
        t_exit
    );

    // store results
    entry_face_ids[0] = entry_face_idx;
    exit_face_ids[0] = exit_face_idx;   
    t_entrys[0] = t_entry;
    t_exits[0] = t_exit;
}

std::vector<torch::Tensor> ray_tetra_intersection_fwd_tensor(
    const torch::Tensor &rays_o,   // [N, 3]
    const torch::Tensor &rays_d,   // [N, 3]
    const torch::Tensor &vertices  // [N, 4, 3]
) {
    GSPLAT_DEVICE_GUARD(rays_o);
    GSPLAT_CHECK_INPUT(rays_o);
    GSPLAT_CHECK_INPUT(rays_d);
    GSPLAT_CHECK_INPUT(vertices);

    uint32_t N = rays_o.size(0);
    torch::Tensor entry_face_ids = torch::empty({N}, rays_o.options().dtype(torch::kInt32));
    torch::Tensor exit_face_ids = torch::empty({N}, rays_o.options().dtype(torch::kInt32));
    torch::Tensor t_entrys = torch::empty({N}, rays_o.options());
    torch::Tensor t_exits = torch::empty({N}, rays_o.options());

    if (N > 0) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        ray_tetra_intersection_fwd_kernel<float>
            <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS,
                0,
                stream>>>(
                N,
                rays_o.data_ptr<float>(),
                rays_d.data_ptr<float>(),
                vertices.data_ptr<float>(),
                entry_face_ids.data_ptr<int32_t>(),
                exit_face_ids.data_ptr<int32_t>(),
                t_entrys.data_ptr<float>(),
                t_exits.data_ptr<float>()
            );
    }
    return {entry_face_ids, exit_face_ids, t_entrys, t_exits};
} 

} // namespace gsplat