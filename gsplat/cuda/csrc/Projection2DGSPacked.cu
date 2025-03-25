#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Projection.h"
#include "Projection2DGS.cuh" // Utils for 2DGS Projection
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_2dgs_packed_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ quats,    // [N, 4]
    const scalar_t *__restrict__ scales,   // [N, 3]
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const scalar_t near_plane,
    const scalar_t far_plane,
    const scalar_t radius_clip,
    const int32_t
        *__restrict__ block_accum,    // [C * blocks_per_row] packing helper
    int32_t *__restrict__ block_cnts, // [C * blocks_per_row] packing helper
    // outputs
    int32_t *__restrict__ indptr,          // [C + 1]
    int64_t *__restrict__ camera_ids,      // [nnz]
    int64_t *__restrict__ gaussian_ids,    // [nnz]
    int32_t *__restrict__ radii,           // [nnz, 2]
    scalar_t *__restrict__ means2d,        // [nnz, 2]
    scalar_t *__restrict__ depths,         // [nnz]
    scalar_t *__restrict__ ray_transforms, // [nnz, 3, 3]
    scalar_t *__restrict__ normals         // [nnz, 3]
) {
    int32_t blocks_per_row = gridDim.x;

    int32_t row_idx = blockIdx.y; // cid
    int32_t block_col_idx = blockIdx.x;
    int32_t block_idx = row_idx * blocks_per_row + block_col_idx;

    int32_t col_idx = block_col_idx * blockDim.x + threadIdx.x; // gid

    bool valid = (row_idx < C) && (col_idx < N);

    // check if points are with camera near and far plane
    vec3 mean_c;
    mat3 R;
    if (valid) {
        // shift pointers to the current camera and gaussian
        means += col_idx * 3;
        viewmats += row_idx * 16;

        // glm is column-major but input is row-major
        R = mat3(
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
        posW2C(R, t, glm::make_vec3(means), mean_c);
        if (mean_c.z < near_plane || mean_c.z > far_plane) {
            valid = false;
        }
    }

    vec2 mean2d;
    mat3 M;
    float radius_x, radius_y;
    vec3 normal;
    if (valid) {
        // build ray transformation matrix and transform from world space to
        // camera space
        quats += col_idx * 4;
        scales += col_idx * 3;
        Ks += row_idx * 9;

        mat3 RS_camera =
            R * quat_to_rotmat(glm::make_vec4(quats)) *
            mat3(scales[0], 0.0, 0.0, 0.0, scales[1], 0.0, 0.0, 0.0, 1.0);
        ;
        mat3 WH = mat3(RS_camera[0], RS_camera[1], mean_c);

        mat3 world_2_pix =
            mat3(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);
        M = glm::transpose(WH) * world_2_pix;

        // compute AABB
        const vec3 M0 = vec3(M[0][0], M[0][1], M[0][2]);
        const vec3 M1 = vec3(M[1][0], M[1][1], M[1][2]);
        const vec3 M2 = vec3(M[2][0], M[2][1], M[2][2]);

        const vec3 temp_point = vec3(1.0f, 1.0f, -1.0f);
        const float distance = sum(temp_point * M2 * M2);

        if (distance == 0.0f)
            valid = false;

        const vec3 f = (1 / distance) * temp_point;
        mean2d = vec2(sum(f * M0 * M2), sum(f * M1 * M2));

        const vec2 temp = {sum(f * M0 * M0), sum(f * M1 * M1)};
        const vec2 half_extend = mean2d * mean2d - temp;
        radius_x = ceil(3.33f * sqrt(max(1e-4, half_extend.x)));
        radius_y = ceil(3.33f * sqrt(max(1e-4, half_extend.y)));

        if (radius_x <= radius_clip && radius_y <= radius_clip) {
            valid = false;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
            mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
            valid = false;
        }

        // normal dual visible
        normal = RS_camera[2];
        float multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1;
        normal *= multipler;
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        // First pass: compute the block-wide sum
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, N_THREADS_PACKED> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        } else {
            aggregate = 0;
        }
        if (threadIdx.x == 0) {
            block_cnts[block_idx] = aggregate;
        }
    } else {
        // Second pass: write out the indices of the non zero elements
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, N_THREADS_PACKED> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if (valid) {
            if (block_idx > 0) {
                int32_t offset = block_accum[block_idx - 1];
                thread_data += offset;
            }
            // write to outputs
            camera_ids[thread_data] = row_idx;   // cid
            gaussian_ids[thread_data] = col_idx; // gid
            radii[thread_data * 2] = (int32_t)radius_x;
            radii[thread_data * 2 + 1] = (int32_t)radius_y;
            means2d[thread_data * 2] = mean2d.x;
            means2d[thread_data * 2 + 1] = mean2d.y;
            depths[thread_data] = mean_c.z;
            ray_transforms[thread_data * 9] = M[0][0];
            ray_transforms[thread_data * 9 + 1] = M[0][1];
            ray_transforms[thread_data * 9 + 2] = M[0][2];
            ray_transforms[thread_data * 9 + 3] = M[1][0];
            ray_transforms[thread_data * 9 + 4] = M[1][1];
            ray_transforms[thread_data * 9 + 5] = M[1][2];
            ray_transforms[thread_data * 9 + 6] = M[2][0];
            ray_transforms[thread_data * 9 + 7] = M[2][1];
            ray_transforms[thread_data * 9 + 8] = M[2][2];
            normals[thread_data * 3] = normal.x;
            normals[thread_data * 3 + 1] = normal.y;
            normals[thread_data * 3 + 2] = normal.z;
        }
        // lane 0 of the first block in each row writes the indptr
        if (threadIdx.x == 0 && block_col_idx == 0) {
            if (row_idx == 0) {
                indptr[0] = 0;
                indptr[C] = block_accum[C * blocks_per_row - 1];
            } else {
                indptr[row_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

void launch_projection_2dgs_packed_fwd_kernel(
    // inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const at::optional<at::Tensor>
        block_accum, // [C * blocks_per_row] packing helper
    // outputs
    at::optional<at::Tensor> block_cnts, // [C * blocks_per_row] packing helper
    at::optional<at::Tensor> indptr,     // [C + 1]
    at::optional<at::Tensor> camera_ids, // [nnz]
    at::optional<at::Tensor> gaussian_ids,   // [nnz]
    at::optional<at::Tensor> radii,          // [nnz, 2]
    at::optional<at::Tensor> means2d,        // [nnz, 2]
    at::optional<at::Tensor> depths,         // [nnz]
    at::optional<at::Tensor> ray_transforms, // [nnz, 3, 3]
    at::optional<at::Tensor> normals         // [nnz]
) {
    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    uint32_t nrows = C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    dim3 threads(N_THREADS_PACKED);
    // limit on the number of blocks: [2**31 - 1, 65535, 65535]
    dim3 grid(blocks_per_row, nrows, 1);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (N == 0 || C == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    projection_2dgs_packed_fwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            near_plane,
            far_plane,
            radius_clip,
            block_accum.has_value() ? block_accum.value().data_ptr<int32_t>()
                                    : nullptr,
            block_cnts.has_value() ? block_cnts.value().data_ptr<int32_t>()
                                   : nullptr,
            indptr.has_value() ? indptr.value().data_ptr<int32_t>() : nullptr,
            camera_ids.has_value() ? camera_ids.value().data_ptr<int64_t>()
                                   : nullptr,
            gaussian_ids.has_value() ? gaussian_ids.value().data_ptr<int64_t>()
                                     : nullptr,
            radii.has_value() ? radii.value().data_ptr<int32_t>() : nullptr,
            means2d.has_value() ? means2d.value().data_ptr<float>() : nullptr,
            depths.has_value() ? depths.value().data_ptr<float>() : nullptr,
            ray_transforms.has_value()
                ? ray_transforms.value().data_ptr<float>()
                : nullptr,
            normals.has_value() ? normals.value().data_ptr<float>() : nullptr
        );
}

template <typename scalar_t>
__global__ void projection_2dgs_packed_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ quats,    // [N, 4]
    const scalar_t *__restrict__ scales,   // [N, 3]
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    // fwd outputs
    const int64_t *__restrict__ camera_ids,      // [nnz]
    const int64_t *__restrict__ gaussian_ids,    // [nnz]
    const scalar_t *__restrict__ ray_transforms, // [nnz, 3]
    // grad outputs
    const scalar_t *__restrict__ v_means2d,        // [nnz, 2]
    const scalar_t *__restrict__ v_depths,         // [nnz]
    const scalar_t *__restrict__ v_ray_transforms, // [nnz, 3, 3]
    const scalar_t *__restrict__ v_normals,        // [nnz, 3]
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    // grad inputs
    scalar_t *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_quats,   // [N, 4] or [nnz, 4] Optional
    scalar_t *__restrict__ v_scales,  // [N, 3] or [nnz, 3] Optional
    scalar_t *__restrict__ v_viewmats // [C, 4, 4] Optional
) {
    // parallelize over nnz.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= nnz) {
        return;
    }
    const int64_t cid = camera_ids[idx];   // camera id
    const int64_t gid = gaussian_ids[idx]; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    ray_transforms += idx * 9;

    v_means2d += idx * 2;
    v_normals += idx * 3;
    v_depths += idx;
    v_ray_transforms += idx * 9;

    // transform Gaussian to camera space
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
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);

    vec4 quat = glm::make_vec4(quats + gid * 4);
    vec2 scale = glm::make_vec2(scales + gid * 3);
    mat3 P = mat3(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);

    mat3 _v_ray_transforms = mat3(
        v_ray_transforms[0],
        v_ray_transforms[1],
        v_ray_transforms[2],
        v_ray_transforms[3],
        v_ray_transforms[4],
        v_ray_transforms[5],
        v_ray_transforms[6],
        v_ray_transforms[7],
        v_ray_transforms[8]
    );

    _v_ray_transforms[2][2] += v_depths[0];

    vec3 v_normal = glm::make_vec3(v_normals);

    vec3 v_mean(0.f);
    vec2 v_scale(0.f);
    vec4 v_quat(0.f);
    compute_ray_transforms_aabb_vjp(
        ray_transforms,
        v_means2d,
        v_normal,
        R,
        P,
        t,
        mean_c,
        quat,
        scale,
        _v_ray_transforms,
        v_quat,
        v_scale,
        v_mean
    );

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    if (sparse_grad) {
        // write out results with sparse layout
        if (v_means != nullptr) {
            v_means += idx * 3;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                v_means[i] = v_mean[i];
            }
        }
        v_quats += idx * 4;
        v_scales += idx * 3;
        v_quats[0] = v_quat[0];
        v_quats[1] = v_quat[1];
        v_quats[2] = v_quat[2];
        v_quats[3] = v_quat[3];
        v_scales[0] = v_scale[0];
        v_scales[1] = v_scale[1];
    } else {
        // write out results with dense layout
        // #if __CUDA_ARCH__ >= 700
        // write out results with warp-level reduction
        auto warp_group_g = cg::labeled_partition(warp, gid);
        if (v_means != nullptr) {
            warpSum(v_mean, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_means += gid * 3;
#pragma unroll
                for (uint32_t i = 0; i < 3; i++) {
                    gpuAtomicAdd(v_means + i, v_mean[i]);
                }
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
        }
    }
}

void launch_projection_2dgs_packed_bwd_kernel(
    // fwd inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor camera_ids,     // [nnz]
    const at::Tensor gaussian_ids,   // [nnz]
    const at::Tensor ray_transforms, // [nnz, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [nnz, 2]
    const at::Tensor v_depths,         // [nnz]
    const at::Tensor v_ray_transforms, // [nnz, 3, 3]
    const at::Tensor v_normals,        // [nnz, 3]
    const bool sparse_grad,
    // grad inputs
    at::Tensor v_means,                 // [N, 3] or [nnz, 3]
    at::Tensor v_quats,                 // [N, 4] or [nnz, 4]
    at::Tensor v_scales,                // [N, 3] or [nnz, 3]
    at::optional<at::Tensor> v_viewmats // [C, 4, 4] Optional
) {
    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    uint32_t nnz = camera_ids.size(0);

    dim3 threads(256);
    dim3 grid((nnz + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (nnz == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    projection_2dgs_packed_bwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            nnz,
            means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            camera_ids.data_ptr<int64_t>(),
            gaussian_ids.data_ptr<int64_t>(),
            ray_transforms.data_ptr<float>(),
            v_means2d.data_ptr<float>(),
            v_depths.data_ptr<float>(),
            v_ray_transforms.data_ptr<float>(),
            v_normals.data_ptr<float>(),
            sparse_grad,
            v_means.data_ptr<float>(),
            v_quats.data_ptr<float>(),
            v_scales.data_ptr<float>(),
            v_viewmats.has_value() ? v_viewmats.value().data_ptr<float>()
                                   : nullptr
        );
}

} // namespace gsplat