#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_ewa_3dgs_packed_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ covars,   // [N, 6] Optional
    const scalar_t *__restrict__ quats,    // [N, 4] Optional
    const scalar_t *__restrict__ scales,   // [N, 3] Optional
    const scalar_t *__restrict__ opacities, // [N] optional
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const int32_t
        *__restrict__ block_accum, // [C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ block_cnts,    // [C * blocks_per_row] packing helper
    int32_t *__restrict__ indptr,        // [C + 1]
    int64_t *__restrict__ camera_ids,    // [nnz]
    int64_t *__restrict__ gaussian_ids,  // [nnz]
    int32_t *__restrict__ radii,         // [nnz, 2]
    scalar_t *__restrict__ means2d,      // [nnz, 2]
    scalar_t *__restrict__ depths,       // [nnz]
    scalar_t *__restrict__ conics,       // [nnz, 3]
    scalar_t *__restrict__ compensations // [nnz] optional
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

    // check if the perspective projection is valid.
    mat2 covar2d;
    vec2 mean2d;
    mat2 covar2d_inv;
    float compensation;
    float det;
    if (valid) {
        // transform Gaussian covariance to camera space
        mat3 covar;
        if (covars != nullptr) {
            // if a precomputed covariance is provided
            covars += col_idx * 6;
            covar = mat3(
                covars[0],
                covars[1],
                covars[2], // 1st column
                covars[1],
                covars[3],
                covars[4], // 2nd column
                covars[2],
                covars[4],
                covars[5] // 3rd column
            );
        } else {
            // if not then compute it from quaternions and scales
            quats += col_idx * 4;
            scales += col_idx * 3;
            quat_scale_to_covar_preci(
                glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
            );
        }
        mat3 covar_c;
        covarW2C(R, covar, covar_c);

        Ks += row_idx * 9;
        switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        }

        det = add_blur(eps2d, covar2d, compensation);
        if (det <= 0.f) {
            valid = false;
        } else {
            // compute the inverse of the 2d covariance
            covar2d_inv = glm::inverse(covar2d);
        }
    }

    // check if the points are in the image region
    float radius_x, radius_y;
    if (valid) {
        float extend = 3.33f;
        if (opacities != nullptr) {
            float opacity = opacities[col_idx];
            if (compensations != nullptr) {
                // we assume compensation term will be applied later on.
                opacity *= compensation;
            }    
            if (opacity < ALPHA_THRESHOLD) {
                valid = false;
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
        radius_x = ceilf(min(extend * sqrtf(covar2d[0][0]), r1));
        radius_y = ceilf(min(extend * sqrtf(covar2d[1][1]), r1));
        
        if (radius_x <= radius_clip && radius_y <= radius_clip) {
            valid = false;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
            mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
            valid = false;
        }
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
            conics[thread_data * 3] = covar2d_inv[0][0];
            conics[thread_data * 3 + 1] = covar2d_inv[0][1];
            conics[thread_data * 3 + 2] = covar2d_inv[1][1];
            if (compensations != nullptr) {
                compensations[thread_data] = compensation;
            }
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

void launch_projection_ewa_3dgs_packed_fwd_kernel(
    // inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const at::optional<at::Tensor>
        block_accum, // [C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    at::optional<at::Tensor> block_cnts, // [C * blocks_per_row] packing helper
    at::optional<at::Tensor> indptr,     // [C + 1]
    at::optional<at::Tensor> camera_ids, // [nnz]
    at::optional<at::Tensor> gaussian_ids, // [nnz]
    at::optional<at::Tensor> radii,        // [nnz, 2]
    at::optional<at::Tensor> means2d,      // [nnz, 2]
    at::optional<at::Tensor> depths,       // [nnz]
    at::optional<at::Tensor> conics,       // [nnz, 3]
    at::optional<at::Tensor> compensations // [nnz] optional
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

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_packed_fwd_kernel",
        [&]() {
            projection_ewa_3dgs_packed_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    quats.has_value() ? quats.value().data_ptr<scalar_t>()
                                      : nullptr,
                    scales.has_value() ? scales.value().data_ptr<scalar_t>()
                                       : nullptr,
                    opacities.has_value() ? opacities.value().data_ptr<scalar_t>()
                                       : nullptr,
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    block_accum.has_value()
                        ? block_accum.value().data_ptr<int32_t>()
                        : nullptr,
                    camera_model,
                    block_cnts.has_value()
                        ? block_cnts.value().data_ptr<int32_t>()
                        : nullptr,
                    indptr.has_value() ? indptr.value().data_ptr<int32_t>()
                                       : nullptr,
                    camera_ids.has_value()
                        ? camera_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    gaussian_ids.has_value()
                        ? gaussian_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    radii.has_value() ? radii.value().data_ptr<int32_t>()
                                      : nullptr,
                    means2d.has_value() ? means2d.value().data_ptr<scalar_t>()
                                        : nullptr,
                    depths.has_value() ? depths.value().data_ptr<scalar_t>()
                                       : nullptr,
                    conics.has_value() ? conics.value().data_ptr<scalar_t>()
                                       : nullptr,
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr
                );
        }
    );
}

template <typename scalar_t>
__global__ void projection_ewa_3dgs_packed_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ covars,   // [N, 6] Optional
    const scalar_t *__restrict__ quats,    // [N, 4] Optional
    const scalar_t *__restrict__ scales,   // [N, 3] Optional
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const scalar_t eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int64_t *__restrict__ camera_ids,     // [nnz]
    const int64_t *__restrict__ gaussian_ids,   // [nnz]
    const scalar_t *__restrict__ conics,        // [nnz, 3]
    const scalar_t *__restrict__ compensations, // [nnz] optional
    // grad outputs
    const scalar_t *__restrict__ v_means2d,       // [nnz, 2]
    const scalar_t *__restrict__ v_depths,        // [nnz]
    const scalar_t *__restrict__ v_conics,        // [nnz, 3]
    const scalar_t *__restrict__ v_compensations, // [nnz] optional
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    // grad inputs
    scalar_t *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_covars,  // [N, 6] or [nnz, 6] Optional
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

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    mat2 covar2d_inv = mat2(conics[0], conics[1], conics[1], conics[2]);
    mat2 v_covar2d_inv =
        mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

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
    mat3 covar;
    vec4 quat;
    vec3 scale;
    if (covars != nullptr) {
        // if a precomputed covariance is provided
        covars += gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // if not then compute it from quaternions and scales
        quat = glm::make_vec4(quats + gid * 4);
        scale = glm::make_vec3(scales + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3 v_covar_c(0.f);
    vec3 v_mean_c(0.f);
    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    }

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // vjp: transform Gaussian covariance to camera space
    vec3 v_mean(0.f);
    mat3 v_covar(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    posW2C_VJP(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

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
        if (v_covars != nullptr) {
            v_covars += idx * 6;
            v_covars[0] = v_covar[0][0];
            v_covars[1] = v_covar[0][1] + v_covar[1][0];
            v_covars[2] = v_covar[0][2] + v_covar[2][0];
            v_covars[3] = v_covar[1][1];
            v_covars[4] = v_covar[1][2] + v_covar[2][1];
            v_covars[5] = v_covar[2][2];
        } else {
            mat3 rotmat = quat_to_rotmat(quat);
            vec4 v_quat(0.f);
            vec3 v_scale(0.f);
            quat_scale_to_covar_vjp(
                quat, scale, rotmat, v_covar, v_quat, v_scale
            );
            v_quats += idx * 4;
            v_scales += idx * 3;
            v_quats[0] = v_quat[0];
            v_quats[1] = v_quat[1];
            v_quats[2] = v_quat[2];
            v_quats[3] = v_quat[3];
            v_scales[0] = v_scale[0];
            v_scales[1] = v_scale[1];
            v_scales[2] = v_scale[2];
        }
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
        if (v_covars != nullptr) {
            // Directly output gradients w.r.t. the covariance
            warpSum(v_covar, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_covars += gid * 6;
                gpuAtomicAdd(v_covars, v_covar[0][0]);
                gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
                gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
                gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
                gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
                gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
            }
        } else {
            // Directly output gradients w.r.t. the quaternion and scale
            mat3 rotmat = quat_to_rotmat(quat);
            vec4 v_quat(0.f);
            vec3 v_scale(0.f);
            quat_scale_to_covar_vjp(
                quat, scale, rotmat, v_covar, v_quat, v_scale
            );
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
        }
    }
    // v_viewmats is always in dense layout
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

void launch_projection_ewa_3dgs_packed_bwd_kernel(
    // fwd inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6]
    const at::optional<at::Tensor> quats,  // [N, 4]
    const at::optional<at::Tensor> scales, // [N, 3]
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor camera_ids,                  // [nnz]
    const at::Tensor gaussian_ids,                // [nnz]
    const at::Tensor conics,                      // [nnz, 3]
    const at::optional<at::Tensor> compensations, // [nnz] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [nnz, 2]
    const at::Tensor v_depths,                      // [nnz]
    const at::Tensor v_conics,                      // [nnz, 3]
    const at::optional<at::Tensor> v_compensations, // [nnz] optional
    const bool sparse_grad,
    // grad inputs
    at::Tensor v_means,                 // [N, 3] or [nnz, 3]
    at::optional<at::Tensor> v_covars,  // [N, 6] or [nnz, 6] Optional
    at::optional<at::Tensor> v_quats,   // [N, 4] or [nnz, 4] Optional
    at::optional<at::Tensor> v_scales,  // [N, 3] or [nnz, 3] Optional
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

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_packed_bwd_kernel",
        [&]() {
            projection_ewa_3dgs_packed_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    nnz,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr
                                       : quats.value().data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : scales.value().data_ptr<scalar_t>(),
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    camera_model,
                    camera_ids.data_ptr<int64_t>(),
                    gaussian_ids.data_ptr<int64_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_means2d.data_ptr<scalar_t>(),
                    v_depths.data_ptr<scalar_t>(),
                    v_conics.data_ptr<scalar_t>(),
                    v_compensations.has_value()
                        ? v_compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    sparse_grad,
                    v_means.data_ptr<scalar_t>(),
                    covars.has_value() ? v_covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr
                                       : v_quats.value().data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : v_scales.value().data_ptr<scalar_t>(),
                    v_viewmats.has_value()
                        ? v_viewmats.value().data_ptr<scalar_t>()
                        : nullptr
                );
        }
    );
}

} // namespace gsplat