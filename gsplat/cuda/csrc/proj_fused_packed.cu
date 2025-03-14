#include "proj_fused.h"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <ATen/cuda/Atomic.cuh>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

/****************************************************************************
 * Fused Projection Forward Pass (Packed)
 ****************************************************************************/

__global__ void proj_fused_packed_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] Optional
    const float *__restrict__ quats,    // [N, 4] Optional
    const float *__restrict__ scales,   // [N, 3] Optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const int32_t *__restrict__ block_accum,    // [C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ block_cnts, // [C * blocks_per_row] packing helper
    int32_t *__restrict__ indptr,       // [C + 1]
    int64_t *__restrict__ camera_ids,   // [nnz]
    int64_t *__restrict__ gaussian_ids, // [nnz]
    int32_t *__restrict__ radii,        // [nnz]
    float *__restrict__ means2d,            // [nnz, 2]
    float *__restrict__ depths,             // [nnz]
    float *__restrict__ conics,             // [nnz, 3]
    float *__restrict__ compensations       // [nnz] optional
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
    float radius;
    if (valid) {
        // take 3 sigma as the radius (non differentiable)
        float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        float v1 = b + sqrt(max(0.1f, b * b - det));
        float v2 = b - sqrt(max(0.1f, b * b - det));
        radius = ceil(3.f * sqrt(max(v1, v2)));

        if (radius <= radius_clip) {
            valid = false;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
            mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
            valid = false;
        }
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        // First pass: compute the block-wide sum
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, N_THREADS> BlockReduce;
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
            typedef cub::BlockScan<int32_t, N_THREADS> BlockScan;
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
            radii[thread_data] = (int32_t)radius;
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


void proj_fused_packed_fwd_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    dim3 blocks,
    dim3 threads,
    // args
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] Optional
    const float *__restrict__ quats,    // [N, 4] Optional
    const float *__restrict__ scales,   // [N, 3] Optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const int32_t *__restrict__ block_accum,    // [C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    int32_t *__restrict__ block_cnts, // [C * blocks_per_row] packing helper
    int32_t *__restrict__ indptr,       // [C + 1]
    int64_t *__restrict__ camera_ids,   // [nnz]
    int64_t *__restrict__ gaussian_ids, // [nnz]
    int32_t *__restrict__ radii,        // [nnz]
    float *__restrict__ means2d,            // [nnz, 2]
    float *__restrict__ depths,             // [nnz]
    float *__restrict__ conics,             // [nnz, 3]
    float *__restrict__ compensations       // [nnz] optional
) {
    proj_fused_packed_fwd_kernel<<<blocks, threads, shmem_size, stream>>>(
        C,
        N,
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        block_accum,
        camera_model,
        block_cnts,
        indptr,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations
    );
}

/****************************************************************************
 * Fused Projection Backward Pass (Packed)
 ****************************************************************************/


__global__ void proj_fused_packed_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] Optional
    const float *__restrict__ quats,    // [N, 4] Optional
    const float *__restrict__ scales,   // [N, 3] Optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int64_t *__restrict__ camera_ids,   // [nnz]
    const int64_t *__restrict__ gaussian_ids, // [nnz]
    const float *__restrict__ conics,             // [nnz, 3]
    const float *__restrict__ compensations,      // [nnz] optional
    // grad outputs
    const float *__restrict__ v_means2d,       // [nnz, 2]
    const float *__restrict__ v_depths,        // [nnz]
    const float *__restrict__ v_conics,        // [nnz, 3]
    const float *__restrict__ v_compensations, // [nnz] optional
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    // grad inputs
    float *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    float *__restrict__ v_covars,  // [N, 6] or [nnz, 6] Optional
    float *__restrict__ v_quats,   // [N, 4] or [nnz, 4] Optional
    float *__restrict__ v_scales,  // [N, 3] or [nnz, 3] Optional
    float *__restrict__ v_viewmats // [C, 4, 4] Optional
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
    posW2C_VJP(
        R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean
    );
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

void proj_fused_packed_bwd_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] Optional
    const float *__restrict__ quats,    // [N, 4] Optional
    const float *__restrict__ scales,   // [N, 3] Optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    const int64_t *__restrict__ camera_ids,   // [nnz]
    const int64_t *__restrict__ gaussian_ids, // [nnz]
    const float *__restrict__ conics,             // [nnz, 3]
    const float *__restrict__ compensations,      // [nnz] optional
    const float *__restrict__ v_means2d,       // [nnz, 2]
    const float *__restrict__ v_depths,        // [nnz]
    const float *__restrict__ v_conics,        // [nnz, 3]
    const float *__restrict__ v_compensations, // [nnz] optional
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    float *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    float *__restrict__ v_covars,  // [N, 6] or [nnz, 6] Optional
    float *__restrict__ v_quats,   // [N, 4] or [nnz, 4] Optional
    float *__restrict__ v_scales,  // [N, 3] or [nnz, 3] Optional
    float *__restrict__ v_viewmats // [C, 4, 4] Optional
) {
    if (n_elements <= 0) {
        return;
    }
    proj_fused_packed_bwd_kernel<<<n_blocks_linear(n_elements), N_THREADS, shmem_size, stream>>>(
        C,
        N,
        nnz,
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        camera_model,
        camera_ids,
        gaussian_ids,
        conics,
        compensations,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        sparse_grad,
        v_means,
        v_covars,
        v_quats,
        v_scales,
        v_viewmats
    );
}