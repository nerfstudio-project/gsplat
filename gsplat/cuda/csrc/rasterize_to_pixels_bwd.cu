#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Backward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_bwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    // fwd inputs
    const vec3<S> *__restrict__ means2d, // [C, N, 3] or [nnz, 3]
    const S *__restrict__ conics,        // [C, N, 6] or [nnz, 6]
    const S *__restrict__ colors,        // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,     // [C, N] or [nnz]
    const S *__restrict__ backgrounds,   // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const DEPTH_MODE depth_mode,
    // fwd outputs
    const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad outputs
    const S *__restrict__ v_render_colors, // [C, image_height, image_width, COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    const S *__restrict__ v_render_depths, // [C, image_height, image_width, 1] optional
    // grad inputs
    vec3<S> *__restrict__ v_means2d_abs, // [C, N, 3] or [nnz, 3]
    vec3<S> *__restrict__ v_means2d,     // [C, N, 3] or [nnz, 3]
    S *__restrict__ v_conics,            // [C, N, 6] or [nnz, 6]
    S *__restrict__ v_colors,            // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities          // [C, N] or [nnz]
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (depth_mode != DEPTH_MODE::DISABLED) {
        v_render_depths += camera_id * image_height * image_width;
    }
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                         // [block_size]
    vec3<S> *mean2d_batch = (vec3<S> *)&id_batch[block_size]; // [block_size]
    S *opac_batch = (S *)&mean2d_batch[block_size];           // [block_size]
    S *conic_batch = (S *)&opac_batch[block_size];            // [block_size * 6]
    S *rgbs_batch = (S *)&conic_batch[block_size * 6]; // [block_size * COLOR_DIM]

    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    // the contribution from gaussians behind the current one
    S buffer_c[COLOR_DIM] = {0.f};
    S buffer_d = 0.f;
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];
    const S v_render_d =
        depth_mode != DEPTH_MODE::DISABLED ? v_render_depths[pix_id] : 0.f;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec3<S> mean2d = means2d[g];
            const S opac = opacities[g];
            mean2d_batch[tr] = mean2d;
            opac_batch[tr] = opac;
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < 6; ++k) {
                conic_batch[tr * 6 + k] = conics[g * 6 + k];
            }
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            S alpha;
            S opac;
            vec2<S> delta;
            S conic00, conic01, conic02, conic11, conic12, conic22;
            S vis;
            vec3<S> mean2d;

            if (valid) {
                conic00 = conic_batch[t * 6 + 0];
                conic01 = conic_batch[t * 6 + 1];
                conic02 = conic_batch[t * 6 + 2];
                conic11 = conic_batch[t * 6 + 3];
                conic12 = conic_batch[t * 6 + 4];
                conic22 = conic_batch[t * 6 + 5];

                mean2d = mean2d_batch[t];
                opac = opac_batch[t];

                delta = {mean2d.x - px, mean2d.y - py};
                S sigma =
                    0.5f * (conic00 * delta.x * delta.x + conic11 * delta.y * delta.y) +
                    conic01 * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            S v_conic_local[6] = {0.f};
            vec3<S> v_mean2d_local = {0.f, 0.f, 0.f};
            vec3<S> v_mean2d_abs_local = {0.f, 0.f, 0.f};
            S v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const S fac = alpha * T;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                S v_alpha = 0.f;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    S rgb = rgbs_batch[t * COLOR_DIM + k];

                    v_alpha += (rgb * T - buffer_c[k] * ra) * v_render_c[k];
                    buffer_c[k] += rgb * fac;
                }

                // contribution from depth map
                S depth;
                S v_depth;
                switch (depth_mode) {
                case DEPTH_MODE::DISABLED:
                    // do nothing
                    break;
                case DEPTH_MODE::LINEAR:
                    S conic22_inv = 1.f / conic22;
                    depth = mean2d.z +
                            (conic02 * (mean2d.x - px) + conic12 * (mean2d.y - py)) *
                                conic22_inv;

                    v_alpha += (depth * T - buffer_d * ra) * v_render_d;
                    buffer_d += depth * fac;

                    v_depth = fac * v_render_d;

                    v_conic_local[2] += (mean2d.x - px) * conic22_inv * v_depth;
                    v_conic_local[4] += (mean2d.y - py) * conic22_inv * v_depth;
                    v_conic_local[5] += -(depth - mean2d.z) * conic22_inv * v_depth;

                    v_mean2d_local.x += conic02 * conic22_inv * v_depth;
                    v_mean2d_local.y += conic12 * conic22_inv * v_depth;
                    v_mean2d_local.z += v_depth;
                    break;
                case DEPTH_MODE::CONSTANT:
                    depth = mean2d.z;
                    v_alpha += (depth * T - buffer_d * ra) * v_render_d;
                    buffer_d += depth * fac;

                    v_depth = fac * v_render_d;
                    v_mean2d_local.z += v_depth;
                    break;
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    S accum = 0.f;
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const S v_sigma = -opac * vis * v_alpha;

                    v_conic_local[0] += 0.5f * v_sigma * delta.x * delta.x;
                    v_conic_local[1] += v_sigma * delta.x * delta.y;
                    v_conic_local[3] += 0.5f * v_sigma * delta.y * delta.y;

                    v_mean2d_local.x +=
                        v_sigma * (conic00 * delta.x + conic01 * delta.y);
                    v_mean2d_local.y +=
                        v_sigma * (conic01 * delta.x + conic11 * delta.y);

                    v_opacity_local = vis * v_alpha;
                }

                if (v_means2d_abs != nullptr) {
                    v_mean2d_abs_local = {abs(v_mean2d_local.x), abs(v_mean2d_local.y),
                                          abs(v_mean2d_local.z)};
                }
            }
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<6, S>(v_conic_local, warp);
            warpSum<decltype(warp), S>(v_mean2d_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum<decltype(warp), S>(v_mean2d_abs_local, warp);
            }
            warpSum<decltype(warp), S>(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                S *v_conic_ptr = (S *)(v_conics) + 6 * g;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < 6; ++k) {
                    gpuAtomicAdd(v_conic_ptr + k, v_conic_local[k]);
                }

                S *v_means2d_ptr = (S *)(v_means2d) + 3 * g;
                gpuAtomicAdd(v_means2d_ptr, v_mean2d_local.x);
                gpuAtomicAdd(v_means2d_ptr + 1, v_mean2d_local.y);
                gpuAtomicAdd(v_means2d_ptr + 2, v_mean2d_local.z);

                if (v_means2d_abs != nullptr) {
                    S *v_means_abs_ptr = (S *)(v_means2d_abs) + 3 * g;
                    gpuAtomicAdd(v_means_abs_ptr, v_mean2d_abs_local.x);
                    gpuAtomicAdd(v_means_abs_ptr + 1, v_mean2d_abs_local.y);
                    gpuAtomicAdd(v_means_abs_ptr + 2, v_mean2d_abs_local.z);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 3] or [nnz, 3]
    const torch::Tensor &conics,                    // [C, N, 6] or [nnz, 6]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    const at::optional<torch::Tensor>
        &v_render_depths, // [C, image_height, image_width, 1]
    // options
    const bool absgrad, const DEPTH_MODE depth_mode) {

    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    if (depth_mode != DEPTH_MODE::DISABLED) {
        assert(v_render_depths.has_value());
        CHECK_INPUT(v_render_depths.value());
    }
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    if (n_isects) {
        const uint32_t shared_mem = tile_size * tile_size *
                                    (sizeof(int32_t) + sizeof(vec3<float>) +
                                     sizeof(float) * (1 + 6 + COLOR_DIM));

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<CDIM, float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_bwd_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                reinterpret_cast<vec3<float> *>(means2d.data_ptr<float>()),
                conics.data_ptr<float>(), colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                depth_mode, render_alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(), v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                depth_mode != DEPTH_MODE::DISABLED
                    ? v_render_depths.value().data_ptr<float>()
                    : nullptr,
                absgrad
                    ? reinterpret_cast<vec3<float> *>(v_means2d_abs.data_ptr<float>())
                    : nullptr,
                reinterpret_cast<vec3<float> *>(v_means2d.data_ptr<float>()),
                v_conics.data_ptr<float>(), v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>());
    }

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 3] or [nnz, 3]
    const torch::Tensor &conics,    // [C, N, 6] or [nnz, 6]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    // TODO: make it optional
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    const at::optional<torch::Tensor>
        &v_render_depths, // [C, image_height, image_width, 1]
    // options
    const bool absgrad, const DEPTH_MODE depth_mode) {

    CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                                                 \
    case N:                                                                            \
        return call_kernel_with_dim<N>(                                                \
            means2d, conics, colors, opacities, backgrounds, image_width,              \
            image_height, tile_size, tile_offsets, flatten_ids, render_alphas,         \
            last_ids, v_render_colors, v_render_alphas, v_render_depths, absgrad,      \
            depth_mode);

    switch (COLOR_DIM) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
    }
}
