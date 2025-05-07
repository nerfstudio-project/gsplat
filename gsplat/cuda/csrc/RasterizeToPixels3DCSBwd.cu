#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

#define MAX_NB_POINTS 8

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dcs_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [C, N, 2] or [nnz, 2]
    const scalar_t *__restrict__ normals, // [C, total_nb_points, 2] or [nnz*6, 2]
    const scalar_t *__restrict__ offsets,       // [C, total_nb_points] or [nnz*6]
    const uint32_t  num_points_per_convex, // [K]
    const scalar_t *__restrict__ delta,                       // [C, N]
    const scalar_t *__restrict__ sigma,                       // [C, N]
    const int32_t *__restrict__ num_points_per_convex_view, // [C, N]
    const int32_t *__restrict__ cumsum_of_points_per_convex, // [C, N]
    const scalar_t *__restrict__ depths,                            // [C, N]
    const vec3 *__restrict__ conics,          // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [C, CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [C, image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [C, N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [C, N, 2] or [nnz, 2]
    scalar_t *__restrict__ v_normals,           // [C, total_nb_points, 2] or [nnz*6, 2]
    scalar_t *__restrict__ v_offsets,           // [C, k] or [nnz*6]
    scalar_t *__restrict__ v_delta,              // [C, N]
    scalar_t *__restrict__ v_sigma,              // [C, N]
    vec3 *__restrict__ v_conics,       // [C, N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [C, N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities // [C, N] or [nnz]
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * CDIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * CDIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

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
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *rgbs_batch =
        (float *)&conic_batch[block_size]; // [block_size * CDIM]

    // 3DCS part
    scalar_t *normals_batch = (scalar_t *)&rgbs_batch[block_size*CDIM];
    // FIXME: Should offset be a vec3?
    scalar_t *offsets_batch = (scalar_t *)&normals_batch[block_size*MAX_NB_POINTS*2];
    int32_t *num_points_per_convex_view_batch = (int32_t *)&offsets_batch[block_size*MAX_NB_POINTS];
    //int32_t *cumsum_of_points_per_convex_batch = reinterpret_cast<int32_t *>(&num_points_per_convex_view_batch[block_size]);
    scalar_t *delta_batch = (scalar_t *)&num_points_per_convex_view_batch[block_size];
    scalar_t *sigma_batch = (scalar_t *)&delta_batch[block_size];
    scalar_t *depths_batch = (scalar_t *)&sigma_batch[block_size];

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b)
    {
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
        if (idx >= range_start)
        {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }

            // 3DCS part
            num_points_per_convex_view_batch[tr] = num_points_per_convex_view[g];
            delta_batch[tr] = delta[g];
            sigma_batch[tr] = sigma[g];
            depths_batch[tr] = depths[g];
#pragma unroll
            for (uint32_t k = 0; k < num_points_per_convex_view[g]; k++)
            {
                normals_batch[6*tr*2 + 2*k] = normals[6*g*2 + 2*k];
            }
#pragma unroll
            for (uint32_t k = 0; k < num_points_per_convex_view[g]; k++)
            {
                normals_batch[6*tr*2 + 2*k + 1] = normals[6*g*2 + 2*k + 1];
            }
#pragma unroll
            for (uint32_t k = 0; k < num_points_per_convex_view[g]; k++)
            {
                offsets_batch[6*tr + k] = offsets[6*g + k];
            }
            
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t)
        {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;

            // Propagate gradients to per-Convex colors and keep
            // gradients w.r.t. alpha (blending factor for a Convex/pixel
            // pair).
            float distances[MAX_NB_POINTS];
            float max_val = -INFINITY;
            float Cx = 0.0f;
            float phi_x = 0.0f;
            float sum_exp = 0.0f;

            if (valid)
            {
                vec3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;

                // 3DCS part.
                for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                {
                    distances[k] = normals_batch[6*t*2 + 2*k] * px + normals_batch[6*t*2 + 2*k + 1] * py + offsets_batch[6*t + k];

                    if (distances[k] > max_val)
                    {
                        max_val = distances[k];
                    }
                }

#pragma unroll
                for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                {
                    sum_exp += __expf(depths_batch[t] * delta_batch[t] * (distances[k]-max_val));
                }

                phi_x = depths_batch[t] * delta_batch[t]*max_val + __logf(sum_exp);

                Cx = 1.0f / (1.0f + __expf(depths_batch[t] * sigma_batch[t] * phi_x));

                alpha = min(0.999f, opac * Cx);
                if (alpha < ALPHA_THRESHOLD)
                {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid))
            {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;

            float v_delta_value_aux = 0.0f;
            float v_sigma_value = 0.0f;

            float v_normal_local[MAX_NB_POINTS*2] = {0.f};
            float v_offset_local[MAX_NB_POINTS]= {0.f};

            // initialize everything to 0, only set if the lane is valid
            if (valid)
            {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * Cx <= 0.999f)
                {
                    const float alpha = min(0.99f, opac * Cx);
                    v_opacity_local = Cx * v_alpha;

                    // Helpful reusable temporary variables
                    float v_C = opac * v_alpha;

                    // Calculate gradients w.r.t sigma
                    v_sigma_value = -depths_batch[t] * phi_x * Cx * (1.0f - Cx) * v_C;   // remove depth here
                    
                    // Calculate gradient w.r.t phi_x
                    float v_phi_x = -sigma_batch[t]  * depths_batch[t] * Cx * (1.0f - Cx) * v_C;  // remove depth here

                    // Calculate gradients with respect to distances
                    float v_distances[MAX_NB_POINTS];
                    for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                    {
                        float exp_val = __expf(depths_batch[t] * delta_batch[t] * (distances[k]-max_val));
                        v_distances[k] = (exp_val / sum_exp) * v_phi_x * delta_batch[t] * depths_batch[t];
                    }

                    for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                    {
                        v_normal_local[2*k] = v_distances[k] * px;
                        v_normal_local[2*k + 1] = v_distances[k] * py;
                    }

                    for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                    {
                        v_offset_local[k] = v_distances[k];
                    }

                    // Gradient with respect to delta
                    float v_delta_value = 0.0f;

                    for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                    {
                        float exp_val = __expf(delta_batch[t] * depths_batch[t] * (distances[k]-max_val));
                        v_delta_value += depths_batch[t] * (distances[k]-max_val) * exp_val / sum_exp;
                    }

                    // Multiply by the chain rule term v_phi_x
                    v_delta_value_aux = (depths_batch[t] * max_val + v_delta_value) * v_phi_x;

                }

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k)
                {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_opacity_local, warp);
            warpSum(v_sigma_value, warp);
            warpSum(v_delta_value_aux, warp);
            warpSum<MAX_NB_POINTS*2>(v_normal_local, warp);
            warpSum<MAX_NB_POINTS>(v_offset_local, warp);
            if (warp.thread_rank() == 0)
            {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k)
                {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);

                // Apply the gradient update to sigma
                gpuAtomicAdd(v_sigma + g, v_sigma_value);

                // Apply the gradient update to delta
                gpuAtomicAdd(v_delta + g, v_delta_value_aux);

                // Calculate gradients w.r.t normals and offsets
                float *v_normals_ptr = (float *)(v_normals) + 2*g*6;
#pragma unroll
                for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                {
                    gpuAtomicAdd(v_normals_ptr + 2*k, v_normal_local[2*k]);
                }

#pragma unroll
                for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                {
                    gpuAtomicAdd(v_normals_ptr + 2*k + 1, v_normal_local[2*k+1]);
                }

                float *v_offsets_ptr = (float *)(v_offsets) + g*6;
#pragma unroll
                for (uint32_t k = 0; k < num_points_per_convex_view_batch[t]; k++)
                {
                    gpuAtomicAdd(v_offsets_ptr + k, v_offset_local[k]);
                }
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dcs_bwd_kernel(
    // 3D convex parameters
    const at::Tensor means2d,                   // [C, N, 2] or [nnz, 2]
    const at::Tensor normals,                   // [C, total_nb_points, 2] or [nnz, 2]
    const at::Tensor offsets,                   // [C, total_nb_points] or [nnz]
    const uint32_t num_points_per_convex,           // 6
    const at::Tensor delta,                     // [C, N]
    const at::Tensor sigma,                     // [C, N]
    const at::Tensor num_points_per_convex_view, // [C, N]
    const at::Tensor cumsum_of_points_per_convex, // [C, N]
    const at::Tensor depths,                    // [C, N]
    const at::Tensor conics,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, 3]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [C, image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [C, N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [C, N, 2] or [nnz, 2]
    at::Tensor v_normals,                   
    at::Tensor v_offsets,
    at::Tensor v_delta,                  
    at::Tensor v_sigma,                  
    at::Tensor v_conics,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [C, N] or [nnz]
) {
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of 3D convexes
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) +
             sizeof(float) * CDIM + sizeof(float)*2*MAX_NB_POINTS + sizeof(float)*MAX_NB_POINTS +
             sizeof(int32_t) /*+ sizeof(int32_t)*/ + sizeof(float) + sizeof(float) + sizeof(float));

    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dcs_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dcs_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            normals.data_ptr<float>(),
            offsets.data_ptr<float>(),
            num_points_per_convex,
            delta.data_ptr<float>(),
            sigma.data_ptr<float>(),
            num_points_per_convex_view.data_ptr<int32_t>(),
            cumsum_of_points_per_convex.data_ptr<int32_t>(),
            depths.data_ptr<float>(),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            v_normals.data_ptr<float>(),
            v_offsets.data_ptr<float>(),
            v_delta.data_ptr<float>(),
            v_sigma.data_ptr<float>(),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dcs_bwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor normals,                                              \
        const at::Tensor offsets,                                              \
        const uint32_t num_points_per_convex,                                  \
        const at::Tensor delta,                                                \
        const at::Tensor sigma,                                                \
        const at::Tensor num_points_per_convex_view,                          \
        const at::Tensor cumsum_of_points_per_convex,                         \
        const at::Tensor depths,                                               \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        at::optional<at::Tensor> v_means2d_abs,                                \
        at::Tensor v_means2d,                                                  \
        at::Tensor v_normals,                                                  \
        at::Tensor v_offsets,                                                  \
        at::Tensor v_delta,                                                    \
        at::Tensor v_sigma,                                                    \
        at::Tensor v_conics,                                                   \
        at::Tensor v_colors,                                                   \
        at::Tensor v_opacities                                                 \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)
#undef __INS__

} // namespace gsplat
