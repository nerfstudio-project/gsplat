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
__global__ void raytracing_to_pixels_bwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    // fwd inputs
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,        // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,     // [C, N] or [nnz]
    const vec10<S> *__restrict__ view2gaussians, // [C, N, 10] or [nnz, 10]
    const S *__restrict__ Ks,                     // [C, 3, 3]
    const S *__restrict__ backgrounds,   // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad outputs
    const S *__restrict__ v_render_colors, // [C, image_height, image_width,
                                           // COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    vec2<S> *__restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
    vec2<S> *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    vec3<S> *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    S *__restrict__ v_colors,            // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities,         // [C, N] or [nnz]
    S *__restrict__ v_view2gaussians     // [C, N, 10] or [nnz, 10]
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
    Ks += camera_id * 9;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    const S focal_x = Ks[0];
    const S focal_y = Ks[4];
    const S cx = Ks[2];
    const S cy = Ks[5];
    const vec3<S> ray = {(px - cx) / focal_x, (py - cy) / focal_y, 1.0};

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
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
    vec10<S> *view2gaussian_batch =
        reinterpret_cast<vec10<float> *>(&conic_batch[block_size]); // [block_size]
    S *rgbs_batch = (S *)&view2gaussian_batch[block_size]; // [block_size * COLOR_DIM]
    

    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    // the contribution from gaussians behind the current one
    S buffer[COLOR_DIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];

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
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            view2gaussian_batch[tr] = view2gaussians[g];
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
            vec3<S> conic;
            const vec10<S> view2gaussian = view2gaussian_batch[t];
            vec3<S> normal;
            S vis;
            S AA, BB, CC, depth, min_value, power;

            if (valid) {
                conic = conic_batch[t];
                vec3<S> xy_opac = xy_opacity_batch[t];
                
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                // S sigma =
                //     0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                //     conic.y * delta.x * delta.y;
                // vis = __expf(-sigma);
                // alpha = min(0.999f, opac * vis);

                normal = {
                    view2gaussian[0] * ray.x + view2gaussian[1] * ray.y + view2gaussian[2], 
                    view2gaussian[1] * ray.x + view2gaussian[3] * ray.y + view2gaussian[4],
                    view2gaussian[2] * ray.x + view2gaussian[4] * ray.y + view2gaussian[5]
                };
            
                // use AA, BB, CC so that the name is unique
                AA = ray.x * normal[0] + ray.y * normal[1] + normal[2];
                BB = 2 * (view2gaussian[6] * ray.x + view2gaussian[7] * ray.y + view2gaussian[8]);
                CC = view2gaussian[9];
                
                // t is the depth of the gaussian
                depth = -BB/(2*AA);
                //TODO take near plane as input
                #define NEAR_PLANE 0.2
                // depth must be positive otherwise it is not valid and we skip it
                if (depth <= NEAR_PLANE)
                    valid = false;

                // the scale of the gaussian is 1.f / sqrt(AA)
			    min_value = -(BB/AA) * (BB/4.) + CC;

                power = -0.5f * min_value;
                if (power > 0.0f){
                    power = 0.0f;
                }
                
                vis = exp(power);
                alpha = min(0.999f, opac * vis);

                if (alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            S v_opacity_local = 0.f;
            S v_view2gaussian_local[10] = {0.f};
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
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_alpha += (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
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
                // here is different to 3DGS, in 3DGS the gradient is computed even if opac * vis > 0.999f
                if (opac * vis <= 0.999f) {
                    const S dL_dG = opac * v_alpha;
                    const S v_sigma = -opac * vis * v_alpha;
                    v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                  v_sigma * (conic.y * delta.x + conic.z * delta.y)};

                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;

                    // gradient of depth
                    S dL_dt = 0.0f;
                    S dL_dnormal[3] = {0.0f, 0.0f, 0.0f};

                    // vis = exp(power);
                    const S dG_dpower = vis;
                    const S dL_dpower = dL_dG * dG_dpower;

                    // // float power = -0.5f * min_value;
                    const S dL_dmin_value = dL_dpower * -0.5f;
                    // float min_value = -(BB*BB)/(4*AA) + CC;
                    // const float dL_dA = dL_dmin_value * (BB*BB)/4 *  1. / (AA*AA);
                    S dL_dA = dL_dmin_value * (BB / AA) * (BB / AA) / 4.f;
                    S dL_dB = dL_dmin_value * -BB / (2 *AA);
                    S dL_dC = dL_dmin_value * 1.0f;
                    // from depth = -BB/(2*AA)
                    dL_dA += dL_dt * BB / (2 * AA * AA);
                    dL_dB += dL_dt * -1.f / (2 * AA);

                    // const float normal[3] = { view2gaussian_j[0] * ray.x + view2gaussian_j[1] * ray.y + view2gaussian_j[2], 
                    // 						view2gaussian_j[1] * ray.x + view2gaussian_j[3] * ray.y + view2gaussian_j[4],
                    // 						view2gaussian_j[2] * ray.x + view2gaussian_j[4] * ray.y + view2gaussian_j[5]};

                    // use AA, BB, CC so that the name is unique
                    // float AA = ray.x * normal[0] + ray.y * normal[1] + normal[2];
                    // float BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
                    // float CC = view2gaussian_j[9];
                    dL_dnormal[0] += dL_dA * ray.x;
                    dL_dnormal[1] += dL_dA * ray.y;
                    dL_dnormal[2] += dL_dA;

                    // write the gradients to global memory directly
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 0]), dL_dnormal[0] * ray.x);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 1]), dL_dnormal[0] * ray.y + dL_dnormal[1] * ray.x);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 2]), dL_dnormal[0] + dL_dnormal[2] * ray.x);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 3]), dL_dnormal[1] * ray.y);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 4]), dL_dnormal[1] + dL_dnormal[2] * ray.y);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 5]), dL_dnormal[2]);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 6]), dL_dB * 2 * ray.x);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 7]), dL_dB * 2 * ray.y);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 8]), dL_dB * 2);
                    // atomicAdd(&(dL_dview2gaussian[global_id * 10 + 9]), dL_dC);
                    v_view2gaussian_local[0] = dL_dnormal[0] * ray.x;
                    v_view2gaussian_local[1] = dL_dnormal[0] * ray.y + dL_dnormal[1] * ray.x;
                    v_view2gaussian_local[2] = dL_dnormal[0] + dL_dnormal[2] * ray.x;
                    v_view2gaussian_local[3] = dL_dnormal[1] * ray.y;
                    v_view2gaussian_local[4] = dL_dnormal[1] + dL_dnormal[2] * ray.y;
                    v_view2gaussian_local[5] = dL_dnormal[2];
                    v_view2gaussian_local[6] = dL_dB * 2 * ray.x;
                    v_view2gaussian_local[7] = dL_dB * 2 * ray.y;
                    v_view2gaussian_local[8] = dL_dB * 2;
                    v_view2gaussian_local[9] = dL_dC;
                }

                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }
            }
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<10, S>(v_view2gaussian_local, warp);
            warpSum<decltype(warp), S>(v_conic_local, warp);
            warpSum<decltype(warp), S>(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum<decltype(warp), S>(v_xy_abs_local, warp);
            }
            warpSum<decltype(warp), S>(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                S *v_conic_ptr = (S *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < 10; ++k) {
                    gpuAtomicAdd((S *)(v_view2gaussians) + 10 * g + k, v_view2gaussian_local[k]);
                }
            }
        }
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const torch::Tensor &view2gaussians,            // [C, N, 10] or [nnz, 10]
    const torch::Tensor &Ks,                        // [C, 3, 3]
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
    // options
    bool absgrad) {

    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(view2gaussians);
    CHECK_INPUT(Ks);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
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
    torch::Tensor v_view2gaussians = torch::zeros_like(view2gaussians);
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    if (n_isects) {
        const uint32_t shared_mem = tile_size * tile_size *
                                    (sizeof(int32_t) + sizeof(vec3<float>) +
                                     sizeof(vec3<float>) + sizeof(vec10<float>) + sizeof(float) * COLOR_DIM);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(raytracing_to_pixels_bwd_kernel<CDIM, float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        raytracing_to_pixels_bwd_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(), opacities.data_ptr<float>(),
                reinterpret_cast<vec10<float> *>(view2gaussians.data_ptr<float>()),
                Ks.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                absgrad
                    ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>())
                    : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(), v_opacities.data_ptr<float>(),
                v_view2gaussians.data_ptr<float>());
    }

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities, v_view2gaussians);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
raytracing_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const torch::Tensor &view2gaussians,            // [C, N, 10] or [nnz, 10]
    const torch::Tensor &Ks,                        // [C, 3, 3]
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
    // options
    bool absgrad) {

    CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                                                 \
    case N:                                                                            \
        return call_kernel_with_dim<N>(                                                \
            means2d, conics, colors, opacities, view2gaussians, Ks,                     \
            backgrounds, image_width,                                                  \
            image_height, tile_size, tile_offsets, flatten_ids, render_alphas,         \
            last_ids, v_render_colors, v_render_alphas, absgrad);

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
