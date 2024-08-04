#include "bindings.h"
#include "utils.cuh"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Backward Pass 2DGS
 ****************************************************************************/
template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_bwd_2dgs_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    // fwd inputs
    const vec2<S> *__restrict__ means2d,
    const S *__restrict__ ray_transformations,
    const S *__restrict__ colors,
    const vec3<S> *__restrict__ normals,
    const S *__restrict__ opacities,
    const S *__restrict__ backgrounds,
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    // fwd outputs
    const S *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    // grad outputs
    const S *__restrict__ v_render_colors,
    const S *__restrict__ v_render_alphas,
    const S *__restrict__ v_render_normals,
    // grad inputs
    vec2<S> *__restrict__ v_means2d_abs,
    vec2<S> *__restrict__ v_means2d,
    S *__restrict__ v_ray_transformations,
    S *__restrict__ v_colors,
    S *__restrict__ v_opacities,
    S *__restrict__ v_normal3d
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
    v_render_normals += camera_id * image_height * image_width * 3;
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
    
    // __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    // __shared__ vec3<S> xy_opacity_batch[MAX_BLOCK_SIZE];
    // __shared__ vec3<S> u_transform_batch[MAX_BLOCK_SIZE];
    // __shared__ vec3<S> v_transform_batch[MAX_BLOCK_SIZE];
    // __shared__ vec3<S> w_transform_batch[MAX_BLOCK_SIZE];
    // __shared__ S rgbs_batch[MAX_BLOCK_SIZE * COLOR_DIM];
    // __shared__ S normals_batch[MAX_BLOCK_SIZE * 3];

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *u_transform_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3<S> *v_transform_batch = 
        reinterpret_cast<vec3<float> *>(&u_transform_batch[block_size]);
    vec3<S> *w_transform_batch = 
        reinterpret_cast<vec3<float> *>(&v_transform_batch[block_size]);
    S *rgbs_batch = (S *)&w_transform_batch[block_size]; // [block_size * COLOR_DIM]
    S *normals_batch = (S *)&rgbs_batch[block_size * COLOR_DIM]; // [block_size * 3]

    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    // the contribution from gaussians behind the current one
    S buffer[COLOR_DIM] = {0.f};
    S buffer_normals[3] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];
    S v_render_n[3];
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < 3; ++k) {
        v_render_n[k] = v_render_normals[pix_id * 3 + k];
    }



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
            u_transform_batch[tr] = {ray_transformations[g * 9 + 0], ray_transformations[g * 9 + 1], ray_transformations[g * 9 + 2]};
            v_transform_batch[tr] = {ray_transformations[g * 9 + 3], ray_transformations[g * 9 + 4], ray_transformations[g * 9 + 5]};
            w_transform_batch[tr] = {ray_transformations[g * 9 + 6], ray_transformations[g * 9 + 7], ray_transformations[g * 9 + 8]};

            const vec3<S> normal = normals[g];
            normals_batch[tr * 3] = normal.x;
            normals_batch[tr * 3 + 1] = normal.y;
            normals_batch[tr * 3 + 2] = normal.z;

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
            S vis; 
            S gauss_weight_3d;
            S gauss_weight_2d;
            S gauss_weight;
            vec2<S> s;
            vec2<S> d;
            vec3<S> h_u;
            vec3<S> h_v;
            vec3<S> intersect;
            vec3<S> w_transform;
            if (valid) { 
                vec3<S> xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                const vec3<S> u_transform = u_transform_batch[t];
                const vec3<S> v_transform = v_transform_batch[t];
                w_transform = w_transform_batch[t];

                h_u = px * w_transform - u_transform;
                h_v = py * w_transform - v_transform;
                

                // cross product of two planes is a line
                intersect = cross_product(h_u, h_v);

                // No intersection
                if (intersect.z == 0.0) valid = false;
                s = {intersect.x / intersect.z, intersect.y / intersect.z};

                gauss_weight_3d = s.x * s.x + s.y * s.y;
                d = {xy_opac.x - px, xy_opac.y - py};
                gauss_weight_2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

                gauss_weight = min(gauss_weight_3d, gauss_weight_2d);
                const S sigma = 0.5f * gauss_weight;
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
            vec3<S> v_u_transform_local = {0.f, 0.f, 0.f};
            vec3<S> v_v_transform_local = {0.f, 0.f, 0.f};
            vec3<S> v_w_transform_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            S v_opacity_local = 0.f;
            S v_normal_local[3] = {0.f};
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

                // update v_normal for this gaussian
                // TODO (WZ): derive the computational graph to see if the gradient flow
                // is correct or not.
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < 3; ++k) {
                    v_normal_local[k] = fac * v_render_n[k];
                }

                for (uint32_t k = 0; k < 3; ++k) {
                    v_alpha += (normals_batch[t * 3 + k] * T - buffer_normals[k] * ra) *
                                v_render_n[k];
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


                //====== 2DGS ======//
                if (opac * vis <= 0.999f) {
                    const S v_G = opac * v_alpha;
                    S v_depth = 0.f;
                    if (gauss_weight_3d <= gauss_weight_2d) {
                        const vec2<S> v_s = {
                            v_G * -vis * s.x + v_depth * w_transform.x,
                            v_G * -vis * s.y + v_depth * w_transform.y
                        };
                        const vec3<S> v_z_w_transform = {s.x, s.y, 1.0};
                        const S v_sx_pz = v_s.x / intersect.z;
                        const S v_sy_pz = v_s.y / intersect.z;
                        const vec3<S> v_intersect = {v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)};
                        
                        
                        const vec3<S> v_h_u = cross_product(h_v, v_intersect);
                        const vec3<S> v_h_v = cross_product(v_intersect, h_u);
                        
                        v_u_transform_local = {-v_h_u.x, -v_h_u.y, -v_h_u.z};
                        v_v_transform_local = {-v_h_v.x, -v_h_v.y, -v_h_v.z};
                        v_w_transform_local = {
                            px * v_h_u.x + py * v_h_v.x + v_depth * v_z_w_transform.x,
                            px * v_h_u.y + py * v_h_v.y + v_depth * v_z_w_transform.y,
                            px * v_h_u.z + py * v_h_v.z + v_depth * v_z_w_transform.z
                        };
                        
                    } else {
                        const S v_G_ddelx = -vis * FilterInvSquare * d.x;
                        const S v_G_ddely = -vis * FilterInvSquare * d.y;
                        v_xy_local = {v_G * v_G_ddelx, v_G * v_G_ddely};
                    }

                    v_opacity_local = vis * v_alpha;
                }

                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }

                PRAGMA_UNROLL
                for (uint32_t k = 0; k < 3; ++k) {
                    buffer_normals[k] += normals_batch[t * 3 + k] * fac;
                }
                
            }
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<3, S>(v_normal_local, warp);
            warpSum<decltype(warp), S>(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum<decltype(warp), S>(v_xy_abs_local, warp);
            }
            warpSum<decltype(warp), S>(v_opacity_local, warp);
            warpSum<decltype(warp), S>(v_u_transform_local, warp);
            warpSum<decltype(warp), S>(v_v_transform_local, warp);
            warpSum<decltype(warp), S>(v_w_transform_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                //====== 2DGS ======//
                // if (gauss_weight_3d <= gauss_weight_2d) {
                S *v_ray_transformation_ptr = (S *)(v_ray_transformations) + 9 * g;
                gpuAtomicAdd(v_ray_transformation_ptr, v_u_transform_local.x);
                gpuAtomicAdd(v_ray_transformation_ptr + 1, v_u_transform_local.y);
                gpuAtomicAdd(v_ray_transformation_ptr + 2, v_u_transform_local.z);
                gpuAtomicAdd(v_ray_transformation_ptr + 3, v_v_transform_local.x);
                gpuAtomicAdd(v_ray_transformation_ptr + 4, v_v_transform_local.y);
                gpuAtomicAdd(v_ray_transformation_ptr + 5, v_v_transform_local.z);
                gpuAtomicAdd(v_ray_transformation_ptr + 6, v_w_transform_local.x);
                gpuAtomicAdd(v_ray_transformation_ptr + 7, v_w_transform_local.y);
                gpuAtomicAdd(v_ray_transformation_ptr + 8, v_w_transform_local.z);
                // } else {
                // printf("%.2f, %.2f \n", gauss_weight_3d, gauss_weight_2d);
                S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }
                // }


                S *v_normal_ptr = (S *)(v_normal3d) + 3 * g;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < 3; ++k) {
                    gpuAtomicAdd(v_normal_ptr + k, v_normal_local[k]);
                }
                
                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_2dgs_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,
    const torch::Tensor &ray_transformations,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &normals,
    const at::optional<torch::Tensor> &backgrounds,
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersection
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids,
    // forward outputs
    const torch::Tensor &render_alphas,
    const torch::Tensor &last_ids,
    // gradients of outputs
    const torch::Tensor &v_render_colors,
    const torch::Tensor &v_render_alphas,
    const torch::Tensor &v_render_normals,
    // options
    bool absgrad
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transformations);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(normals);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    CHECK_INPUT(v_render_normals);
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
    torch::Tensor v_ray_transformations = torch::zeros_like(ray_transformations);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }
    torch::Tensor v_normals = torch::zeros_like(normals);
    if (n_isects) {
        const uint32_t shared_mem = tile_size * tile_size *
                                    (sizeof(int32_t) + sizeof(vec3<float>) +
                                     sizeof(vec3<float>) + +sizeof(vec3<float>) + 
                                     sizeof(vec3<float>) + sizeof(float) * COLOR_DIM
                                     +sizeof(float) * 3);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        // if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_2dgs_kernel<COLOR_DIM, float>,
        //                          cudaFuncAttributeMaxDynamicSharedMemorySize,
        //                          shared_mem) != cudaSuccess) {
        //     AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
        //              " bytes), try lowering tile_size.");
        // }
        switch (COLOR_DIM) {
        case 1:
            rasterize_to_pixels_bwd_2dgs_kernel<1, float><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()), opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                v_render_normals.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
            );
            break;
        case 2:
            rasterize_to_pixels_bwd_2dgs_kernel<2, float><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()), opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                v_render_normals.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
            );
            break;
        case 3:
            rasterize_to_pixels_bwd_2dgs_kernel<3, float><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()), opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                v_render_normals.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
            );
            break;     
        case 4:
            rasterize_to_pixels_bwd_2dgs_kernel<4, float><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()), opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                v_render_normals.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
            );
            // CUDA_CHECK_ERROR;
            // CUDA_SAFE_CALL(cudaStreamSynchronize(stream.stream()));
            break;
        case 8:
            rasterize_to_pixels_bwd_2dgs_kernel<8, float><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()), opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                v_render_normals.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
            );
            break;
        case 16:
            rasterize_to_pixels_bwd_2dgs_kernel<16, float><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()), opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                v_render_normals.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
            );
            break;
        case 32:
            rasterize_to_pixels_bwd_2dgs_kernel<32, float><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()), opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                v_render_normals.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
            );
            break;
        default:
            AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
        }
    }

    return std::make_tuple(v_means2d_abs, v_means2d, v_ray_transformations, v_colors, v_opacities, v_normals);
}


// template <uint32_t COLOR_DIM>
// __global__ void rasterize_to_pixels_bwd_2dgs_kernel(
//     const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
//     // fwd inputs
//     const float2 *__restrict__ means2d,
//     const float *__restrict__ ray_transformations,
//     const float *__restrict__ colors,
//     const float3 *__restrict__ normals,
//     const float *__restrict__ opacities,
//     const float *__restrict__ backgrounds,
//     const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
//     const uint32_t tile_width, const uint32_t tile_height,
//     const int32_t *__restrict__ tile_offsets,
//     const int32_t *__restrict__ flatten_ids,
//     // fwd outputs
//     const float *__restrict__ render_alphas,
//     const int32_t *__restrict__ last_ids,
//     // grad outputs
//     const float *__restrict__ v_render_colors,
//     const float *__restrict__ v_render_alphas,
//     const float *__restrict__ v_render_normals,
//     // grad inputs
//     float2 *__restrict__ v_means2d_abs,
//     float2 *__restrict__ v_means2d,
//     float *__restrict__ v_ray_transformations,
//     float *__restrict__ v_colors,
//     float *__restrict__ v_opacities,
//     float *__restrict__ v_normal3d
// ) {
//     auto block = cg::this_thread_block();
//     uint32_t camera_id = block.group_index().x;
//     uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
//     uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
//     uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

//     tile_offsets += camera_id * tile_height * tile_width;
//     render_alphas += camera_id * image_height * image_width;
//     last_ids += camera_id * image_height * image_width;
//     v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
//     v_render_alphas += camera_id * image_height * image_width;
//     v_render_normals += camera_id * image_height * image_width * 3;
//     if (backgrounds != nullptr) {
//         backgrounds += camera_id * COLOR_DIM;
//     }


//     const float px = (float)j + 0.5f;
//     const float py = (float)i + 0.5f;
//     // clamp this value to the last pixel
//     const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

//     // keep not rasterizing threads around for reading data
//     bool inside = (i < image_height && j < image_width);

//     // have all threads in tile process the same gaussians in batches
//     // first collect gaussians between range.x and range.y in batches
//     // which gaussians to look through in this tile
//     int32_t range_start = tile_offsets[tile_id];
//     int32_t range_end = 
//         (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
//             ? n_isects
//             : tile_offsets[tile_id + 1];
//     const uint32_t block_size = block.size();
//     const uint32_t num_batches = 
//         (range_end - range_start + block_size - 1) / block_size;
    
//     __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
//     __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
//     __shared__ float3 u_transform_batch[MAX_BLOCK_SIZE];
//     __shared__ float3 v_transform_batch[MAX_BLOCK_SIZE];
//     __shared__ float3 w_transform_batch[MAX_BLOCK_SIZE];
//     __shared__ float rgbs_batch[MAX_BLOCK_SIZE * COLOR_DIM];
//     __shared__ float normals_batch[MAX_BLOCK_SIZE * 3];

//     // this is the T AFTER the last gaussian in this pixel
//     float T_final = 1.0f - render_alphas[pix_id];
//     float T = T_final;
//     // the contribution from gaussians behind the current one
//     float buffer[COLOR_DIM] = {0.f};
//     float buffer_normals[3] = {0.f};
//     // index of last gaussian to contribute to this pixel
//     const int32_t bin_final = inside ? last_ids[pix_id] : 0;

//     // df/d_out for this pixel
//     float v_render_c[COLOR_DIM];
//     PRAGMA_UNROLL
//     for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//         v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
//     }
//     const float v_render_a = v_render_alphas[pix_id];
//     float v_render_n[3];
//     PRAGMA_UNROLL
//     for (uint32_t k = 0; k < 3; ++k) {
//         v_render_n[k] = v_render_normals[pix_id * 3 + k];
//     }



//     // collect and process batches of gaussians
//     // each thread loads one gaussian at a time before rasterizing
//     const uint32_t tr = block.thread_rank();
//     cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
//     const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
//     for (uint32_t b = 0; b < num_batches; ++b) {
//         // resync all threads before writing next batch of shared mem
//         block.sync();

//         // each thread fetch 1 gaussian from back to front 
//         // 0 index will be furthest back in batch
//         // index of gaussian to load
//         // batch end is the index of the last gaussian in the batch
//         // These values can be negative so must be int32 instead of uint32
//         const int32_t batch_end = range_end - 1 - block_size * b;
//         const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
//         const int32_t idx = batch_end - tr;
//         if (idx >= range_start) {
//             int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
//             id_batch[tr] = g;
//             const float2 xy = means2d[g];
//             const float opac = opacities[g];
//             xy_opacity_batch[tr] = {xy.x, xy.y, opac};
//             u_transform_batch[tr] = {ray_transformations[g * 9 + 0], ray_transformations[g * 9 + 1], ray_transformations[g * 9 + 2]};
//             v_transform_batch[tr] = {ray_transformations[g * 9 + 3], ray_transformations[g * 9 + 4], ray_transformations[g * 9 + 5]};
//             w_transform_batch[tr] = {ray_transformations[g * 9 + 6], ray_transformations[g * 9 + 7], ray_transformations[g * 9 + 8]};

//             const float3 normal = normals[g];
//             normals_batch[tr * 3] = normal.x;
//             normals_batch[tr * 3 + 1] = normal.y;
//             normals_batch[tr * 3 + 2] = normal.z;

//             PRAGMA_UNROLL
//             for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//                 rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
//             }

//         }

//         // wait for other threads to collect the gaussians in batch
//         block.sync();
//         // process gaussians in the current batch for this pixel
//         // 0 index is the furthest back gaussian in the batch
//         for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
//             bool valid = inside;
//             if (batch_end - t > bin_final) {
//                 valid = 0;
//             }
//             float alpha;
//             float opac;
//             float vis; 
//             float gauss_weight_3d;
//             float gauss_weight_2d;
//             float gauss_weight;
//             float2 s;
//             float2 d;
//             float3 h_u;
//             float3 h_v;
//             float3 intersect;
//             float3 w_transform;
//             if (valid) { 
//                 float3 xy_opac = xy_opacity_batch[t];
//                 opac = xy_opac.z;
//                 const float3 u_transform = u_transform_batch[t];
//                 const float3 v_transform = v_transform_batch[t];
//                 w_transform = w_transform_batch[t];

//                 h_u = px * w_transform - u_transform;
//                 h_v = py * w_transform - v_transform;
                

//                 // cross product of two planes is a line
//                 intersect = cross_product(h_u, h_v);

//                 // No intersection
//                 if (intersect.z == 0.0) valid = false;
//                 s = {intersect.x / intersect.z, intersect.y / intersect.z};

//                 gauss_weight_3d = f2_norm2(s);
//                 d = {xy_opac.x - px, xy_opac.y - py};
//                 gauss_weight_2d = FilterInvSquare * f2_norm2(d);

//                 gauss_weight = min(gauss_weight_3d, gauss_weight_2d);
//                 const float sigma = 0.5f * gauss_weight;
//                 vis = __expf(-sigma);
//                 alpha = min(0.999f, opac * vis);
//                 if (sigma < 0.f || alpha < 1.f / 255.f) {
//                     valid = false;
//                 }
//             }


//             // if all threads are inactive in this warp, skip this loop
//             if (!warp.any(valid)) {
//                 continue;
//             }
//             float v_rgb_local[COLOR_DIM] = {0.f};
//             float3 v_u_transform_local = {0.f, 0.f, 0.f};
//             float3 v_v_transform_local = {0.f, 0.f, 0.f};
//             float3 v_w_transform_local = {0.f, 0.f, 0.f};
//             float2 v_xy_local = {0.f, 0.f};
//             float2 v_xy_abs_local = {0.f, 0.f};
//             float v_opacity_local = 0.f;
//             float2 v_densification_local = {0.f, 0.f};
//             float v_normal_local[3] = {0.f};
//             // initialize everything to 0, only set if the lane is valid
//             if (valid) {
//                 // compute the current T for this gaussian
//                 float ra = 1.0f / (1.0f - alpha);
//                 T *= ra;
//                 // update v_rgb for this gaussian
//                 const float fac = alpha * T;
//                 PRAGMA_UNROLL
//                 for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//                     v_rgb_local[k] = fac * v_render_c[k];
//                 }
//                 // contribution from this pixel
//                 float v_alpha = 0.f;
//                 for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//                     v_alpha += (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
//                                 v_render_c[k];
//                 }

//                 // update v_normal for this gaussian
//                 // TODO (WZ): derive the computational graph to see if the gradient flow
//                 // is correct or not.
//                 PRAGMA_UNROLL
//                 for (uint32_t k = 0; k < 3; ++k) {
//                     v_normal_local[k] = fac * v_render_n[k];
//                 }

//                 for (uint32_t k = 0; k < 3; ++k) {
//                     v_alpha += (normals_batch[t * 3 + k] * T - buffer_normals[k] * ra) *
//                                 v_render_n[k];
//                 }

//                 v_alpha += T_final * ra * v_render_a;

//                 // contribution from background pixel
//                 if (backgrounds != nullptr) {
//                     float accum = 0.f;
//                     PRAGMA_UNROLL
//                     for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//                         accum += backgrounds[k] * v_render_c[k];
//                     }
//                     v_alpha += -T_final * ra * accum;
//                 }


//                 //====== 2DGS ======//
//                 if (opac * vis <= 0.999f) {
//                     const float v_G = opac * v_alpha;
//                     float v_depth = 0.f;
//                     if (gauss_weight_3d <= gauss_weight_2d) {
//                         const float2 v_s = {
//                             v_G * -vis * s.x + v_depth * w_transform.x,
//                             v_G * -vis * s.y + v_depth * w_transform.y
//                         };
//                         const float3 v_z_w_transform = {s.x, s.y, 1.0};
//                         const float v_sx_pz = v_s.x / intersect.z;
//                         const float v_sy_pz = v_s.y / intersect.z;
//                         const float3 v_intersect = {v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)};
//                         const float3 v_h_u = cross_product(h_v, v_intersect);
//                         const float3 v_h_v = cross_product(v_intersect, h_u);
                        
//                         v_u_transform_local = {-v_h_u.x, -v_h_u.y, -v_h_u.z};
//                         v_v_transform_local = {-v_h_v.x, -v_h_v.y, -v_h_v.z};
//                         v_w_transform_local = {
//                             px * v_h_u.x + py * v_h_v.x + v_depth * v_z_w_transform.x,
//                             px * v_h_u.y + py * v_h_v.y + v_depth * v_z_w_transform.y,
//                             px * v_h_u.z + py * v_h_v.z + v_depth * v_z_w_transform.z
//                         };
                        
//                     } else {
//                         const float v_G_ddelx = -vis * FilterInvSquare * d.x;
//                         const float v_G_ddely = -vis * FilterInvSquare * d.y;
//                         v_xy_local = {v_G * v_G_ddelx, v_G * v_G_ddely};
//                     }

//                     v_opacity_local = vis * v_alpha;
//                 }

//                 PRAGMA_UNROLL
//                 for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//                     buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
//                 }

//                 PRAGMA_UNROLL
//                 for (uint32_t k = 0; k < 3; ++k) {
//                     buffer_normals[k] += normals_batch[t * 3 + k] * fac;
//                 }
                
//                 float depth = w_transform.z;
//                 v_densification_local.x = v_u_transform_local.z * depth;
//                 v_densification_local.y = v_v_transform_local.z * depth;
//             }
//             warpSum<COLOR_DIM, float>(v_rgb_local, warp);
//             warpSum<3, float>(v_normal_local, warp);
//             warpSum(v_xy_local, warp);
//             if (v_means2d_abs != nullptr) {
//                 warpSum(v_xy_abs_local, warp);
//             }
//             warpSum(v_opacity_local, warp);
//             warpSum(v_u_transform_local, warp);
//             warpSum(v_v_transform_local, warp);
//             warpSum(v_w_transform_local, warp);
//             warpSum(v_densification_local, warp);
//             if (warp.thread_rank() == 0) {
//                 int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
//                 float *v_rgb_ptr = (float *)(v_colors) + COLOR_DIM * g;
//                 PRAGMA_UNROLL
//                 for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//                     atomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
//                 }

//                 //====== 2DGS ======//
//                 // if (gauss_weight_3d <= gauss_weight_2d) {
//                 float *v_ray_transformation_ptr = (float *)(v_ray_transformations) + 9 * g;
//                 atomicAdd(v_ray_transformation_ptr, v_u_transform_local.x);
//                 atomicAdd(v_ray_transformation_ptr + 1, v_u_transform_local.y);
//                 atomicAdd(v_ray_transformation_ptr + 2, v_u_transform_local.z);
//                 atomicAdd(v_ray_transformation_ptr + 3, v_v_transform_local.x);
//                 atomicAdd(v_ray_transformation_ptr + 4, v_v_transform_local.y);
//                 atomicAdd(v_ray_transformation_ptr + 5, v_v_transform_local.z);
//                 atomicAdd(v_ray_transformation_ptr + 6, v_w_transform_local.x);
//                 atomicAdd(v_ray_transformation_ptr + 7, v_w_transform_local.y);
//                 atomicAdd(v_ray_transformation_ptr + 8, v_w_transform_local.z);
//                 // } else {
//                 // printf("%.2f, %.2f \n", gauss_weight_3d, gauss_weight_2d);
//                 float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
//                 atomicAdd(v_xy_ptr, v_xy_local.x);
//                 atomicAdd(v_xy_ptr + 1, v_xy_local.y);

//                 if (v_means2d_abs != nullptr) {
//                     float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
//                     atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
//                     atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
//                 }
//                 // }


//                 float *v_normal_ptr = (float *)(v_normal3d) + 3 * g;
//                 PRAGMA_UNROLL
//                 for (uint32_t k = 0; k < 3; ++k) {
//                     atomicAdd(v_normal_ptr + k, v_normal_local[k]);
//                 }
                
//                 atomicAdd(v_opacities + g, v_opacity_local);
//             }
//         }
//     }
// }

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
// rasterize_to_pixels_bwd_2dgs_tensor(
//     // Gaussian parameters
//     const torch::Tensor &means2d,
//     const torch::Tensor &ray_transformations,
//     const torch::Tensor &colors,
//     const torch::Tensor &opacities,
//     const torch::Tensor &normals,
//     const at::optional<torch::Tensor> &backgrounds,
//     // image size
//     const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
//     // intersection
//     const torch::Tensor &tile_offsets,
//     const torch::Tensor &flatten_ids,
//     // forward outputs
//     const torch::Tensor &render_alphas,
//     const torch::Tensor &last_ids,
//     // gradients of outputs
//     const torch::Tensor &v_render_colors,
//     const torch::Tensor &v_render_alphas,
//     const torch::Tensor &v_render_normals,
//     // options
//     bool absgrad
// ) {
//     DEVICE_GUARD(means2d);
//     CHECK_INPUT(means2d);
//     CHECK_INPUT(ray_transformations);
//     CHECK_INPUT(colors);
//     CHECK_INPUT(opacities);
//     CHECK_INPUT(normals);
//     CHECK_INPUT(tile_offsets);
//     CHECK_INPUT(flatten_ids);
//     CHECK_INPUT(render_alphas);
//     CHECK_INPUT(last_ids);
//     CHECK_INPUT(v_render_colors);
//     CHECK_INPUT(v_render_alphas);
//     CHECK_INPUT(v_render_normals);
//     if (backgrounds.has_value()) {
//         CHECK_INPUT(backgrounds.value());
//     }

//     bool packed = means2d.dim() == 2;

//     uint32_t C = tile_offsets.size(0);         // number of cameras
//     uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
//     uint32_t n_isects = flatten_ids.size(0);
//     uint32_t COLOR_DIM = colors.size(-1);
//     uint32_t tile_height = tile_offsets.size(1);
//     uint32_t tile_width = tile_offsets.size(2);

//     // Each block covers a tile on the image. In total there are
//     // C * tile_height * tile_width blocks.
//     dim3 threads = {tile_size, tile_size, 1};
//     dim3 blocks = {C, tile_height, tile_width};

//     torch::Tensor v_means2d = torch::zeros_like(means2d);
//     torch::Tensor v_ray_transformations = torch::zeros_like(ray_transformations);
//     torch::Tensor v_colors = torch::zeros_like(colors);
//     torch::Tensor v_opacities = torch::zeros_like(opacities);
//     torch::Tensor v_means2d_abs;
//     if (absgrad) {
//         v_means2d_abs = torch::zeros_like(means2d);
//     }
//     torch::Tensor v_normals = torch::zeros_like(normals);
//     if (n_isects) {
//         at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
//         switch (COLOR_DIM) {
//         case 1:
//             rasterize_to_pixels_bwd_2dgs_kernel<1><<<blocks, threads, 0, stream>>>(
//                 C, N, n_isects, packed, (float2 *)means2d.data_ptr<float>(),
//                 ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
//                 (float3 *)normals.data_ptr<float>(), opacities.data_ptr<float>(),
//                 backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                         : nullptr,
//                 image_width, image_height, tile_size, tile_width, tile_height,
//                 tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
//                 render_alphas.data_ptr<float>(), 
//                 last_ids.data_ptr<int32_t>(),
//                 v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
//                 v_render_normals.data_ptr<float>(),
//                 absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
//                 (float2 *)v_means2d.data_ptr<float>(),
//                 v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
//                 v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
//             );
//             break;
//         case 2:
//             rasterize_to_pixels_bwd_2dgs_kernel<2><<<blocks, threads, 0, stream>>>(
//                 C, N, n_isects, packed, (float2 *)means2d.data_ptr<float>(),
//                 ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
//                 (float3 *)normals.data_ptr<float>(), opacities.data_ptr<float>(),
//                 backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                         : nullptr,
//                 image_width, image_height, tile_size, tile_width, tile_height,
//                 tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
//                 render_alphas.data_ptr<float>(), 
//                 last_ids.data_ptr<int32_t>(),
//                 v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
//                 v_render_normals.data_ptr<float>(),
//                 absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
//                 (float2 *)v_means2d.data_ptr<float>(),
//                 v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
//                 v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
//             );
//             break;
//         case 3:
//             rasterize_to_pixels_bwd_2dgs_kernel<3><<<blocks, threads, 0, stream>>>(
//                 C, N, n_isects, packed, (float2 *)means2d.data_ptr<float>(),
//                 ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
//                 (float3 *)normals.data_ptr<float>(), opacities.data_ptr<float>(),
//                 backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                         : nullptr,
//                 image_width, image_height, tile_size, tile_width, tile_height,
//                 tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
//                 render_alphas.data_ptr<float>(), 
//                 last_ids.data_ptr<int32_t>(),
//                 v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
//                 v_render_normals.data_ptr<float>(),
//                 absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
//                 (float2 *)v_means2d.data_ptr<float>(),
//                 v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
//                 v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
//             );
//             break;     
//         case 4:
//             rasterize_to_pixels_bwd_2dgs_kernel<4><<<blocks, threads, 0, stream>>>(
//                 C, N, n_isects, packed, (float2 *)means2d.data_ptr<float>(),
//                 ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
//                 (float3 *)normals.data_ptr<float>(), opacities.data_ptr<float>(),
//                 backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                         : nullptr,
//                 image_width, image_height, tile_size, tile_width, tile_height,
//                 tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
//                 render_alphas.data_ptr<float>(), 
//                 last_ids.data_ptr<int32_t>(),
//                 v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
//                 v_render_normals.data_ptr<float>(),
//                 absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
//                 (float2 *)v_means2d.data_ptr<float>(),
//                 v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
//                 v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
//             );
//             // CUDA_CHECK_ERROR;
//             // CUDA_SAFE_CALL(cudaStreamSynchronize(stream.stream()));
//             break;
//         case 8:
//             rasterize_to_pixels_bwd_2dgs_kernel<8><<<blocks, threads, 0, stream>>>(
//                 C, N, n_isects, packed, (float2 *)means2d.data_ptr<float>(),
//                 ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
//                 (float3 *)normals.data_ptr<float>(), opacities.data_ptr<float>(),
//                 backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                         : nullptr,
//                 image_width, image_height, tile_size, tile_width, tile_height,
//                 tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
//                 render_alphas.data_ptr<float>(), 
//                 last_ids.data_ptr<int32_t>(),
//                 v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
//                 v_render_normals.data_ptr<float>(),
//                 absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
//                 (float2 *)v_means2d.data_ptr<float>(),
//                 v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
//                 v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
//             );
//             break;
//         case 16:
//             rasterize_to_pixels_bwd_2dgs_kernel<16><<<blocks, threads, 0, stream>>>(
//                 C, N, n_isects, packed, (float2 *)means2d.data_ptr<float>(),
//                 ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
//                 (float3 *)normals.data_ptr<float>(), opacities.data_ptr<float>(),
//                 backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                         : nullptr,
//                 image_width, image_height, tile_size, tile_width, tile_height,
//                 tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
//                 render_alphas.data_ptr<float>(), 
//                 last_ids.data_ptr<int32_t>(),
//                 v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
//                 v_render_normals.data_ptr<float>(),
//                 absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
//                 (float2 *)v_means2d.data_ptr<float>(),
//                 v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
//                 v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
//             );
//             break;
//         case 32:
//             rasterize_to_pixels_bwd_2dgs_kernel<32><<<blocks, threads, 0, stream>>>(
//                 C, N, n_isects, packed, (float2 *)means2d.data_ptr<float>(),
//                 ray_transformations.data_ptr<float>(), colors.data_ptr<float>(),
//                 (float3 *)normals.data_ptr<float>(), opacities.data_ptr<float>(),
//                 backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                         : nullptr,
//                 image_width, image_height, tile_size, tile_width, tile_height,
//                 tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
//                 render_alphas.data_ptr<float>(), 
//                 last_ids.data_ptr<int32_t>(),
//                 v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
//                 v_render_normals.data_ptr<float>(),
//                 absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
//                 (float2 *)v_means2d.data_ptr<float>(),
//                 v_ray_transformations.data_ptr<float>(), v_colors.data_ptr<float>(),
//                 v_opacities.data_ptr<float>(), v_normals.data_ptr<float>()
//             );
//             break;
//         default:
//             AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
//         }
//     }

//     return std::make_tuple(v_means2d_abs, v_means2d, v_ray_transformations, v_colors, v_opacities, v_normals);
// }