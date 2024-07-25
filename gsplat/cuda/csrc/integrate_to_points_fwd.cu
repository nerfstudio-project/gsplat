#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * integrate to Points Forward Pass
 * Current implementation is the same as GOF but it could be potentially better parallelized if we distribute the points uniformly instead of using the thread the it projected into
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void integrate_to_points_fwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, 
    const uint32_t PN, const uint32_t point_n_isects,
    const bool packed,
    const vec2<S> *__restrict__ points2d, // [C, N, 2] or [nnz, 2]
    const S *__restrict__ point_depths,  // [C, N, 3] or [nnz, 3]
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,        // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,     // [C, N] or [nnz]
    const vec10<S> *__restrict__ view2gaussians, // [C, N, 10] or [nnz, 10]
    const S *__restrict__ Ks,                     // [C, 3, 3]
    const S *__restrict__ backgrounds,   // [C, COLOR_DIM]
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const int32_t *__restrict__ point_tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ point_flatten_ids,  // [n_isects]
    S *__restrict__ render_colors, // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    S *__restrict__ out_integrated_alphas, // [C, PN]
    int32_t *__restrict__ last_ids // [C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * (COLOR_DIM + 1 + 3);
    render_alphas += camera_id * image_height * image_width;
    out_integrated_alphas += camera_id * PN;
    last_ids += camera_id * image_height * image_width * 2;
    Ks += camera_id * 9;
    
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }

    S px = (S)j + 0.5f;
    S py = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;
    
    const S focal_x = Ks[0];
    const S focal_y = Ks[4];
    const S cx = Ks[2];
    const S cy = Ks[5];
    const vec3<S> ray = {(px - cx) / focal_x, (py - cy) / focal_y, 1.0};

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    const uint32_t block_size = block.size();
    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;
    int32_t point_range_start = point_tile_offsets[tile_id];
    int32_t point_range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? point_n_isects
            : point_tile_offsets[tile_id + 1];
    uint32_t point_num_batches = (point_range_end - point_range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
    vec10<S> *view2gaussian_batch =
        reinterpret_cast<vec10<float> *>(&conic_batch[block_size]); // [block_size]
    vec3<S> *points2d_depth_batch =
        reinterpret_cast<vec3<float> *>(&view2gaussian_batch[block_size]); // [block_size]
    
    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x slower
    // so we stick with float for now.
    S T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t last_contributor = 0;
    uint32_t max_contributor = -1;
    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    // + 1 for depth and + 3 for normal
    S pix_out[COLOR_DIM + 1 + 3] = {0.f};

    #define MAX_NUM_CONTRIBUTORS 256

    uint32_t n_contrib_local = 0;
	uint16_t contributed_ids[MAX_NUM_CONTRIBUTORS*4] = { 0 };
	// use 4 additional corner points so that we have more accurate estimation of contributed_ids
	S corner_Ts[5] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
	S offset_xs[5] = { 0.0f, -0.5f, 0.5f, -0.5f, 0.5f };
	S offset_ys[5] = { 0.0f, -0.5f, -0.5f, 0.5f, 0.5f };

    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            view2gaussian_batch[tr] = view2gaussians[g];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            // why not use pointer so we don't need to copy again?
            const vec10<S> view2gaussian = view2gaussian_batch[t];

            bool used = false;
            for (uint32_t k = 0; k < 5; ++k){
                const vec3<S> ray_k = {(px + offset_xs[k] - cx) / focal_x, (py +offset_ys[k] - cy) / focal_y, 1.0};

                const vec3<S> normal = {
                    view2gaussian[0] * ray_k.x + view2gaussian[1] * ray_k.y + view2gaussian[2], 
                    view2gaussian[1] * ray_k.x + view2gaussian[3] * ray_k.y + view2gaussian[4],
                    view2gaussian[2] * ray_k.x + view2gaussian[4] * ray_k.y + view2gaussian[5]
                };

                // use AA, BB, CC so that the name is unique
                S AA = ray_k.x * normal[0] + ray_k.y * normal[1] + normal[2];
                S BB = 2 * (view2gaussian[6] * ray_k.x + view2gaussian[7] * ray_k.y + view2gaussian[8]);
                S CC = view2gaussian[9];
                
                // t is the depth of the gaussian
                S depth = -BB/(2*AA);
                
                //TODO take near plane as input
                #define NEAR_PLANE 0.01f
                // depth must be positive otherwise it is not valid and we skip it
                if (depth <= NEAR_PLANE)
                    continue;
                
                // the scale of the gaussian is 1.f / sqrt(AA)
                S min_value = -(BB/AA) * (BB/4.) + CC;

                S power = -0.5f * min_value;
                if (power > 0.0f){
                    power = 0.0f;
                }
                
                S alpha = min(0.999f, opac * exp(power));

                if (alpha < 1.f / 255.f) {
                    continue;
                }

                const S next_T = corner_Ts[k] * (1.0f - alpha);
                if (next_T <= 1e-4) { // this pixel is done: exclusive
                    // done = true;
                    continue;
                }

                int32_t g = id_batch[t];
                const S vis = alpha * corner_Ts[k];
                if (k == 0){
                    const S *c_ptr = colors + g * COLOR_DIM;
                    PRAGMA_UNROLL
                    for (uint32_t ch = 0; ch < COLOR_DIM; ++ch) {
                        pix_out[ch] += c_ptr[ch] * vis;
                    }    
                }
                
                last_contributor = batch_start + t - range_start;

                // if (pix_id == 550 * 1237 + 600 && k == 0){
                //     printf("contributed_ids for middle pixel: %d\n", last_contributor);
                // }

                corner_Ts[k] = next_T;
                used = true;
            }
            if (used){
				
				contributed_ids[n_contrib_local] = (u_int16_t)last_contributor;
				n_contrib_local += 1;

				if (n_contrib_local >= MAX_NUM_CONTRIBUTORS * 4){
					done = true;
					printf("ERROR: Maximal contributors are met. This should be fixed! %d\n", n_contrib_local);
					break;
				}
                // if (pix_id == 550 * 1237 + 600){
                //     printf("contributed_ids: %d\n", last_contributor);
                // }
			}

        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward pass and
        // it can be very small and causing large diff in gradients with float32.
        // However, double precision makes the backward pass 1.5x slower so we stick
        // with float for now.
        render_alphas[pix_id] = 1.0f - T;
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * (COLOR_DIM + 1 + 3) + k] =
                backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
        }
    }
    
    // return;

    // Allocate storage for batches of collectively fetched data.
    #define MAX_NUM_PROJECTED 256
	int32_t projected_ids[MAX_NUM_PROJECTED] = { 0 };
	float2 projected_xy[MAX_NUM_PROJECTED] = { 0.f };
	float projected_depth[MAX_NUM_PROJECTED] = { 0.f };

	//TODO add a for loop here in case we got more points than MAX_NUM_PROJECTED
	uint32_t point_counter_last = 0;
	bool point_done = !inside;
	int total_projected = 0;

    while (true){
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(point_done) >= block_size) {
            break;
        }

        int num_projected = 0;
		bool excced_max_projected = false;
		done = false;

        uint32_t point_counter = 0;

        // check how many points are projected to this pixel
        for (uint32_t b = 0; b < point_num_batches; ++b) {
            // resync all threads before beginning next batch
            // end early if entire tile is done
            if (__syncthreads_count(done) >= block_size) {
                break;
            }

            // each thread fetch 1 gaussian from front to back
            // index of gaussian to load
            uint32_t point_batch_start = point_range_start + block_size * b;
            uint32_t idx = point_batch_start + tr;
            if (idx < point_range_end) {
                int32_t g = point_flatten_ids[idx]; // flatten index in [C * N] or [nnz]
                id_batch[tr] = g;
                const vec2<S> xy = points2d[g];
                const S depth = point_depths[g];
                points2d_depth_batch[tr] = {xy.x, xy.y, depth};
            }

            // wait for other threads to collect the gaussians in batch
            block.sync();

            // process gaussians in the current batch for this pixel
            uint32_t batch_size = min(block_size, point_range_end - point_batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
                point_counter++;
				if (point_counter <= point_counter_last){
					continue;
				}
                const vec3<S> point_xy_depth = points2d_depth_batch[t];
                const float2 point_xy = {point_xy_depth.x, point_xy_depth.y};
                const float depth = point_xy_depth.z;
                if ((point_xy.x >= (px - 0.5)) && (point_xy.x < (px + 0.5)) && 
					(point_xy.y >= (py - 0.5)) && (point_xy.y < (py + 0.5))){
					//TODO maybe add a check for depth to filter some points
					if (true ){

						if (num_projected >= MAX_NUM_PROJECTED){
							done = true;
							excced_max_projected = true;
                            break;
						}

						projected_ids[num_projected] = id_batch[t];
						projected_xy[num_projected] = point_xy;
						projected_depth[num_projected] = depth;
						num_projected += 1;
					}
                    // if (pix_id == 550 * 1237 + 600){
                    //     printf("projected_points_ids: %d points_xy_depth: %.10f %.10f %.10f\n", id_batch[t], point_xy.x, point_xy.y, depth);
                    // }
                    // ((points2d[0, :, 0:1] >= 600) & (points2d[0, :, 0:1] <= 601) & (points2d[0, :, 1:2] >= 401) & (points2d[0, :, 1:2] <=402) ).float().argmax()
				}

            }
        }
        // return;

        point_counter_last = point_counter - 1;
		point_done = !excced_max_projected;
		total_projected += num_projected;

        // reiterate all primitives
        //TODO we could allocate the memory with dynamic size
		float point_alphas[MAX_NUM_PROJECTED] = { 0.f};
		float point_Ts[MAX_NUM_PROJECTED] = {0.f};
		for (int i = 0; i < num_projected; i++){
			point_Ts[i] = 1.f;
		}

        uint32_t num_iterated = 0;
		// bool second_done = !inside;
        done = !inside;
		uint16_t num_contributed_second = 0;

        for (uint32_t b = 0; b < num_batches; ++b) {
            // resync all threads before beginning next batch
            // end early if entire tile is done
            if (__syncthreads_count(done) >= block_size) {
                break;
            }

            // each thread fetch 1 gaussian from front to back
            // index of gaussian to load
            uint32_t batch_start = range_start + block_size * b;
            uint32_t idx = batch_start + tr;
            if (idx < range_end) {
                int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
                id_batch[tr] = g;
                const vec2<S> xy = means2d[g];
                const S opac = opacities[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opac};
                conic_batch[tr] = conics[g];
                view2gaussian_batch[tr] = view2gaussians[g];
            }

            // wait for other threads to collect the gaussians in batch
            block.sync();

            // process gaussians in the current batch for this pixel
            uint32_t batch_size = min(block_size, range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
                uint32_t cur_idx = batch_start + t - range_start;
                
                if (cur_idx > last_contributor){
                    done = true;
                    continue;
                }

                if (cur_idx != (uint32_t)contributed_ids[num_contributed_second]){
					continue;
				} else{
					num_contributed_second += 1;
				}

                const vec3<S> conic = conic_batch[t];
                const vec3<S> xy_opac = xy_opacity_batch[t];
                const S opac = xy_opac.z;
                // why not use pointer so we don't need to copy again?
                const vec10<S> view2gaussian = view2gaussian_batch[t];

                for (uint32_t k = 0; k < num_projected; ++k){
                    // if (pix_id == 550 * 1237 + 600 && k == 0){
                    //     printf("ealpha last_contributor: %d num_iterated: %d\n", batch_start + t - range_start, cur_idx);
                    // }

                    const vec3<S> ray_k = {(projected_xy[k].x - cx) / focal_x, (projected_xy[k].y - cy) / focal_y, 1.0};
                    const S ray_depth = projected_depth[k];

                    const vec3<S> normal = {
                        view2gaussian[0] * ray_k.x + view2gaussian[1] * ray_k.y + view2gaussian[2], 
                        view2gaussian[1] * ray_k.x + view2gaussian[3] * ray_k.y + view2gaussian[4],
                        view2gaussian[2] * ray_k.x + view2gaussian[4] * ray_k.y + view2gaussian[5]
                    };

                    // use AA, BB, CC so that the name is unique
                    S AA = ray_k.x * normal[0] + ray_k.y * normal[1] + normal[2];
                    S BB = 2 * (view2gaussian[6] * ray_k.x + view2gaussian[7] * ray_k.y + view2gaussian[8]);
                    S CC = view2gaussian[9];
                    
                    // t is the depth of the gaussian
                    S depth = -BB/(2*AA);
                    
                    if (depth > ray_depth){
						depth = ray_depth;
					}
                    
                    S power = -0.5f * (AA * depth * depth + BB * depth + CC);
                    if (power > 0.0f){
                        power = 0.0f;
                    }
                    
                    S alpha = min(0.999f, opac * exp(power));

                    if (alpha < 1.f / 255.f) {
                        continue;
                    }

                    const S next_T = point_Ts[k] * (1.0f - alpha);
                    const S vis = alpha * point_Ts[k];
                    point_alphas[k] += vis;
                    point_Ts[k] = next_T;
                }
            }
        }

        if (inside){
            for (int k = 0; k < num_projected; k++){
				out_integrated_alphas[projected_ids[k]] = point_alphas[k];
				// // write colors
				// for (int ch = 0; ch < CHANNELS; ch++)
				// 	out_color_integrated[CHANNELS * projected_ids[k] + ch] = C[ch] + corner_Ts[0] * bg_color[ch];;
			}
        }

    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor> call_kernel_with_dim(
    // Point parameters
    const torch::Tensor &points2d,                   // [C, N, 2]
    const torch::Tensor &point_depths,                    // [C, N, 3]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const torch::Tensor &view2gaussians,            // [C, N, 10] or [nnz, 10]
    const torch::Tensor &Ks,                        // [C, 3, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    // points intersections
    const torch::Tensor &point_tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &point_flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(view2gaussians);
    CHECK_INPUT(Ks);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(points2d);
    CHECK_INPUT(point_depths);
    CHECK_INPUT(point_tile_offsets);
    CHECK_INPUT(point_flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    uint32_t PN = points2d.size(1);
    uint32_t point_n_isects = point_flatten_ids.size(0);
    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    // + 1 for depth and + 3 for normal
    torch::Tensor renders = torch::zeros({C, image_height, image_width, channels + 1 + 3},
                                         means2d.options().dtype(torch::kFloat32));
    torch::Tensor alphas = torch::empty({C, image_height, image_width, 1},
                                        means2d.options().dtype(torch::kFloat32));
    torch::Tensor out_integrated_alphas = torch::full({C, PN}, 1.0,
                                        means2d.options().dtype(torch::kFloat32));
    printf("out_integrated_alphas size: %d %d\n", out_integrated_alphas.size(0), out_integrated_alphas.size(1));
    // 1 for last_ids and 1 for max_contributor 
    torch::Tensor last_ids = torch::empty({C, image_height, image_width, 2},
                                          means2d.options().dtype(torch::kInt32));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) + sizeof(vec10<float>) + sizeof(vec3<float>));

    // TODO: an optimization can be done by passing the actual number of channels into
    // the kernel functions and avoid necessary global memory writes. This requires
    // moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(integrate_to_points_fwd_kernel<CDIM, float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                 " bytes), try lowering tile_size.");
    }
    integrate_to_points_fwd_kernel<CDIM, float>
        <<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, PN, point_n_isects, packed,
            reinterpret_cast<vec2<float> *>(points2d.data_ptr<float>()),
            point_depths.data_ptr<float>(),
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(), opacities.data_ptr<float>(),
            reinterpret_cast<vec10<float> *>(view2gaussians.data_ptr<float>()),
            Ks.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            point_tile_offsets.data_ptr<int32_t>(), point_flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(), alphas.data_ptr<float>(),
            out_integrated_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());

    return std::make_tuple(renders, out_integrated_alphas);
}

std::tuple<torch::Tensor, torch::Tensor> integrate_to_points_fwd_tensor(
    // Point parameters
    const torch::Tensor &points2d,                   // [C, N, 2]
    const torch::Tensor &point_depths,                    // [C, N, 3]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const torch::Tensor &view2gaussians,            // [C, N, 10] or [nnz, 10]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    // points intersections
    const torch::Tensor &point_tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &point_flatten_ids   // [n_isects]
) {
    CHECK_INPUT(colors);
    uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                                                 \
    case N:                                                                            \
        return call_kernel_with_dim<N>(points2d, point_depths,                         \
                                       means2d, conics, colors, opacities,             \
                                       view2gaussians, Ks,                             \
                                       backgrounds, image_width, image_height,         \
                                       tile_size, tile_offsets, flatten_ids,           \
                                       point_tile_offsets, point_flatten_ids);

    // TODO: an optimization can be done by passing the actual number of channels into
    // the kernel functions and avoid necessary global memory writes. This requires
    // moving the channel padding from python to C side.
    switch (channels) {
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
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}
