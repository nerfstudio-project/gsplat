#include "bindings.h"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "tetra.cuh" // ray_tetra_intersection

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Indices in Range
 ****************************************************************************/

template <typename T>
__global__ void rasterize_to_indices_in_range_kernel(
    const uint32_t range_start,
    const uint32_t range_end,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec2<T> *__restrict__ means2d, // [C, N, 2]
    const vec3<T> *__restrict__ conics,  // [C, N, 3]
    const T *__restrict__ densities,     // [C, N]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // --- culling ---
    const bool enable_culling,
    const float *__restrict__ camtoworlds, // [C, 4, 4]
    const float *__restrict__ Ks,          // [C, 3, 3]
    const float *__restrict__ means3d,     // [N, 3]
    const float *__restrict__ precis,      // [N, 6]
    const float *__restrict__ tvertices,   // [N, 4, 3]
    // --- culling ---
    const T *__restrict__ transmittances,     // [C, image_height, image_width]
    const int32_t *__restrict__ chunk_starts, // [C, image_height, image_width]
    int32_t *__restrict__ chunk_cnts,         // [C, image_height, image_width]
    int64_t *__restrict__ gaussian_ids,       // [n_elems]
    int64_t *__restrict__ pixel_ids           // [n_elems]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // move pointers to the current camera
    tile_offsets += camera_id * tile_height * tile_width;
    transmittances += camera_id * image_height * image_width;

    T px = (T)j + 0.5f;
    T py = (T)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    bool first_pass = chunk_starts == nullptr;
    int32_t base;
    if (!first_pass && inside) {
        chunk_starts += camera_id * image_height * image_width;
        base = chunk_starts[pix_id];
    }

    // ---- culling ----
    vec3<T> ray_d, ray_o;
    if (enable_culling && inside) {
        const float *camtoworld = camtoworlds + 16 * camera_id;
        const float *K = Ks + 9 * camera_id;
        float u = (px - K[2]) / K[0];
        float v = (py - K[5]) / K[4];
        float inv_len = rsqrtf(u * u + v * v + 1.f);
        ray_d = vec3<T>(u * inv_len, v * inv_len, inv_len);  // camera space
        ray_d = vec3<T>(camtoworld[0] * ray_d.x + camtoworld[1] * ray_d.y +
                             camtoworld[2] * ray_d.z,
                             camtoworld[4] * ray_d.x + camtoworld[5] * ray_d.y +
                             camtoworld[6] * ray_d.z,
                             camtoworld[8] * ray_d.x + camtoworld[9] * ray_d.y +
                             camtoworld[10] * ray_d.z); // world space
        ray_o = vec3<T>(camtoworld[3], camtoworld[7], camtoworld[11]);
    }
    // ---- culling ----

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t isect_range_start = tile_offsets[tile_id];
    int32_t isect_range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (isect_range_end - isect_range_start + block_size - 1) / block_size;

    if (range_start >= num_batches) {
        // this entire tile has been processed in the previous iterations
        // so we don't need to do anything.
        return;
    }

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<T> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<T> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
        ); // [block_size]
    // ---- culling ----
    vec3<T> *mean3d_batch =
        reinterpret_cast<vec3<float> *>(&conic_batch[block_size]); // [block_size]
    T *preci_batch =
        reinterpret_cast<T *>(&mean3d_batch[block_size]); // [block_size * 6]
    T *tvertices_batch =
        reinterpret_cast<T *>(&preci_batch[block_size * 6]); // [block_size * 4 * 3]
    // ---- culling ----

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we (should) use double for it. However double make
    // bwd 1.5x slower so we stick with float for now.
    T trans, next_trans;
    if (inside) {
        trans = transmittances[pix_id];
        next_trans = trans;
    }

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    int32_t cnt = 0;
    for (uint32_t b = range_start; b < min(range_end, num_batches); ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = isect_range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < isect_range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<T> xy = means2d[g];

            // ray attributes
            const float *camtoworld = camtoworlds + 16 * camera_id;
            const float *K = Ks + 9 * camera_id;
            float u = (px - K[2]) / K[0];
            float v = (py - K[5]) / K[4];
            float inv_len = rsqrtf(u * u + v * v + 1.f);
            ray_d = vec3<T>(u * inv_len, v * inv_len, inv_len);  // camera space
            ray_d = vec3<T>(camtoworld[0] * ray_d.x + camtoworld[1] * ray_d.y +
                                camtoworld[2] * ray_d.z,
                                camtoworld[4] * ray_d.x + camtoworld[5] * ray_d.y +
                                camtoworld[6] * ray_d.z,
                                camtoworld[8] * ray_d.x + camtoworld[9] * ray_d.y +
                                camtoworld[10] * ray_d.z); // world space
            ray_o = vec3<T>(camtoworld[3], camtoworld[7], camtoworld[11]);
            
            // gaussian attributes
            vec3<T> mean3d = vec3<T>(
                means3d[g * 3], means3d[g * 3 + 1], means3d[g * 3 + 2]);
            mat3<T> preci3x3 = mat3<T>(
                precis[g * 6], precis[g * 6 + 1], precis[g * 6 + 2],
                precis[g * 6 + 1], precis[g * 6 + 3], precis[g * 6 + 4],
                precis[g * 6 + 2], precis[g * 6 + 4], precis[g * 6 + 5]);
            T density = densities[g];
            
            T opac = integral_opacity(
                ray_o, ray_d, 0.f, INFINITY, true, 
                mean3d, preci3x3, density);

            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            // ---- culling ----
            if (enable_culling) {
                // TODO: assuming non-packed for now
                int32_t gaussian_id = g % N;
                mean3d_batch[tr] = vec3<T>{
                    means3d[gaussian_id * 3], means3d[gaussian_id * 3 + 1], means3d[gaussian_id * 3 + 2]};
                const float *preci_ptr = precis + gaussian_id * 6;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < 6; ++k) {
                    preci_batch[tr * 6 + k] = preci_ptr[k];
                }
                const float *tvertice_ptr = tvertices + gaussian_id * 12;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < 12; ++k) {
                    tvertices_batch[tr * 12 + k] = tvertice_ptr[k];
                }
            }
            // ---- culling ----
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, isect_range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<T> conic = conic_batch[t];
            const vec3<T> xy_opac = xy_opacity_batch[t];
            const T opac = xy_opac.z;
            const vec2<T> delta = {xy_opac.x - px, xy_opac.y - py};
            const T sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
            T alpha = min(0.999f, opac * __expf(-sigma));

            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            // ---- culling ----
            if (enable_culling) {
                // calculate intersection
                vec3<T> v0 = vec3<T>(
                    tvertices_batch[t * 12], 
                    tvertices_batch[t * 12 + 1], 
                    tvertices_batch[t * 12 + 2]);
                vec3<T> v1 = vec3<T>(
                    tvertices_batch[t * 12 + 3], 
                    tvertices_batch[t * 12 + 4], 
                    tvertices_batch[t * 12 + 5]);
                vec3<T> v2 = vec3<T>(
                    tvertices_batch[t * 12 + 6], 
                    tvertices_batch[t * 12 + 7], 
                    tvertices_batch[t * 12 + 8]);
                vec3<T> v3 = vec3<T>(
                    tvertices_batch[t * 12 + 9], 
                    tvertices_batch[t * 12 + 10], 
                    tvertices_batch[t * 12 + 11]);

                T tmin, tmax;
                int32_t entry_face_idx, exit_face_idx;
                bool hit = ray_tetra_intersection(
                    // inputs
                    ray_o, ray_d,
                    v0, v1, v2, v3,
                    // outputs
                    entry_face_idx,
                    exit_face_idx,
                    tmin,
                    tmax
                );

                if (!hit) {
                    continue;
                }

                // doing integral
                float ratio = 0.0f;
                vec3<T> mean3d = mean3d_batch[t];
                mat3<T> preci3x3 = mat3<T>(
                    preci_batch[t * 6], preci_batch[t * 6 + 1], preci_batch[t * 6 + 2],
                    preci_batch[t * 6 + 1], preci_batch[t * 6 + 3], preci_batch[t * 6 + 4],
                    preci_batch[t * 6 + 2], preci_batch[t * 6 + 4], preci_batch[t * 6 + 5]);
                ratio = integral(ray_o, ray_d, tmin, tmax, mean3d, preci3x3);
                // alpha *= ratio;
                alpha = 1.0f - powf(1.0f - alpha, ratio); // this produces smaller diff
                if (alpha < 1.f / 255.f) {
                    continue;
                }
            }
            // ---- culling ----

            next_trans = trans * (1.0f - alpha);
            if (next_trans <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            if (first_pass) {
                // First pass of this function we count the number of gaussians
                // that contribute to each pixel
                cnt += 1;
            } else {
                // Second pass we write out the gaussian ids and pixel ids
                int32_t g = id_batch[t]; // flatten index in [C * N]
                gaussian_ids[base + cnt] = g % N;
                pixel_ids[base + cnt] =
                    pix_id + camera_id * image_height * image_width;
                cnt += 1;
            }

            trans = next_trans;
        }
    }

    if (inside && first_pass) {
        chunk_cnts += camera_id * image_height * image_width;
        chunk_cnts[pix_id] = cnt;
    }
}

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_tensor(
    const uint32_t range_start,
    const uint32_t range_end,           // iteration steps
    const torch::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &densities, // [C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    // --- culling ---
    const bool enable_culling,
    const at::optional<torch::Tensor>  &camtoworlds, // [C, 4, 4]
    const at::optional<torch::Tensor>  &Ks,          // [C, 3, 3]
    const at::optional<torch::Tensor>  &means3d,     // [N, 3]
    const at::optional<torch::Tensor>  &precis,      // [N, 6]
    const at::optional<torch::Tensor>  &tvertices    // [N, 4, 3]
    // --- culling ---
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(densities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);

    if (enable_culling) {
        GSPLAT_CHECK_INPUT(camtoworlds.value());
        GSPLAT_CHECK_INPUT(Ks.value());
        GSPLAT_CHECK_INPUT(means3d.value());
        GSPLAT_CHECK_INPUT(precis.value());
        GSPLAT_CHECK_INPUT(tvertices.value());
    }

    uint32_t C = means2d.size(0); // number of cameras
    uint32_t N = means2d.size(1); // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

    if (enable_culling) {
        shared_mem += tile_size * tile_size *
        (sizeof(vec3<float>) + sizeof(float) * 6 + sizeof(float) * 12);
    }

    if (cudaFuncSetAttribute(
            rasterize_to_indices_in_range_kernel<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }

    // First pass: count the number of gaussians that contribute to each pixel
    int64_t n_elems;
    torch::Tensor chunk_starts;
    if (n_isects) {
        torch::Tensor chunk_cnts = torch::zeros(
            {C * image_height * image_width},
            means2d.options().dtype(torch::kInt32)
        );
        rasterize_to_indices_in_range_kernel<float>
            <<<blocks, threads, shared_mem, stream>>>(
                range_start,
                range_end,
                C,
                N,
                n_isects,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                densities.data_ptr<float>(),
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                // --- culling ---
                enable_culling,
                camtoworlds.has_value() ? camtoworlds.value().data_ptr<float>() : nullptr,
                Ks.has_value() ? Ks.value().data_ptr<float>() : nullptr,
                means3d.has_value() ? means3d.value().data_ptr<float>() : nullptr,
                precis.has_value() ? precis.value().data_ptr<float>() : nullptr,
                tvertices.has_value() ? tvertices.value().data_ptr<float>() : nullptr,
                // --- culling ---
                transmittances.data_ptr<float>(),
                nullptr,
                chunk_cnts.data_ptr<int32_t>(),
                nullptr,
                nullptr
            );

        torch::Tensor cumsum =
            torch::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
        n_elems = cumsum[-1].item<int64_t>();
        chunk_starts = cumsum - chunk_cnts;
    } else {
        n_elems = 0;
    }

    // Second pass: allocate memory and write out the gaussian and pixel ids.
    torch::Tensor gaussian_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    torch::Tensor pixel_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    if (n_elems) {
        rasterize_to_indices_in_range_kernel<float>
            <<<blocks, threads, shared_mem, stream>>>(
                range_start,
                range_end,
                C,
                N,
                n_isects,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                densities.data_ptr<float>(),
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                // --- culling ---
                enable_culling,
                camtoworlds.has_value() ? camtoworlds.value().data_ptr<float>() : nullptr,
                Ks.has_value() ? Ks.value().data_ptr<float>() : nullptr,
                means3d.has_value() ? means3d.value().data_ptr<float>() : nullptr,
                precis.has_value() ? precis.value().data_ptr<float>() : nullptr,
                tvertices.has_value() ? tvertices.value().data_ptr<float>() : nullptr,
                // --- culling ---
                transmittances.data_ptr<float>(),
                chunk_starts.data_ptr<int32_t>(),
                nullptr,
                gaussian_ids.data_ptr<int64_t>(),
                pixel_ids.data_ptr<int64_t>()
            );
    }
    return std::make_tuple(gaussian_ids, pixel_ids);
}

} // namespace gsplat