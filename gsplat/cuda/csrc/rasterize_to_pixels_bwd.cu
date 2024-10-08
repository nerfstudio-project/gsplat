#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "tetra.cuh" // ray_tetra_intersection

#define INV_SQRT_PI 0.564189583547756286948079451560772585844050629329f

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Backward Pass
 ****************************************************************************/

template <typename S>
inline __device__ void integral_opacity_vjp(
    const S density,
    // ray
    const vec3<S> ray_o, const vec3<S> ray_d,
    // gaussian
    const vec3<S> mean3d, const mat3<S> precision,
    // grad input
    const S v_opacity,
    // grad output
    vec3<S> &v_mean3d, mat3<S> &v_precision, S &v_density) {
    vec3<S> mu = mean3d - ray_o;
    S rr = glm::dot(ray_d, precision * ray_d);

    if (rr > 1e-6f) { // numerical issue
        S rr_inv = 1.0f / rr;
        S dr = glm::dot(mu, precision * ray_d);
        S dd = glm::dot(mu, precision * mu);
        // S bb = -dr * rr_inv;
        S cc = dd - dr * dr * rr_inv;
        S _integral = 2.0f * __expf(-0.5f * cc) * sqrtf(0.5f * PI / rr);
        S opacity = 1.0 - __expf(-density * _integral);

        v_density += v_opacity * _integral * (1.0 - opacity);
        
        S v_integral = v_opacity * density * (1.0 - opacity);
        S v_cc = -0.5f * _integral * v_integral;
        S v_rr = -_integral * rr_inv * v_integral;
        
        S v_dd = v_cc;
        S v_dr = - 2.0f * dr * rr_inv * v_cc;
        v_rr += dr * dr * rr_inv * rr_inv * v_cc;

        v_mean3d += precision * ray_d * v_dr; // from dr
        v_mean3d += 2.0f * precision * mu * v_dd; // from dd
        v_precision += glm::outerProduct(ray_d, ray_d) * v_rr; // from rr
        v_precision += glm::outerProduct(mu, ray_d) * v_dr; // from dr
        v_precision += glm::outerProduct(mu, mu) * v_dd; // from dd
    }
}


template <typename S>
inline __device__ void integral_vjp(
    // fwd inputs
    const vec3<S> ray_o, const vec3<S> ray_d, const S tmin, const S tmax,
    const vec3<S> mean3d, const mat3<S> precision,
    // grad outputs
    const S v_ratio,
    // grad inputs
    vec3<S> &v_mean3d, mat3<S> &v_precision, S &v_tmin, S &v_tmax) {
    vec3<S> mu = mean3d - ray_o;
    S aa = glm::dot(ray_d, precision * ray_d);
    if (aa <= 1e-6f) { // numerical issue
        v_mean3d = glm::vec3(0.f);
        v_precision = glm::mat3(0.f);
        return;
    }

    S raa = 1.f / aa;
    S raa2 = raa * raa;
    S bb = glm::dot(ray_d, precision * mu);
    S beta1 = sqrtf(aa * 0.5f);
    S beta2 = bb * raa;

    // d_erf/d_x = 2/sqrt(pi) * exp(-x^2)
    // v_beta1 = (
    //     INV_SQRT_PI * exp(-(beta1 * (tmax - beta2)) ** 2) * (tmax - beta2) * v_ratio
    //     - INV_SQRT_PI * exp(-(beta1 * (tmin - beta2)) ** 2) * (tmin - beta2) *
    //     v_ratio
    // )
    // v_beta2 = (
    //     - INV_SQRT_PI * exp(-(beta1 * (tmax - beta2)) ** 2) * beta1 * v_ratio
    //     + INV_SQRT_PI * exp(-(beta1 * (tmin - beta2)) ** 2) * beta1 * v_ratio
    // )
    S tmax_beta2 = tmax - beta2;
    S tmin_beta2 = tmin - beta2;
    S x1 = beta1 * tmax_beta2;
    S x2 = beta1 * tmin_beta2;
    S helper_max = __expf(-x1 * x1) * v_ratio;
    S helper_min = __expf(-x2 * x2) * v_ratio;
    S v_beta1 = INV_SQRT_PI * (helper_max * tmax_beta2 - helper_min * tmin_beta2);
    S v_beta2 = INV_SQRT_PI * (helper_min * beta1 - helper_max * beta1);
    S v_bb = v_beta2 * raa;
    S v_aa = 0.25f / beta1 * v_beta1 - bb * raa2 * v_beta2;

    v_tmax = INV_SQRT_PI * helper_max * beta1;
    v_tmin = - INV_SQRT_PI * helper_min * beta1;

    // Y = M * X
    // df/dX = M^T * df/dY
    // df/dM = X * df/dY
    v_mean3d = v_bb * glm::transpose(precision) * ray_d;
    v_precision =
        v_aa * glm::outerProduct(ray_d, ray_d) + v_bb * glm::outerProduct(ray_d, mu);
}


template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz] optional
    const S *__restrict__ backgrounds, // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // --- culling ---
    const bool enable_culling,
    const S *__restrict__ camtoworlds, // [C, 4, 4]
    const S *__restrict__ Ks,          // [C, 3, 3]
    const S *__restrict__ means3d,     // [N, 3]
    const S *__restrict__ precis,      // [N, 6]
    const S *__restrict__ tvertices,   // [N, 4, 3]
    // --- culling ---
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
    S *__restrict__ v_colors,   // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities, // [C, N] or [nnz]
    S *__restrict__ v_densities, // [C, N] or [nnz]
    // --- culling ---
    vec3<S> *__restrict__ v_means3d, // [N, 3]
    S *__restrict__ v_precis,  // [N, 6]
    S *__restrict__ v_tvertices,// [N, 4, 3]
    const S *__restrict__ densities // [C, N]
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
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // ---- culling ----
    vec3<S> ray_d, ray_o;
    if (inside) {
        const float *camtoworld = camtoworlds + 16 * camera_id;
        const float *K = Ks + 9 * camera_id;
        float u = (px - K[2]) / K[0];
        float v = (py - K[5]) / K[4];
        float inv_len = rsqrtf(u * u + v * v + 1.f);
        ray_d = vec3<S>(u * inv_len, v * inv_len, inv_len);  // camera space
        ray_d = vec3<S>(camtoworld[0] * ray_d.x + camtoworld[1] * ray_d.y +
                             camtoworld[2] * ray_d.z,
                             camtoworld[4] * ray_d.x + camtoworld[5] * ray_d.y +
                             camtoworld[6] * ray_d.z,
                             camtoworld[8] * ray_d.x + camtoworld[9] * ray_d.y +
                             camtoworld[10] * ray_d.z); // world space
        ray_o = vec3<S>(camtoworld[3], camtoworld[7], camtoworld[11]);
    }
    // ---- culling ----

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
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
        );                                         // [block_size]
    S *rgbs_batch = (S *)&conic_batch[block_size]; // [block_size * COLOR_DIM]
    // ---- culling ----
    vec3<S> *mean3d_batch =
        reinterpret_cast<vec3<float> *>(&rgbs_batch[block_size * COLOR_DIM]); // [block_size]
    S *preci_batch =
        reinterpret_cast<S *>(&mean3d_batch[block_size]); // [block_size * 6]
    S *tvertices_batch =
        reinterpret_cast<S *>(&preci_batch[block_size * 6]); // [block_size * 4 * 3]
    // ---- culling ----

    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    // the contribution from gaussians behind the current one
    S buffer[COLOR_DIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
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
            if (densities != nullptr) {
                const S densi = densities[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, densi};
            } else {
                const S opac = opacities[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opac};                
            }
            conic_batch[tr] = conics[g];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
            // ---- culling ----
            // TODO: assuming non-packed for now
            int32_t gaussian_id = g % N;
            mean3d_batch[tr] = vec3<S>{
                means3d[gaussian_id * 3], means3d[gaussian_id * 3 + 1], means3d[gaussian_id * 3 + 2]};
            const float *preci_ptr = precis + gaussian_id * 6;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < 6; ++k) {
                preci_batch[tr * 6 + k] = preci_ptr[k];
            }
            if (enable_culling) {
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
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            S alpha;
            S opac;
            vec2<S> delta;
            vec3<S> conic;
            S vis;
            S sigma;
            // ---- culling ----
            bool hit;
            S tmin, tmax;
            S ratio;
            S alpha_no_cull;
            vec3<S> mean3d;
            mat3<S> preci3x3;
            vec3<S> v0, v1, v2, v3;
            int32_t entry_face_idx, exit_face_idx;
            // ---- culling ----

            if (valid) {
                conic = conic_batch[t];
                vec3<S> xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z; // density or opacity
                
                if (densities != nullptr) {
                    mean3d = mean3d_batch[t];
                    preci3x3 = mat3<S>(
                        preci_batch[t * 6], preci_batch[t * 6 + 1], preci_batch[t * 6 + 2],
                        preci_batch[t * 6 + 1], preci_batch[t * 6 + 3], preci_batch[t * 6 + 4],
                        preci_batch[t * 6 + 2], preci_batch[t * 6 + 4], preci_batch[t * 6 + 5]);
                    // opac is actually density
                    vis = integral_opacity(opac, ray_o, ray_d, mean3d, preci3x3);
                } else {
                    delta = {xy_opac.x - px, xy_opac.y - py};
                    sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
                    vis = opac * __expf(-sigma);
                    if (sigma < 0.f) {
                        valid = false;
                    }
                }
                alpha = min(0.999f, vis);

                if (alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // ---- culling ----
            if (enable_culling) {
                // calculate intersection
                v0 = vec3<S>(
                    tvertices_batch[t * 12], 
                    tvertices_batch[t * 12 + 1], 
                    tvertices_batch[t * 12 + 2]);
                v1 = vec3<S>(
                    tvertices_batch[t * 12 + 3], 
                    tvertices_batch[t * 12 + 4], 
                    tvertices_batch[t * 12 + 5]);
                v2 = vec3<S>(
                    tvertices_batch[t * 12 + 6], 
                    tvertices_batch[t * 12 + 7], 
                    tvertices_batch[t * 12 + 8]);
                v3 = vec3<S>(
                    tvertices_batch[t * 12 + 9], 
                    tvertices_batch[t * 12 + 10], 
                    tvertices_batch[t * 12 + 11]);

                hit = ray_tetra_intersection(
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
                    valid = false;
                } else {
                    // doing integral
                    mean3d = mean3d_batch[t];
                    preci3x3 = mat3<S>(
                        preci_batch[t * 6], preci_batch[t * 6 + 1], preci_batch[t * 6 + 2],
                        preci_batch[t * 6 + 1], preci_batch[t * 6 + 3], preci_batch[t * 6 + 4],
                        preci_batch[t * 6 + 2], preci_batch[t * 6 + 4], preci_batch[t * 6 + 5]);
                    ratio = integral(ray_o, ray_d, tmin, tmax, mean3d, preci3x3);
                    alpha_no_cull = alpha;
                    // alpha *= ratio;
                    alpha = 1.0f - powf(1.0f - alpha, ratio); // this produces smaller diff
                    if (alpha < 1.f / 255.f) {
                        valid = false;
                    }
                }
            }
            // ---- culling ----

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            S v_opacity_local = 0.f;
            S v_density_local = 0.f;
            // ---- culling ----
            vec3<S> v_means3d_local = {0.f, 0.f, 0.f};
            S v_percis_local[6] = {0.f};
            S v_tvertices_local[12] = {0.f};
            // ---- culling ----
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const S fac = alpha * T;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                S v_alpha = 0.f;
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_alpha +=
                        (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                        v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    S accum = 0.f;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                // ---- culling ----
                if (enable_culling) {
                    // b = 1.0f - powf(1.0f - a, r);
                    // df/da = df/db * r * (1.0f - b) / (1.0f - a)
                    // df/dr = df/db * - powf(1.0f - a, r) * ln(1.0f - a)
                    S v_ratio = v_alpha * -powf(1.0f - alpha_no_cull, ratio) *
                                    __logf(1.0f - alpha_no_cull);
                    v_alpha *= ratio * (1.0f - alpha) / (1.0f - alpha_no_cull);

                    vec3<S> v_mean3d(0.f);
                    mat3<S> v_precision(0.f);
                    S v_tmin, v_tmax;
                    integral_vjp(ray_o, ray_d, tmin, tmax, mean3d, preci3x3, 
                                    v_ratio, v_mean3d, v_precision, v_tmin, v_tmax);
                    v_means3d_local = v_mean3d;
                    v_percis_local[0] = v_precision[0][0];
                    v_percis_local[1] = v_precision[0][1] + v_precision[1][0];
                    v_percis_local[2] = v_precision[0][2] + v_precision[2][0];
                    v_percis_local[3] = v_precision[1][1];
                    v_percis_local[4] = v_precision[1][2] + v_precision[2][1];
                    v_percis_local[5] = v_precision[2][2];

                    vec3<S> v_v0(0.f), v_v1(0.f), v_v2(0.f), v_v3(0.f);
                    ray_tetra_intersection_vjp_shortcut(
                        ray_o, ray_d, v0, v1, v2, v3,
                        entry_face_idx, exit_face_idx,
                        v_tmin, v_tmax,
                        v_v0, v_v1, v_v2, v_v3
                    );

                    v_tvertices_local[0] = v_v0.x;
                    v_tvertices_local[1] = v_v0.y;
                    v_tvertices_local[2] = v_v0.z;
                    v_tvertices_local[3] = v_v1.x;
                    v_tvertices_local[4] = v_v1.y;
                    v_tvertices_local[5] = v_v1.z;
                    v_tvertices_local[6] = v_v2.x;
                    v_tvertices_local[7] = v_v2.y;
                    v_tvertices_local[8] = v_v2.z;
                    v_tvertices_local[9] = v_v3.x;
                    v_tvertices_local[10] = v_v3.y;
                    v_tvertices_local[11] = v_v3.z;
                }
                // ---- culling ----

                if (vis <= 0.999f) {
                    if (densities != nullptr) {
                        vec3<S> v_mean3d(0.f);
                        mat3<S> v_precision(0.f);
                        S v_density = 0.f;
                        integral_opacity_vjp(
                            opac, ray_o, ray_d, mean3d, preci3x3, 
                            v_alpha, v_mean3d, v_precision, v_density);

                        v_means3d_local = v_mean3d;
                        v_percis_local[0] = v_precision[0][0];
                        v_percis_local[1] = v_precision[0][1] + v_precision[1][0];
                        v_percis_local[2] = v_precision[0][2] + v_precision[2][0];
                        v_percis_local[3] = v_precision[1][1];
                        v_percis_local[4] = v_precision[1][2] + v_precision[2][1];
                        v_percis_local[5] = v_precision[2][2];
                        v_density_local = v_density;
                    } else {
                        v_opacity_local = __expf(-sigma) * v_alpha;
                        S v_sigma = -vis * v_alpha;
                        v_conic_local = {
                            0.5f * v_sigma * delta.x * delta.x,
                            v_sigma * delta.x * delta.y,
                            0.5f * v_sigma * delta.y * delta.y
                        };
                        v_xy_local = {
                            v_sigma * (conic.x * delta.x + conic.y * delta.y),
                            v_sigma * (conic.y * delta.x + conic.z * delta.y)
                        };
                        if (v_means2d_abs != nullptr) {
                            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                        }
                    }
                }

                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }
            }
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            if (densities != nullptr) {
                warpSum<decltype(warp), S>(v_density_local, warp);
            } else {
                warpSum<decltype(warp), S>(v_conic_local, warp);
                warpSum<decltype(warp), S>(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum<decltype(warp), S>(v_xy_abs_local, warp);
                }
                warpSum<decltype(warp), S>(v_opacity_local, warp);
            }
            warpSum<decltype(warp), S>(v_means3d_local, warp);
            warpSum<6, S>(v_percis_local, warp);
            if (enable_culling) {
                warpSum<12, S>(v_tvertices_local, warp);
            }
            
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                if (densities != nullptr) {
                    gpuAtomicAdd(v_densities + g, v_density_local);
                } else {                    
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
                }

                int32_t gaussian_id = g % N;
                S *v_means3d_ptr = (S *)(v_means3d) + 3 * gaussian_id;
                gpuAtomicAdd(v_means3d_ptr, v_means3d_local.x);
                gpuAtomicAdd(v_means3d_ptr + 1, v_means3d_local.y);
                gpuAtomicAdd(v_means3d_ptr + 2, v_means3d_local.z);

                S *v_precis_ptr = (S *)(v_precis) + 6 * gaussian_id;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < 6; ++k) {
                    gpuAtomicAdd(v_precis_ptr + k, v_percis_local[k]);
                }

                // ---- culling ----
                if (enable_culling) {                    
                    S *v_tvertices_ptr = (S *)(v_tvertices) + 12 * gaussian_id;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < 12; ++k) {
                        gpuAtomicAdd(v_tvertices_ptr + k, v_tvertices_local[k]);
                    }                    
                }
            }
        }
    }
}

template <uint32_t CDIM>
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // --- culling ---
    const bool enable_culling,
    const at::optional<torch::Tensor>  &camtoworlds, // [C, 4, 4]
    const at::optional<torch::Tensor>  &Ks,          // [C, 3, 3]
    const at::optional<torch::Tensor>  &means3d,     // [N, 3]
    const at::optional<torch::Tensor>  &precis,      // [N, 6]
    const at::optional<torch::Tensor>  &tvertices,    // [N, 4, 3]
    // --- culling ---
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad,
    const at::optional<torch::Tensor> &densities
) {

    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    // GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    GSPLAT_CHECK_INPUT(render_alphas);
    GSPLAT_CHECK_INPUT(last_ids);
    GSPLAT_CHECK_INPUT(v_render_colors);
    GSPLAT_CHECK_INPUT(v_render_alphas);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    if (enable_culling) {
        GSPLAT_CHECK_INPUT(tvertices.value());
    }
    GSPLAT_CHECK_INPUT(camtoworlds.value());
    GSPLAT_CHECK_INPUT(Ks.value());
    GSPLAT_CHECK_INPUT(means3d.value());
    GSPLAT_CHECK_INPUT(precis.value());

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
    torch::Tensor v_opacities, v_densities;
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    torch::Tensor v_tvertices, v_means3d, v_precis;
    if (enable_culling) {
        v_tvertices = torch::zeros_like(tvertices.value());
    }
    v_means3d = torch::zeros_like(means3d.value());
    v_precis = torch::zeros_like(precis.value());
    
    if (densities.has_value()) {
        v_densities = torch::zeros_like(densities.value());
    } else {
        v_opacities = torch::zeros_like(opacities);
    }

    if (n_isects) {
        uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
             sizeof(float) * COLOR_DIM);
    
        if (enable_culling) {
            shared_mem += tile_size * tile_size *
            (sizeof(vec3<float>) + sizeof(float) * 6 + sizeof(float) * 12);
        } else {
            // No tvertices
            shared_mem += tile_size * tile_size *
            (sizeof(vec3<float>) + sizeof(float) * 6);
        }

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(
                rasterize_to_pixels_bwd_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size."
            );
        }
        rasterize_to_pixels_bwd_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(),
                densities.has_value() ? nullptr : opacities.data_ptr<float>(),
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
                // --- culling ---
                enable_culling,
                camtoworlds.has_value() ? camtoworlds.value().data_ptr<float>() : nullptr,
                Ks.has_value() ? Ks.value().data_ptr<float>() : nullptr,
                means3d.has_value() ? means3d.value().data_ptr<float>() : nullptr,
                precis.has_value() ? precis.value().data_ptr<float>() : nullptr,
                tvertices.has_value() ? tvertices.value().data_ptr<float>() : nullptr,
                // --- culling ---
                render_alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(
                              v_means2d_abs.data_ptr<float>()
                          )
                        : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                densities.has_value() ? nullptr : v_opacities.data_ptr<float>(),
                densities.has_value() ? v_densities.data_ptr<float>() : nullptr,
                // --- culling ---
                reinterpret_cast<vec3<float> *>(v_means3d.data_ptr<float>()),
                v_precis.data_ptr<float>(),
                enable_culling ? v_tvertices.data_ptr<float>() : nullptr,
                // --- culling ---
                densities.has_value() ? densities.value().data_ptr<float>() : nullptr
            );
    }

    return std::make_tuple(
        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities, 
        v_means3d, v_precis, v_tvertices, v_densities
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // --- culling ---
    const bool enable_culling,
    const at::optional<torch::Tensor>  &camtoworlds, // [C, 4, 4]
    const at::optional<torch::Tensor>  &Ks,          // [C, 3, 3]
    const at::optional<torch::Tensor>  &means3d,     // [N, 3]
    const at::optional<torch::Tensor>  &precis,      // [N, 6]
    const at::optional<torch::Tensor>  &tvertices,    // [N, 4, 3]
    // --- culling ---
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad,
    const at::optional<torch::Tensor> &densities
) {

    GSPLAT_CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            enable_culling,                                                    \
            camtoworlds,                                                       \
            Ks,                                                                \
            means3d,                                                           \
            precis,                                                            \
            tvertices,                                                         \
            render_alphas,                                                     \
            last_ids,                                                          \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            absgrad,                                                           \
            densities                                                          \
        );

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

} // namespace gsplat