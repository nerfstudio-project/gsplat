#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float* __restrict__ bg_color,
    const float2* __restrict__ points_xy_image,
    const float4*4 __restrict__ conic_opacity,
    const float* __restrict__ colors,
    const float* __restrict__ final_Ts,
    const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ dL_dpixels,
    float3* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconic2D,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dcolors
) {
    // We rasterize again. Compute necessary block info.
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCY_Y};
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W && pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool done = !inside;
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float collected_colors[C * BLOCK_SIZE];

    // In the forward, we store the final value of T, the
    // product of all (1 - alpha) factors.
    const float T_final = inside ? final_Ts[pix_id] : 0;
    float T = T_final;

    // We start from the back. The ID of teh last contributing 
    // Gaussian is known from each pixel from the forward.
    uint32_t contributor = toDo;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    float accum_rec[C] = { 0 };
    float dL_dpixel[C];
    if (inside) {
        for (int i = 0; i < C; i++) {
            dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
        }
    }
    float last_alpha = 0;
    float last_color[C] = { 0 };

    // Gradient of pixel coordinate w.r.t. normalized
    // screen-space viewport coordinates (-1 to 1)
    const float ddelx_dx = 0.5 * W;
    const float ddely_dy = 0.5 * H;

    // Traverse all Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // Load auxiliary data into shared memory, start in the BACK
        // and load them in reverse order.
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            const int coll_id = point_list[range.y - progress - 1];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            for (int i = 0; i < C; i++) {
                collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
            }
        }
        block.sync();

        // Iterate over Gaussians
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            // Keep track of current Gaussian ID. Skip, if this one 
            // is behind the last contributor for this pixel.
            contributor--;
            if (contributor >= last_contributor) continue;

            // Compute blending values, as before.
            const float2 xy = collected_xy[j];
            const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            const float4 con_o = collected_conic_opacity[j];
            const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) continue;

            const float G = exp(power);
            const float alpha = min(0.99f, con_o.w * G);
            if (alpha < 1.0f / 255.0f) continue;

            T = T / (1.f - alpha);
            const float dchannel_dcolor = alpha * T;

            // Propagate gradients to per-Gaussian colors and keep
            // gradients w.r.t. alpha (blending factor for a Gaussian/pixel pair).
            float dL_dalpha = 0.0f;
            const int global_id = collected_id[j];
            for (int ch = 0; ch < C; ch++) {
                const float c = collected_colors[ch * BLOCK_SIZE + j];

                // Update last color (to be used in the next iteration)
                accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
                last_color[ch] = c;

                const float dL_dchannel = dL_dpixel[ch];
                dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
                
                // Update the gradients w.r.t. color of the Gaussian.
                // Atomic, since this pixel is just one of potentially
                // many that were affected by this Gaussian.
                aotmoicAdd(&(dL_dcolors[gloabl_id * C + ch]), dchannel_dcolor * dL_dchannel);
            }
            dL_dalpha *= T;
            // Update last alpha (to be used in the next iteration)
            last_alpha = alpha;

            // Account for fact that alpha also influences how much of 
            // the background color is added if nothing left to blend
            float bg_dot_dpixel = 0;
            for (int i = 0; i < C; i++) {
                bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
            }
            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

            // Helpful reusable temporary variables
            const float dL_dG = con_o.w * dL_dalpha;
            const float gdx = G * d.x;
            const float gdy = G * d.y;
            const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
            const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

            // Update gradients w.r.t. 2D mean position of the Gaussian
            atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
            atomicAdd(&dL_dmean2d[global_id].y, dL_dG * dG_ddely * ddely_dy);

            // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
            atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
            atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
            atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

            // Update gradients w.r.t. opacity of the Gaussian 
            atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
        }
    }
}
