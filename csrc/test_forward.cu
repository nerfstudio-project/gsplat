#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>

#include "forward.cuh"
#include "tgaimage.h"


float random_float() {
    return (float) std::rand() / RAND_MAX;
}


int main() {
    int num_points = 2048;
    const float fov_x = M_PI / 2.f;
    const int W = 512;
    const int H = 512;
    const float focal = 0.5 * (float) W / tan(0.5 * fov_x);
    const dim3 tile_bounds = {(W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1};
    const dim3 img_size = {W, H, 1};
    const dim3 block = {BLOCK_X, BLOCK_Y, 1};

    int num_cov3d = num_points * 6;
    int num_view = 16;

    float3 *means = new float3[num_points];
    float3 *scales = new float3[num_points];
    float4 *quats = new float4[num_points];
    float3 *rgbs = new float3[num_points];
    float *opacities = new float[num_points];
    float viewmat [] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 8.f,
        0.f, 0.f, 0.f, 1.f
    };

    // silly initialization of gaussians
    // rotate pi/4 about (1, 1, 0)
    float bd = 2.f;
    // float norm = sqrt(2);
    // float theta = M_PI / 4.f;
    // float w = norm / tan(theta / 2.f);
    float u, v, w;
    std::srand(std::time(nullptr));
    for (int i = 0; i < num_points; ++i) {
        means[i] = {bd * (random_float() - 0.5f), bd * (random_float() - 0.5f), bd * (random_float() - 0.5f)};
        scales[i] = {random_float(), random_float(), random_float()};
        rgbs[i] = {random_float(), random_float(), random_float()};

        // float v = (float) i - (float) num_points * 0.5f;
        // means[i] = {v * 0.1f, v * 0.1f, (float) i};
        // scales[i] = {0.5f, 5.f, 1.f};
        // rgbs[i] = {(float) (i % 3 == 0), (float) (i % 3 == 1), (float) (i % 3 == 2)};

        // quats[i] = {w, norm, norm, 0.f};  // w x y z convention
        // quats[i] = {1.f, 0.f, 0.f, 0.f};  // w x y z convention
        // random quat
        u = random_float();
        v = random_float();
        w = random_float();
        quats[i] = {
            sqrt(1.f - u) * sin(2.f * (float) M_PI * v),
            sqrt(1.f - u) * cos(2.f * (float) M_PI * v),
            sqrt(u) * sin(2.f * (float) M_PI * w),
            sqrt(u) * cos(2.f * (float) M_PI * w)
        };

        opacities[i] = 0.9f;
    }

    float3 *scales_d, *means_d, *rgbs_d;
    float4 *quats_d;
    float *viewmat_d, *opacities_d;

    cudaMalloc((void**) &scales_d, num_points * sizeof(float3));
    cudaMalloc((void**) &means_d, num_points * sizeof(float3));
    cudaMalloc((void**) &quats_d, num_points * sizeof(float4));
    cudaMalloc((void**) &rgbs_d, num_points * sizeof(float3));
    cudaMalloc((void**) &opacities_d, num_points * sizeof(float));
    cudaMalloc((void**) &viewmat_d, num_view * sizeof(float));

    cudaMemcpy(scales_d, scales, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(means_d, means, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(rgbs_d, rgbs, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(opacities_d, opacities, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(quats_d, quats, num_points * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(viewmat_d, viewmat, num_view * sizeof(float), cudaMemcpyHostToDevice);

    // allocate memory for outputs
    float *covs3d = new float[num_cov3d];
    float2 *xy = new float2[num_points];
    float *z = new float[num_points];
    int *radii = new int[num_points];
    float3 *conics = new float3[num_points];
    uint32_t *num_tiles_hit = new uint32_t[num_points];

    float *covs3d_d, *z_d;
    float2 *xy_d;
    float3 *conics_d;
    int *radii_d;
    uint32_t *num_tiles_hit_d;
    cudaMalloc((void**)&covs3d_d, num_cov3d * sizeof(float));
    cudaMalloc((void**)&xy_d, num_points * sizeof(float2));
    cudaMalloc((void**)&z_d, num_points * sizeof(float));
    cudaMalloc((void**)&radii_d, num_points * sizeof(int));
    cudaMalloc((void**)&conics_d, num_points * sizeof(float3));
    cudaMalloc((void**)&num_tiles_hit_d, num_points * sizeof(uint32_t));

    project_gaussians_forward_impl(
        num_points,
        means_d,
        scales_d,
        1.f,
        quats_d,
        viewmat_d,
        viewmat_d,
        focal,
        focal,
        img_size,
        tile_bounds,
        covs3d_d,
        xy_d,
        z_d,
        radii_d,
        conics_d,
        num_tiles_hit_d
    );
    cudaMemcpy(covs3d, covs3d_d, num_cov3d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xy, xy_d, num_points * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, z_d, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(radii, radii_d, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_tiles_hit, num_tiles_hit_d, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t num_intersects;
    uint32_t *cum_tiles_hit = new uint32_t[num_points];
    uint32_t *cum_tiles_hit_d;
    cudaMalloc((void**) &cum_tiles_hit_d, num_points * sizeof(uint32_t));
    compute_cumulative_intersects(num_points, num_tiles_hit_d, num_intersects, cum_tiles_hit_d);
    // printf("num_intersects %d\n", num_intersects);
    // cudaMemcpy(cum_tiles_hit, cum_tiles_hit_d, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_points; ++i) {
    //     printf("num_tiles_hit %d, %d\n", i, num_tiles_hit[i]);
    // }

    uint64_t *isect_ids_sorted_d;
    uint32_t *gaussian_ids_sorted_d;  // sorted by tile and depth
    uint64_t *isect_ids_sorted = new uint64_t[num_intersects];
    uint32_t *gaussian_ids_sorted = new uint32_t[num_intersects];
    cudaMalloc((void**) &isect_ids_sorted_d, num_intersects * sizeof(uint64_t));
    cudaMalloc((void**) &gaussian_ids_sorted_d, num_intersects * sizeof(uint32_t));

    uint64_t *isect_ids_unsorted_d;
    uint32_t *gaussian_ids_unsorted_d;  // sorted by tile and depth
    uint64_t *isect_ids_unsorted = new uint64_t[num_intersects];
    uint32_t *gaussian_ids_unsorted = new uint32_t[num_intersects];
    cudaMalloc((void**) &isect_ids_unsorted_d, num_intersects * sizeof(uint64_t));
    cudaMalloc((void**) &gaussian_ids_unsorted_d, num_intersects * sizeof(uint32_t));

    int num_tiles = tile_bounds.x * tile_bounds.y;
    uint2 *tile_bins_d;  // start and end indices for each tile
    uint2 *tile_bins = new uint2[num_tiles];
    cudaMalloc((void**) &tile_bins_d, num_tiles * sizeof(uint2));

    bin_and_sort_gaussians(
        num_points,
        num_intersects,
        xy_d,
        z_d,
        radii_d,
        cum_tiles_hit_d,
        tile_bounds,
        isect_ids_unsorted_d,
        gaussian_ids_unsorted_d,
        isect_ids_sorted_d,
        gaussian_ids_sorted_d,
        tile_bins_d
    );
    cudaMemcpy(isect_ids_unsorted, isect_ids_unsorted_d, num_intersects * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(gaussian_ids_unsorted, gaussian_ids_unsorted_d, num_intersects * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(isect_ids_sorted, isect_ids_sorted_d, num_intersects * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(gaussian_ids_sorted, gaussian_ids_sorted_d, num_intersects * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < num_intersects; ++i) {
    //     printf("%d unsorted isect %016lx point %03d\n", i, isect_ids_unsorted[i], gaussian_ids_unsorted[i]);
    // }
    // for (int i = 0; i < num_intersects; ++i) {
    //     printf("sorted isect %016lx point %03d\n", isect_ids_sorted[i], gaussian_ids_sorted[i]);
    // }
    // cudaMemcpy(tile_bins, tile_bins_d, num_tiles * sizeof(uint2), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_tiles; ++i) {
    //     printf("tile_bins %d %d %d\t", i, tile_bins[i].x, tile_bins[i].y);
    // }
    // std::cout << std::endl;

    float3 *out_img = new float3[W * H];
    float3 *out_img_d;
    cudaMalloc((void**)&out_img_d, W * H * sizeof(float3));

    rasterize_forward_impl(
        tile_bounds,
        block,
        img_size,
        gaussian_ids_sorted_d,
        tile_bins_d,
        xy_d,
        conics_d,
        rgbs_d,
        opacities_d,
        out_img_d
    );
    cudaMemcpy(out_img, out_img_d, W * H * sizeof(float3), cudaMemcpyDeviceToHost);
    TGAImage image(W, H, TGAImage::RGB);
    int idx;
    float3 c;
    TGAColor col;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            idx = y * W + x;
            c = out_img[idx];
            col[0] = (uint8_t) (255.f * c.x);
            col[1] = (uint8_t) (255.f * c.y);
            col[2] = (uint8_t) (255.f * c.z);
            col[3] = 255;
            image.set(x, y, col);
        }
    }
    image.write_tga_file("output.tga");

    printf("freeing memory...\n");

    cudaFree(scales_d);
    cudaFree(quats_d);
    cudaFree(rgbs_d);
    cudaFree(opacities_d);
    cudaFree(covs3d_d);
    cudaFree(viewmat_d);
    cudaFree(xy_d);
    cudaFree(z_d);
    cudaFree(radii_d);
    cudaFree(num_tiles_hit_d);
    cudaFree(cum_tiles_hit_d);
    cudaFree(tile_bins_d);
    cudaFree(isect_ids_unsorted_d);
    cudaFree(gaussian_ids_unsorted_d);
    cudaFree(isect_ids_sorted_d);
    cudaFree(gaussian_ids_sorted_d);
    cudaFree(conics_d);
    cudaFree(out_img_d);

    delete[] scales;
    delete[] means;
    delete[] rgbs;
    delete[] opacities;
    delete[] quats;
    delete[] covs3d;
    delete[] xy;
    delete[] z;
    delete[] radii;
    delete[] num_tiles_hit;
    delete[] cum_tiles_hit;
    delete[] tile_bins;
    delete[] gaussian_ids_sorted;
    delete[] out_img;
    return 0;
}
