#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <math.h>

#include "config.h"
#include "forward.cuh"
#include "tgaimage.h"

float random_float() { return (float)std::rand() / RAND_MAX; }


float4 random_quat() {
    float u = random_float();
    float v = random_float();
    float w = random_float();
    return {
        sqrt(1.f - u) * sin(2.f * (float) M_PI * v),
        sqrt(1.f - u) * cos(2.f * (float) M_PI * v),
        sqrt(u) * sin(2.f * (float) M_PI * w),
        sqrt(u) * cos(2.f * (float) M_PI * w)
    };
}

int main(int argc, char *argv[]) {
    int num_points = 256;
    int n = 16;
    if (argc > 1) {
        num_points = std::stoi(argv[1]);
    }
    printf("rendering %d points\n", num_points);

    float bd = 2.f;
    if (argc > 2) {
        bd = std::stof(argv[2]);
    }
    printf("using scene bounds %.2f\n", bd);

    int mode = 0;
    if (argc > 3) {  // debug
        mode = std::stoi(argv[3]);
    }
    printf("%d mode\n", mode);
    const float fov_x = M_PI / 2.f;
    const int W = 512;
    const int H = 512;
    const float focal = 0.5 * (float)W / tan(0.5 * fov_x);
    const dim3 tile_bounds = {
        (W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1};
    const dim3 img_size = {W, H, 1};
    const dim3 block = {BLOCK_X, BLOCK_Y, 1};

    int num_cov3d = num_points * 6;
    int num_view = 16;
    const int C = 3;

    float3 *means = new float3[num_points];
    float3 *scales = new float3[num_points];
    float4 *quats = new float4[num_points];
    float *rgbs = new float[C * num_points];
    float *opacities = new float[num_points];
    // world to camera matrix
    float viewmat[] = {
        1.f,
        0.f,
        0.f,
        0.f,
        0.f,
        1.f,
        0.f,
        0.f,
        0.f,
        0.f,
        1.f,
        10.f,
        0.f,
        0.f,
        0.f,
        1.f};

    // random initialization of gaussians
    std::srand(std::time(nullptr));
    if (mode == 0) {
        for (int i = 0; i < num_points; ++i) {
            float v = (float)i - (float)num_points * 0.5f;
            means[i] = {v * 0.1f, v * 0.1f, (float)i * 0.1f};
            printf("%d, %.2f %.2f %.2f\n", i, means[i].x, means[i].y, means[i].z);
            scales[i] = {1.f, 0.5f, 1.f};
            for (int c = 0; c < C; ++c) {
                rgbs[C * i + c] = (float)(i % C == c);
            }
            quats[i] = {1.f, 0.f, 0.f, 0.f};
            opacities[i] = 0.9f;
        }
    } else if (mode == 1) {
        for (int i = 0; i < num_points; ++i) {
            means[i] = {
                bd * (random_float() - 0.5f),
                bd * (random_float() - 0.5f),
                bd * (random_float() - 0.5f)
            };
            printf("%d, %.2f %.2f %.2f\n", i, means[i].x, means[i].y, means[i].z);
            scales[i] = {random_float(), random_float(), random_float()};
            for (int c = 0; c < C; ++c) {
                rgbs[C * i + c] = random_float();
            }
            quats[i] = {1.f, 0.f, 0.f, 0.f};
            opacities[i] = 0.9f;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int idx = n * i + j;
                float x = 2.f * ((float) j / (float) n - 0.5f);
                float y = 2.f * ((float) i / (float) n - 0.5f);
                // float z = 2.f * (float) idx / (float) num_points;
                float z = 5.f + random_float();
                // float z = (float) idx;
                means[idx] = {x, y, z};
                printf("%d, %.2f %.2f %.2f\n", idx, x, y, z);
                scales[idx] = {0.5f, 0.5f, 0.5f};
                for (int c = 0; c < C; ++c) {
                    // rgbs[C * idx + c] = (float)(i % C == c);
                    rgbs[C * idx + c] = random_float();
                }
                quats[idx] = {0.f, 0.f, 0.f, 1.f};  // w x y z convention
                opacities[idx] = 0.9f;
            }
        }
    }

    float3 *scales_d, *means_d;
    float *rgbs_d;
    float4 *quats_d;
    float *viewmat_d, *opacities_d;

    cudaMalloc((void **)&scales_d, num_points * sizeof(float3));
    cudaMalloc((void **)&means_d, num_points * sizeof(float3));
    cudaMalloc((void **)&quats_d, num_points * sizeof(float4));
    cudaMalloc((void **)&rgbs_d, C * num_points * sizeof(float));
    cudaMalloc((void **)&opacities_d, num_points * sizeof(float));
    cudaMalloc((void **)&viewmat_d, num_view * sizeof(float));

    cudaMemcpy(
        scales_d, scales, num_points * sizeof(float3), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        means_d, means, num_points * sizeof(float3), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        rgbs_d, rgbs, C * num_points * sizeof(float), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        opacities_d,
        opacities,
        num_points * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        quats_d, quats, num_points * sizeof(float4), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        viewmat_d, viewmat, num_view * sizeof(float), cudaMemcpyHostToDevice
    );

    // allocate memory for outputs
    float *covs3d = new float[num_cov3d];
    float2 *xy = new float2[num_points];
    float *z = new float[num_points];
    int *radii = new int[num_points];
    float3 *conics = new float3[num_points];
    int32_t *num_tiles_hit = new int32_t[num_points];

    float *covs3d_d, *z_d;
    float2 *xy_d;
    float3 *conics_d;
    int *radii_d;
    int32_t *num_tiles_hit_d;
    cudaMalloc((void **)&covs3d_d, num_cov3d * sizeof(float));
    cudaMalloc((void **)&xy_d, num_points * sizeof(float2));
    cudaMalloc((void **)&z_d, num_points * sizeof(float));
    cudaMalloc((void **)&radii_d, num_points * sizeof(int));
    cudaMalloc((void **)&conics_d, num_points * sizeof(float3));
    cudaMalloc((void **)&num_tiles_hit_d, num_points * sizeof(int32_t));

    printf("projecting into 2d...\n");
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
    cudaMemcpy(
        covs3d, covs3d_d, num_cov3d * sizeof(float), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(xy, xy_d, num_points * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, z_d, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        radii, radii_d, num_points * sizeof(int), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        num_tiles_hit,
        num_tiles_hit_d,
        num_points * sizeof(int32_t),
        cudaMemcpyDeviceToHost
    );

    printf("binning and sorting...\n");
    int32_t num_intersects;
    int32_t *cum_tiles_hit = new int32_t[num_points];
    int32_t *cum_tiles_hit_d;
    cudaMalloc((void **)&cum_tiles_hit_d, num_points * sizeof(int32_t));
    compute_cumulative_intersects(
        num_points, num_tiles_hit_d, num_intersects, cum_tiles_hit_d
    );
    printf("num_intersects %d\n", num_intersects);
    // cudaMemcpy(cum_tiles_hit, cum_tiles_hit_d, num_points * sizeof(int32_t),
    // cudaMemcpyDeviceToHost); for (int i = 0; i < num_points; ++i) {
    //     printf("num_tiles_hit %d, %d\n", i, num_tiles_hit[i]);
    // }

    int64_t *isect_ids_sorted_d;
    int32_t *gaussian_ids_sorted_d; // sorted by tile and depth
    int64_t *isect_ids_sorted = new int64_t[num_intersects];
    int32_t *gaussian_ids_sorted = new int32_t[num_intersects];
    cudaMalloc((void **)&isect_ids_sorted_d, num_intersects * sizeof(int64_t));
    cudaMalloc(
        (void **)&gaussian_ids_sorted_d, num_intersects * sizeof(int32_t)
    );

    int64_t *isect_ids_unsorted_d;
    int32_t *gaussian_ids_unsorted_d; // sorted by tile and depth
    int64_t *isect_ids_unsorted = new int64_t[num_intersects];
    int32_t *gaussian_ids_unsorted = new int32_t[num_intersects];
    cudaMalloc(
        (void **)&isect_ids_unsorted_d, num_intersects * sizeof(int64_t)
    );
    cudaMalloc(
        (void **)&gaussian_ids_unsorted_d, num_intersects * sizeof(int32_t)
    );

    int num_tiles = tile_bounds.x * tile_bounds.y;
    int2 *tile_bins_d; // start and end indices for each tile
    int2 *tile_bins = new int2[num_tiles];
    cudaMalloc((void **)&tile_bins_d, num_tiles * sizeof(int2));

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

    // cudaMemcpy(
    //     isect_ids_unsorted,
    //     isect_ids_unsorted_d,
    //     num_intersects * sizeof(int64_t),
    //     cudaMemcpyDeviceToHost
    // );
    // cudaMemcpy(
    //     gaussian_ids_unsorted,
    //     gaussian_ids_unsorted_d,
    //     num_intersects * sizeof(int32_t),
    //     cudaMemcpyDeviceToHost
    // );
    // cudaMemcpy(
    //     isect_ids_sorted,
    //     isect_ids_sorted_d,
    //     num_intersects * sizeof(int64_t),
    //     cudaMemcpyDeviceToHost
    // );
    cudaMemcpy(
        gaussian_ids_sorted,
        gaussian_ids_sorted_d,
        num_intersects * sizeof(int32_t),
        cudaMemcpyDeviceToHost
    );
    //
    // for (int i = 0; i < num_intersects; ++i) {
    //     printf("%d unsorted isect %016lx point %03d\n", i,
    //     isect_ids_unsorted[i], gaussian_ids_unsorted[i]);
    // }
    // for (int i = 0; i < num_intersects; ++i) {
    //     printf("sorted isect %016lx point %03d\n", isect_ids_sorted[i],
    //     gaussian_ids_sorted[i]);
    // }
    // cudaMemcpy(tile_bins, tile_bins_d, num_tiles * sizeof(int2),
    // cudaMemcpyDeviceToHost); for (int i = 0; i < num_tiles; ++i) {
    //     printf("tile_bins %d %d %d\t", i, tile_bins[i].x, tile_bins[i].y);
    // }
    // std::cout << std::endl;

    float *out_img = new float[C * W * H];
    float *out_img_d;
    float *final_Ts_d;
    int *final_idx_d;
    cudaMalloc((void **)&out_img_d, C * W * H * sizeof(float));
    cudaMalloc((void **)&final_Ts_d, W * H * sizeof(float));
    cudaMalloc((void **)&final_idx_d, W * H * sizeof(int));

    // const float background[3] = {1.0f, 1.0f, 1.0f};
    const float background[3] = {0.0f, 0.0f, 0.0f};
    float *background_d;
    cudaMalloc((void **)&background_d, C * sizeof(float));
    cudaMemcpy(
        background_d, background, C * sizeof(float), cudaMemcpyHostToDevice
    );

    printf("final rasterize pass\n");
    rasterize_forward_impl (
        tile_bounds,
        block,
        img_size,
        C,
        gaussian_ids_sorted_d,
        tile_bins_d,
        xy_d,
        conics_d,
        rgbs_d,
        opacities_d,
        final_Ts_d,
        final_idx_d,
        out_img_d,
        background_d
    );
    cudaMemcpy(
        out_img, out_img_d, C * W * H * sizeof(float), cudaMemcpyDeviceToHost
    );
    TGAImage image(W, H, TGAImage::RGB);
    int idx;
    float3 c;
    TGAColor col;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            idx = y * W + x;
            for (int c = 0; c < C; ++c) {
                col[c] = (uint8_t)(255.f * out_img[C * idx + c]);
            }
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
