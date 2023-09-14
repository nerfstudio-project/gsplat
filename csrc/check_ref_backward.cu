#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <math.h>

#include "backward.cuh"
#include "backward_ref.cuh"
#include "forward.cuh"
#include "helpers.cuh"

float random_float() { return (float)std::rand() / RAND_MAX; }

float4 random_quat() {
    float u = random_float();
    float v = random_float();
    float w = random_float();
    return {
        sqrt(1.f - u) * sin(2.f * (float)M_PI * v),
        sqrt(1.f - u) * cos(2.f * (float)M_PI * v),
        sqrt(u) * sin(2.f * (float)M_PI * w),
        sqrt(u) * cos(2.f * (float)M_PI * w)};
}

float3 compute_conic(const float3 cov2d) {
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return {0.f, 0.f, 0.f};
    float inv_det = 1.f / det;
    float3 conic;
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;
    return conic;
}

bool outside_error (float x, float y, float acceptable_error=1e-6) {
    return abs(x - y) > acceptable_error;
};

void compare_project2d_mean_backward() {
    float3 mean = {random_float(), random_float(), random_float()};
    // clang-format off
    float proj[] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    // clang-format on
    float2 dL_dmean2d = {random_float(), random_float()};

    float3 dL_dmean = project_pix_vjp(proj, mean, (dim3){1, 1, 1}, dL_dmean2d);
    float3 dL_dmean_ref = projectMean2DBackward(mean, proj, dL_dmean2d);

    const bool is_dL_dmean_off {
        outside_error(dL_dmean.x, dL_dmean_ref.x) ||
        outside_error(dL_dmean.y, dL_dmean_ref.y) ||
        outside_error(dL_dmean.z, dL_dmean_ref.z)
    };

    if (is_dL_dmean_off) {

        std::cout << "dL_dmean is off!\n"
        << "-------------------------------\n";

        printf("project2d backward\n");
        printf("dL_dmean\n");
        printf("ours %.2e %.2e %.2e\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
        printf(
            "ref  %.2e %.2e %.2e\n", dL_dmean_ref.x, dL_dmean_ref.y, dL_dmean_ref.z
        );

        std::cout << "\n\n";
    }

}

void compare_conic_backward() {
    float3 cov2d = {random_float(), random_float(), random_float()};
    float3 conic = compute_conic(cov2d);
    float3 dL_dconic = {random_float(), random_float(), random_float()};
    float3 dL_dcov2d;
    cov2d_to_conic_vjp(conic, dL_dconic, dL_dcov2d);
    float3 dL_dcov2d_ref = computeConicBackward(cov2d, dL_dconic);

    // Jonathan Zakharov:
    // TODO: Figure out why this is sometimes giving false positives
    //       or why fires off but all values are equal when printed out
    // IDEA: Maybe there is no bug, just that error thershold is finer
    //       than the float precision we're displaying
    const bool is_dL_dcov2d_off {
        outside_error(dL_dcov2d.x, dL_dcov2d_ref.x) ||
        outside_error(dL_dcov2d.y, dL_dcov2d_ref.y) ||
        outside_error(dL_dcov2d.z, dL_dcov2d_ref.z)
    };

    if (is_dL_dcov2d_off) {

        std::cout << "dL_dcov2d is off!\n" 
            << "-------------------\n";

        printf("dL_dcov2d\n");
        printf("ours %.2e %.2e %.2e\n", dL_dcov2d.x, dL_dcov2d.y, dL_dcov2d.z);
        printf(
            "ref  %.2e %.2e %.2e\n",
            dL_dcov2d_ref.x,
            dL_dcov2d_ref.y,
            dL_dcov2d_ref.z
        );

        std::cout << "\n\n";
    }

}

void compare_cov3d_backward() {
    float3 scale = {random_float(), random_float(), random_float()};
    float4 quat = random_quat();
    float4 quat_ref = {quat.w, quat.x, quat.y, quat.z};
    float dL_dcov3d[] = {
        random_float(),
        random_float(),
        random_float(),
        random_float(),
        random_float(),
        random_float()};
    float3 dL_ds = {0.f, 0.f, 0.f};
    float4 dL_dq = {0.f, 0.f, 0.f, 0.f};
    scale_rot_to_cov3d_vjp(scale, 1.f, quat, dL_dcov3d, dL_ds, dL_dq);

    float3 dL_ds_ref = {0.f, 0.f, 0.f};
    float4 dL_dq_ref = {0.f, 0.f, 0.f, 0.f};
    computeCov3DBackward(scale, 1.f, quat_ref, dL_dcov3d, dL_ds_ref, dL_dq_ref);

    const bool is_dL_dscale_off {
        outside_error(dL_ds.x, dL_ds_ref.x) ||
        outside_error(dL_ds.y, dL_ds_ref.y) ||
        outside_error(dL_ds.z, dL_ds_ref.z)
    };

    if (is_dL_dscale_off) {

        std::cout << "dL_dscale is off!\n"
            << "---------------------\n";

        printf("dL_dscale\n");
        printf("ours %.2e %.2e %.2e\n", dL_ds.x, dL_ds.y, dL_ds.z);
        printf("ref  %.2e %.2e %.2e\n", dL_ds_ref.x, dL_ds_ref.y, dL_ds_ref.z);

        std::cout << "\n\n";
    }

    const bool is_dL_dquat_off {
        outside_error(dL_dq.x, dL_dq_ref.x) ||
        outside_error(dL_dq.y, dL_dq_ref.y) ||
        outside_error(dL_dq.z, dL_dq_ref.z) ||
        outside_error(dL_dq.w, dL_dq_ref.w)
    };

    if (is_dL_dquat_off) {

        std::cout << "dL_dquat is off!\n"
            << "-----------------\n";

        printf("dL_dquat\n");
        printf("ours %.2e %.2e %.2e %.2e\n", dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w);
        printf(
            "ref  %.2e %.2e %.2e %.2e\n",
            dL_dq_ref.x,
            dL_dq_ref.y,
            dL_dq_ref.z,
            dL_dq_ref.w
        );

        std::cout << "\n\n";
    }
}

void compare_cov2d_ewa_backward() {
    float3 mean = {random_float(), random_float(), random_float()};
    float3 scale = {random_float(), random_float(), random_float()};
    float4 quat = random_quat();
    float cov3d[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    scale_rot_to_cov3d(scale, 1.f, quat, cov3d);
    float3 dL_dcov2d = {random_float(), random_float(), random_float()};
    float3 dL_dmean, dL_dmean_ref;
    float dL_dcov[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float dL_dcov_ref[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // functions expect different view matrix convention
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
    float viewmat_ref[] = {
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
        0.f,
        0.f,
        0.f,
        10.f,
        1.f};
    computeCov2DBackward(
        mean,
        cov3d,
        viewmat_ref,
        1.f,
        1.f,
        1.f,
        1.f,
        dL_dcov2d,
        dL_dmean_ref,
        dL_dcov_ref
    );
    project_cov3d_ewa_vjp(
        mean, cov3d, viewmat, 1.f, 1.f, dL_dcov2d, dL_dmean, dL_dcov
    );
    
    const bool is_dL_dmean_off {
        outside_error(dL_dmean.x, dL_dmean_ref.x) ||
        outside_error(dL_dmean.y, dL_dmean_ref.y) ||
        outside_error(dL_dmean.z, dL_dmean_ref.z)
    };

    //printf("cov2d_ewa backward\n");
    if (is_dL_dmean_off) {

        std::cout << "dL_dmean is off!\n"
            << "-----------------\n";

        printf("dL_dmean\n");
        printf("ours %.2e %.2e %.2e\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
        printf(
            "ref  %.2e %.2e %.2e\n", dL_dmean_ref.x, dL_dmean_ref.y, dL_dmean_ref.z
        );

        std::cout << "\n\n";
    }

    bool is_dL_dcov_off{ false };
    for (int i = 0; i < 6; ++i) // if any of dL_dcov[i] is off
        if (outside_error(dL_dcov[i], dL_dcov_ref[i])) {
            is_dL_dcov_off = true;
            break;
        }

    if (is_dL_dcov_off) {   
        std::cout << "dL_dcov is off!\n"
            << "-----------------\n";

        printf("dL_dcov\nours ");
        for (int i = 0; i < 6; ++i)
            printf("%.2e ", dL_dcov[i]);

        printf("\ntheirs ");
        for (int i = 0; i < 6; ++i)
            printf("%.2e ", dL_dcov_ref[i]);

        std::cout << "\n\n";
    }
}

void compare_rasterize_backward() {
    constexpr int N = 8;
    float2 p = {0.f, 0.f};
    float T_final = 5e-3;
    constexpr int C = 3;
    float dL_dout[C];
    for (int i = 0; i < C; ++i) {
        dL_dout[i] = random_float();
    }

    float2 xys[N];
    float3 conics[N];
    float opacities[N];
    float4 conics_o[N];
    float rgbs[C * N];
    for (int i = 0; i < N; ++i) {
        float v = (float)i - (float)N * 0.5f;
        xys[i] = {0.1f * v, 0.1f * v};
        conics[i] = {10.f, 0.f, 10.f};
        opacities[i] = 0.5f;
        conics_o[i] = {conics[i].x, conics[i].y, conics[i].z, opacities[i]};
        for (int c = 0; c < C; ++c) {
            rgbs[C * i + c] = random_float();
        }
    }
    float dL_drgb[C * N] = {0.f};
    float dL_drgb_ref[C * N] = {0.f};
    float dL_do[N] = {0.f};
    float dL_do_ref[N] = {0.f};
    float2 dL_dm[N] = {0.f};
    float2 dL_dm_ref[N] = {0.f};
    float3 dL_dc[N] = {0.f};
    float3 dL_dc_ref[N] = {0.f};

    rasterize_vjp(
        N,
        p,
        C,
        xys,
        conics,
        opacities,
        rgbs,
        T_final,
        dL_dout,
        dL_drgb,
        dL_do,
        dL_dm,
        dL_dc
    );
    rasterizeBackward(
        N,
        2,
        2,
        p,
        xys,
        conics_o,
        rgbs,
        T_final,
        dL_dout,
        dL_drgb_ref,
        dL_do_ref,
        dL_dm_ref,
        dL_dc_ref
    );

    //printf("rasterize backward\n");
    //printf("dL_dout %.2e %.2e %.2e\n", dL_dout[0], dL_dout[1], dL_dout[2]);
    for (int i = 0; i < N; ++i) {
        const auto something_went_wrong = [&]() {
            printf("\n");
            printf("\n");
            printf("Something went wrong!");
            printf("\n");
            printf("\n");
            printf("dL_drgb %d\n", i);
            printf(
                "ours %.2e %.2e %.2e\n",
                dL_drgb[C * i + 0],
                dL_drgb[C * i + 1],
                dL_drgb[C * i + 2]
            );
            printf(
                "theirs %.2e %.2e %.2e\n",
                dL_drgb_ref[C * i + 0],
                dL_drgb_ref[C * i + 1],
                dL_drgb_ref[C * i + 2]
            );
            printf("\n");
            printf("dL_do %d\n", i);
            printf("ours %.2e\n", dL_do[i]);
            printf("theirs %.2e\n", dL_do_ref[i]);
            printf("\n");
            printf("dL_dm %d\n", i);
            printf("ours %.2e %.2e\n", dL_dm[i].x, dL_dm[i].y);
            printf("theirs %.2e %.2e\n", dL_dm_ref[i].x, dL_dm_ref[i].y);
            printf("\n");
            printf("dL_dc %d\n", i);
            printf("ours %.2e %.2e %.2e\n", dL_dc[i].x, dL_dc[i].y, dL_dc[i].z);
            printf(
                "theirs %.2e %.2e %.2e\n",
                dL_dc_ref[i].x,
                dL_dc_ref[i].y,
                dL_dc_ref[i].z
            );
            printf("\n");
        };

        const bool is_dL_drgb_off{
            outside_error(dL_drgb[C * i + 0], dL_drgb_ref[C * i + 0]) ||
            outside_error(dL_drgb[C * i + 1], dL_drgb_ref[C * i + 1]) ||
            outside_error(dL_drgb[C * i + 2], dL_drgb_ref[C * i + 2]) 
        };
            
        const bool is_dL_do_off{
            outside_error(dL_do[i], dL_do_ref[i])
        };

        const bool is_dL_dm_off{
            outside_error(dL_dm[i].x, dL_dm_ref[i].x) ||
            outside_error(dL_dm[i].y, dL_dm_ref[i].y)
        };

        const bool is_dL_dc_off{
            outside_error(dL_dc[i].x, dL_dc_ref[i].x) ||
            outside_error(dL_dc[i].y, dL_dc_ref[i].y) ||
            outside_error(dL_dc[i].z, dL_dc_ref[i].z)
        };

        const bool error_occured{
            is_dL_drgb_off ||
            is_dL_do_off   ||
            is_dL_dm_off   ||
            is_dL_dc_off
        };

        if (error_occured) {
            something_went_wrong();
            assert(false);
        }
    }
}

int main() {
    for (int x = 0; x < 100000; x++) {
        std::cout << "<Check " << x << ">\n"
            << "=====================\n";
        compare_project2d_mean_backward();
        compare_conic_backward();
        compare_cov3d_backward();
        compare_cov2d_ewa_backward();
        compare_rasterize_backward();
    }
    return 0;
}
