#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>

#include "backward.cuh"
#include "forward.cuh"
#include "helpers.cuh"
#include "serial_backward.cuh"

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

float percent_error(float exp_val, float true_val) {
    return (abs(true_val - exp_val) / true_val) * 100.f;
}

// takes a vector of float pairs, where first value is experimental,
// and second is true value
std::vector<float>
percent_errors(const std::vector<std::pair<float, float>> &values) {
    std::vector<float> p_errors;

    for (int i = 0; i < values.size(); ++i) {
        const float exp_val = values[i].first;
        const float true_val = values[i].second;
        const float p_error{percent_error(exp_val, true_val)};
        p_errors.push_back(p_error);
    }

    return p_errors;
}

// takes a vector of float pairs, where pair's first value is
// "experimental value", and second is "true value". If any pair's percent
// exceeds error threshold, the entire vector will be displayed in a nice lil
// format
void print_errors(
    const std::vector<std::pair<float, float>> &values,
    const std::string &name,
    float error_threshold = 0.01f // percent error (0.01%)
) {
    const std::vector<float> p_errors{percent_errors(values)};

    // only print if input is unexpected
    bool data_within_error{true};
    for (float p : p_errors)
        if (p > error_threshold)
            data_within_error = false;

    if (data_within_error)
        return;

    // format cout for how we want to display data
    std::cout << std::setprecision(2);
    ;

    std::cout << name << ":\n"
              << "     ours:     refs:\n";

    for (int i = 0; i < values.size(); ++i) {
        const float d1 = values[i].first;
        const float d2 = values[i].second;
        std::cout << '[' << i << "]: " << std::scientific << std::setw(10) << d1
                  << ", " << std::setw(10) << d2
                  << "\t(percent error=" << std::fixed << p_errors[i] << ")\n";
    }
    std::cout << '\n';
}

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

    float3 dL_dmean = project_pix_vjp(proj, mean, (dim3){2, 2, 1}, dL_dmean2d);
    float3 dL_dmean_ref = projectMean2DBackward(mean, proj, dL_dmean2d);

    // comparison
    const std::vector<std::pair<float, float>> dmean_data{
        {dL_dmean.x, dL_dmean_ref.x},
        {dL_dmean.y, dL_dmean_ref.y},
        {dL_dmean.z, dL_dmean_ref.z},
    };
    print_errors(dmean_data, "dmean (project2d)");
}

void compare_conic_backward() {
    float3 cov2d = {random_float(), random_float(), random_float()};
    float3 conic = compute_conic(cov2d);
    float3 dL_dconic = {random_float(), random_float(), random_float()};
    float3 dL_dcov2d;
    cov2d_to_conic_vjp(conic, dL_dconic, dL_dcov2d);
    float3 dL_dcov2d_ref = computeConicBackward(cov2d, dL_dconic);

    // comparison
    const std::vector<std::pair<float, float>> dcov2d_data{
        {dL_dcov2d.x, dL_dcov2d_ref.x},
        {dL_dcov2d.y, dL_dcov2d_ref.y},
        {dL_dcov2d.z, dL_dcov2d_ref.z},
    };
    print_errors(dcov2d_data, "dcov2d (conic)");
}

void compare_cov3d_backward() {
    float3 scale = {random_float(), random_float(), random_float()};
    float4 quat = random_quat();
    float4 quat_ref = quat;
    // float4 quat_ref = {quat.w, quat.x, quat.y, quat.z};
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

    // comparison
    const std::vector<std::pair<float, float>> ds_data{
        {dL_ds.x, dL_ds_ref.x},
        {dL_ds.y, dL_ds_ref.y},
        {dL_ds.z, dL_ds_ref.z},
    };
    print_errors(ds_data, "ds (cov3d)");

    const std::vector<std::pair<float, float>> dquat_data{
        // {dL_dq.x, dL_dq_ref.y},
        // {dL_dq.y, dL_dq_ref.z},
        // {dL_dq.z, dL_dq_ref.w},
        // {dL_dq.w, dL_dq_ref.x},
        {dL_dq.x, dL_dq_ref.x},
        {dL_dq.y, dL_dq_ref.y},
        {dL_dq.z, dL_dq_ref.z},
        {dL_dq.w, dL_dq_ref.w},
    };
    print_errors(dquat_data, "dquat (cov3d)");
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

    // comparison
    const std::vector<std::pair<float, float>> dmean_data{
        {dL_dmean.x, dL_dmean_ref.x},
        {dL_dmean.x, dL_dmean_ref.x},
        {dL_dmean.x, dL_dmean_ref.x},
    };
    print_errors(dmean_data, "dmean (cov2d_ewa)");

    std::vector<std::pair<float, float>> dcov_data;
    for (int i = 0; i < 6; ++i)
        dcov_data.push_back({dL_dcov[i], dL_dcov_ref[i]});

    print_errors(dcov_data, "dcov (cov2d_ewa)");
}

void compare_rasterize_backward(const int N) {
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

    // comparison
    std::vector<std::pair<float, float>> drgb_data;
    for (int i = 0; i < C * N; ++i)
        drgb_data.push_back({dL_drgb[i], dL_drgb_ref[i]});
    print_errors(drgb_data, "drgb (rasterize)");

    std::vector<std::pair<float, float>> do_data;
    for (int i = 0; i < N; ++i)
        do_data.push_back({dL_do[i], dL_do_ref[i]});
    print_errors(do_data, "do (rasterize)");

    std::vector<std::pair<float, float>> dm_data;
    for (int i = 0; i < N; ++i) {
        dm_data.push_back({dL_dm[i].x, dL_dm_ref[i].x});
        dm_data.push_back({dL_dm[i].y, dL_dm_ref[i].y});
    }
    print_errors(dm_data, "dm (rasterize)");

    std::vector<std::pair<float, float>> dc_data;
    for (int i = 0; i < N; ++i) {
        dc_data.push_back({dL_dc[i].x, dL_dc_ref[i].x});
        dc_data.push_back({dL_dc[i].y, dL_dc_ref[i].y});
        dc_data.push_back({dL_dc[i].z, dL_dc_ref[i].z});
    }
    print_errors(dc_data, "dc (rasterize)");
}

int main(int argc, char *argv[]) {
    int num_iters = 1000;
    if (argc > 1) {
        num_iters = std::stoi(argv[1]);
    }
    int num_points = 8;
    if (argc > 2) {
        num_points = std::stoi(argv[2]);
    }
    for (int x = 0; x < num_iters; x++) {
        std::cout << "<Check " << x << ">\n"
                  << "=====================\n";
        compare_project2d_mean_backward();
        compare_conic_backward();
        compare_cov3d_backward();
        compare_cov2d_ewa_backward();
        compare_rasterize_backward(num_points);
    }
    return 0;
}
