#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <math.h>

#include "helpers.cuh"
#include "config.h"
#include "tgaimage.h"

float random_float() {
    return (float) std::rand() / RAND_MAX;
}

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

void test_quat_to_rotmat_jvp() {
    float4 quat = random_quat();
    printf("quat %.2e %.2e %.2e %.2e\n", quat.x, quat.y, quat.z, quat.w); 
    glm::vec4 quat_v = glm::vec4(quat.x, quat.y, quat.z, quat.w);
    glm::vec3 ones = glm::vec3(0.1f);
    glm::mat3 v_Rmat = glm::mat3(ones, ones, ones);
    // estimate columns of vjp
    glm::vec4 v_quat_est;
    glm::mat3 Rdiff;
    float eps = 1e-7f;
    glm::vec4 eps_v, quat_p, quat_m;

    for (int c = 0; c < 4; ++c) {
        eps_v = glm::vec4();
        eps_v[c] = eps;
        quat_p = quat_v + eps_v;
        quat_m = quat_v - eps_v;
        printf("eps_v %.2e %.2e %.2e %.2e\n", eps_v[0], eps_v[1], eps_v[2], eps_v[3]);
        Rdiff = quat_to_rotmat(
            {quat_p[0], quat_p[1], quat_p[2], quat_p[3]}
        ) - quat_to_rotmat(
            {quat_m[0], quat_m[1], quat_m[2], quat_m[3]}
        );
        v_quat_est[c] = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                v_quat_est[c] += v_Rmat[i][j] * Rdiff[i][j] * 0.5f / eps;
            }
        }
    }
    printf("v_quat_est %.2e %.2e %.2e %.2e\n", v_quat_est[0], v_quat_est[1], v_quat_est[2], v_quat_est[3]);
    float4 v_quat = quat_to_rotmat_vjp(quat, v_Rmat);
    printf("v_quat %.2e %.2e %.2e %.2e\n", v_quat.x, v_quat.y, v_quat.z, v_quat.w); 
}


int main() {
    test_quat_to_rotmat_jvp();
    return 0;
}
