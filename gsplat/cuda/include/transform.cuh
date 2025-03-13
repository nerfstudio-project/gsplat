#pragma once

#include "types.cuh"

// Transforms a 3D position from world coordinates to camera coordinates.
// [R | t] is the world-to-camera transformation.
inline __device__ void posW2C(
    const mat3 R,  
    const vec3 t, 
    const vec3 pW, // Input position in world coordinates
    vec3 &pC // Output position in camera coordinates
) {
    pC = R * pW + t;
}

// Computes the vector-Jacobian product (VJP) for posW2C.
// This function computes gradients of the transformation with respect to inputs.
inline __device__ void posW2C_VJP(
    // Forward inputs
    const mat3 R, 
    const vec3 t, 
    const vec3 pW, // Input position in world coordinates
    // Gradient output
    const vec3 v_pC, // Gradient of the output position in camera coordinates
    // Gradient inputs (to be accumulated)
    mat3 &v_R, // Gradient w.r.t. R
    vec3 &v_t, // Gradient w.r.t. t
    vec3 &v_pW // Gradient w.r.t. pW
) {
    // Using the rule for differentiating a linear transformation: 
    // For D = W * X, G = dL/dD
    // dL/dW = G * X^T, dL/dX = W^T * G
    v_R += glm::outerProduct(v_pC, pW);
    v_t += v_pC;
    v_pW += glm::transpose(R) * v_pC;
}

// Transforms a covariance matrix from world coordinates to camera coordinates.
inline __device__ void covarW2C(
    const mat3 R, 
    const mat3 covarW, // Input covariance matrix in world coordinates
    mat3 &covarC // Output covariance matrix in camera coordinates
) {
    covarC = R * covarW * glm::transpose(R);
}

// Computes the vector-Jacobian product (VJP) for covarW2C.
// This function computes gradients of the transformation with respect to inputs.
inline __device__ void covarW2C_VJP(    
    // Forward inputs
    const mat3 R, 
    const mat3 covarW, // Input covariance matrix in world coordinates
    // Gradient output
    const mat3 v_covarC, // Gradient of the output covariance matrix in camera coordinates
    // Gradient inputs (to be accumulated)
    mat3 &v_R, // Gradient w.r.t. rotation matrix
    mat3 &v_covarW // Gradient w.r.t. world covariance matrix
) {
    // Using the rule for differentiating quadratic forms: 
    // For D = W * X * W^T, G = dL/dD
    // dL/dX = W^T * G * W
    // dL/dW = G * W * X^T + G^T * W * X
    v_R += v_covarC * R * glm::transpose(covarW) + glm::transpose(v_covarC) * R * covarW;
    v_covarW += glm::transpose(R) * v_covarC * R;
}