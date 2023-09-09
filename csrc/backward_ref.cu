#include <iostream>
#include "backward_ref.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cooperative_groups.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)


inline __host__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}


inline __host__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__host__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__host__ __device__ float3 projectMean2DBackward(
    const float3 m, const float* proj, const float2 dL_dmean2D
) {

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	float3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D.x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D.y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D.x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D.y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D.x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D.y;
    return dL_dmean;
}


__host__ __device__ void computeCov3DBackward(
    const float3 scale,
    const float mod,
    const float4 q,
    const float* dL_dcov3D,
    float3 &dL_dscale,
    float4 &dL_dq
) {
	// Recompute (intermediate) results for the 3D covariance computation.
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	S[0][0] = scale.x * mod;
	S[1][1] = scale.y * mod;
	S[2][2] = scale.z * mod;

	glm::mat3 M = S * R;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	dL_dscale.x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale.y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale.z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= scale.x * mod;
	dL_dMt[1] *= scale.y * mod;
	dL_dMt[2] *= scale.z * mod;

	// Gradients of loss w.r.t. normalized quaternion
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
}


__host__ __device__ float3 computeConicBackward(
    const float3 &cov2D,
    const float3 &dL_dconic
) {

	// float a = cov2D.x + 0.1f;
	// float b = cov2D.y;
	// float c = cov2D.z + 0.1f;

	float a = cov2D.x;
	float b = cov2D.y;
	float c = cov2D.z;

	float denom = a * c - b * b;
	float dL_da = 0;
    float dL_db = 0;
    float dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);
    }
    return {dL_da, dL_db, dL_dc};
}


__host__ __device__ void computeCov2DBackward(
    const float3 &mean,
    const float *cov3D,
    const float *view_matrix,
    const float h_x,
    const float h_y,
	const float tan_fovx,
    const float tan_fovy,
    const float3 &dL_dcov2D,
    float3 &dL_dmean,
    float *dL_dcov
) {
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	// t.x = min(limx, max(-limx, txtz)) * t.z;
	// t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
    float dL_da = dL_dcov2D.x;
    float dL_db = dL_dcov2D.y;
    float dL_dc = dL_dcov2D.z;
    // printf("%.2e %.2e %.2e\n", dL_da, dL_db, dL_dc);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
    // given gradients w.r.t. 2D covariance matrix (diagonal).
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
    dL_dcov[3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
    dL_dcov[5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
    // Off-diagonal elements appear twice --> double the gradient.
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
    dL_dcov[2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
    dL_dcov[4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;
    // printf("dL_dT\n");
    // printf("%.2e %.2e %.2e\n", dL_dT00, dL_dT01, dL_dT02);
    // printf("%.2e %.2e %.2e\n", dL_dT10, dL_dT11, dL_dT12);

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);
}


// test function rasterizing N gaussians in 1 pixel
__host__ __device__ void rasterizeBackward(
    const int N,
    const int W,
    const int H,
    const float2 pixf,
    const float2 *collected_xy,
    const float4 *collected_conic_opacity,
    const float *collected_colors,
    const float T_final,
    const float *dL_dpixel,
    float *dL_dcolor,
    float *dL_dopacity,
    float2 *dL_dmean2D,
    float3 *dL_dconic2D
) {
	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	float T = T_final;
    const int C = 3;

	float accum_rec[C] = { 0.f };
	float last_color[C] = { 0 };
	float last_alpha = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

    // Iterate over Gaussians
    for (int j = 0; j < N; j++)
    {
        // Compute blending values, as before.
        const float2 xy = collected_xy[j];
        const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
        const float4 con_o = collected_conic_opacity[j];
        const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;

        const float G = exp(power);
        const float alpha = min(0.99f, con_o.w * G);
        if (alpha < 1.0f / 255.0f)
            continue;

        T = T / (1.f - alpha);
        const float dchannel_dcolor = alpha * T;

        // Propagate gradients to per-Gaussian colors and keep
        // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
        // pair).
        float dL_dalpha = 0.0f;
        // const int global_id = collected_id[j];
        for (int ch = 0; ch < C; ch++)
        {
            const float c = collected_colors[C * j + ch];
            // Update last color (to be used in the next iteration)
            accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
            last_color[ch] = c;

            const float dL_dchannel = dL_dpixel[ch];
            dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
            // Update the gradients w.r.t. color of the Gaussian. 
            // Atomic, since this pixel is just one of potentially
            // many that were affected by this Gaussian.
            dL_dcolor[C * j + ch] += dchannel_dcolor * dL_dchannel;
            // atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
        }
        dL_dalpha *= T;
        // printf("ref %d dL_dalpha %.2e alpha %.2e T %.2e\n", j, dL_dalpha, alpha, T);
        // Update last alpha (to be used in the next iteration)
        last_alpha = alpha;

        // Account for fact that alpha also influences how much of
        // the background color is added if nothing left to blend
        // float bg_dot_dpixel = 0;
        // for (int i = 0; i < C; i++)
        //     bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
        // dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


        // Helpful reusable temporary variables
        const float dL_dG = con_o.w * dL_dalpha;
        const float gdx = G * d.x;
        const float gdy = G * d.y;
        const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
        const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

        // Update gradients w.r.t. 2D mean position of the Gaussian
        dL_dmean2D[j].x += dL_dG * dG_ddelx * ddelx_dx;
        dL_dmean2D[j].y += dL_dG * dG_ddely * ddely_dy;
        // atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
        // atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

        // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
        dL_dconic2D[j].x = -0.5f * gdx * d.x * dL_dG;
        dL_dconic2D[j].y = -0.5f * gdx * d.y * dL_dG;
        dL_dconic2D[j].z = -0.5f * gdy * d.y * dL_dG;
        // atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
        // atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
        // atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

        // Update gradients w.r.t. opacity of the Gaussian
        dL_dopacity[j] += G * dL_dalpha;
        // atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
    }
}

// copying the part that's relevant from our rasterize method
__host__ __device__ void rasterize_vjp(
    const int N,
    const float2 p,
    const int channels,
    const float2 *xys,
    const float3 *conics,
    const float *opacities,
    const float *rgbs,
    const float T_final,
    const float *v_out,
    float *v_rgb,
    float *v_opacity,
    float2 *v_xy,
    float3 *v_conic
) {
    float T = T_final;
    // float S[channels] = {0.f};
    float *S = new float[channels];
    float3 conic;
    float2 center, delta;
    float sigma, vis, fac, opac, alpha, ra;
    float v_alpha, v_sigma;

    for (int g = 0; g < N; ++g) {
        conic = conics[g];
        center = xys[g];
        delta = {center.x - p.x, center.y - p.y};
        sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) -
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        opac = opacities[g];
        vis = exp(-sigma);
        alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }
        ra = 1.f / (1.f - alpha);

        // compute the current T for this gaussian
        T *= ra;
        // update v_rgb for this gaussian
        fac = alpha * T;
        v_alpha = 0.f;
        for (int c = 0; c < channels; ++c) {
            v_rgb[channels * g + c] += fac * v_out[c];
            v_alpha += (rgbs[channels * g + c] * T - S[c] * ra) * v_out[c];
            // update contribution from back
            S[c] += rgbs[channels * g + c] * fac;
        }

        // update v_opacity for this gaussian
        v_opacity[g] += vis * v_alpha;
        // atomicAdd(&(v_opacity[g]), vis * v_alpha);

        // compute vjps for conics and means
        // d_sigma / d_delta = conic * delta
        // d_sigma / d_conic = delta * delta.T
        v_sigma = -opac * vis * v_alpha;

        v_conic[g].x += 0.5f * v_sigma * delta.x * delta.x;
        v_conic[g].y += 0.5f * v_sigma * delta.x * delta.y;
        v_conic[g].z += 0.5f * v_sigma * delta.y * delta.y;
        // atomicAdd(&(v_conic[g].x), v_sigma * delta.x * delta.x);
        // atomicAdd(&(v_conic[g].y), v_sigma * delta.x * delta.y);
        // atomicAdd(&(v_conic[g].z), v_sigma * delta.y * delta.y);

        v_xy[g].x += v_sigma * (conic.x * delta.x + conic.y * delta.y);
        v_xy[g].y += v_sigma * (conic.y * delta.x + conic.z * delta.y);
        // atomicAdd(
        //     &(v_xy[g].x),
        //     v_sigma * (conic.x * delta.x + conic.y * delta.y)
        // );
        // atomicAdd(
        //     &(v_xy[g].y),
        //     v_sigma * (conic.y * delta.x + conic.z * delta.y)
        // );
    }
    delete S;
}
