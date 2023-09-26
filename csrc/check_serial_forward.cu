#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>

#include "forward.cuh"
#include "helpers.cuh"

#include "reference/cuda_rasterizer/forward.cu"

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

// Host version of ref preprocess
__host__ void host_preprocessCUDA(
    int P,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, 
    int H,
	const float tan_fovx,
     float tan_fovy,
	const float focal_x, 
    float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid, // tile_grid = ((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	uint32_t* tiles_touched)
{
    auto idx = 0;
	if (idx >= P)
		return;
    int C = 3;
	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
    
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view))
		return;

    //printf("ref: p_view %d %.2f %.2f %.2f\n", idx, p_view.x, p_view.y, p_view.z);

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}
	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
    //printf("ref: cov2d %d, %.2f %.2f %.2f\n", idx, cov.x, cov.y, cov.z);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
    //printf("ref: conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);

    // Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
    //printf("ref: point_image %d %.2f %.2f \n", idx, point_image.x, point_image.y);
	uint2 rect_min, rect_max;
    //printf("grid %d %d \n", grid.x, grid.y);
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		//return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		//glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		//rgb[idx * C + 0] = result.x;
		//rgb[idx * C + 1] = result.y;
		//rgb[idx * C + 2] = result.z;
	}
    
	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	//conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    int32_t tile_area = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    printf("ref rast: point %d x %.2f y %.2f z %.2f, radius %d, # tiles %d, tile_min %d %d, tile_max %d %d \n", idx, point_image.x, point_image.y, depths[idx], radii[idx], tile_area, rect_min.x, rect_min.y, rect_max.x, rect_max.y);
}

// Host version of diff rast projection kernel
__host__ __device__ void host_project_gaussians_forward_kernel(
    const int num_points,
    const float3 *means3d,
    const float3 *scales,
    const float glob_scale,
    const float4 *quats,
    const float *viewmat,
    const float *projmat,
    const float fx,
    const float fy,
    const dim3 img_size,
    const dim3 tile_bounds,
    float *covs3d,
    float2 *xys,
    float *depths,
    int *radii,
    float3 *conics,
    int32_t *num_tiles_hit
) {
    // unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    int idx = 0;
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
    if (idx >= num_points) {
        return;
    }

    float3 p_world = means3d[idx];
    //printf("diff rast: p_world %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y, p_world.z);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view)) {
        // printf("%d is out of frustum z %.2f, returning\n", idx, p_view.z);
        return;
    }
    //printf("diff rast: p_view %d %.2f %.2f %.2f\n", idx, p_view.x, p_view.y, p_view.z);

    // compute the projected covariance
    float3 scale = scales[idx];
    float4 quat = quats[idx];
    //printf("%d scale %.2f %.2f %.2f\n", idx, scale.x, scale.y, scale.z);
    //printf("%d quat %.2f %.2f %.2f %.2f\n", idx, quat.w, quat.x, quat.y, quat.z);
    float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(p_world, cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy);
    //printf("diff rast: cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    //printf("diff rast: conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;

    // compute the projected mean
    float2 center = project_pix(projmat, p_world, img_size);
    //printf("diff rast: point_image %d %.2f %.2f \n", idx, center.x, center.y);
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    //printf("diff rast: tile_min %d %d \n", tile_min.x, tile_min.y);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    xys[idx] = center;
    printf("diff rast: point %d x %.2f y %.2f z %.2f, radius %d, # tiles %d, tile_min %d %d, tile_max %d %d \n", idx, center.x, center.y, depths[idx], radii[idx], tile_area, tile_min.x, tile_min.y, tile_max.x, tile_max.y);
}

void compare_cov2d_forward(){
    // TODO: test with more than one point and varying cov3d/viewmat
    int num_points = 1;
    float fx = 1;
    float fy = 1;
    const int W = 256;
    const int H = 256;
    float tan_fovx = 0.5 * W / fx;
    float tan_fovy = 0.5 * H / fy;
    float viewmat[] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 8.f,
        0.f, 0.f, 0.f, 1.f};
    float viewmatColumnMajor[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 8.0f, 1.0f
    };
    float cov3d[]{
        1.f, 0.f, 0.f, 
        1.f, 0.f, 1.f
    };
    float3 mean = {1.f, 1.f, 1.f};

    // diff rast
    float3 diff_rast_cov2d = project_cov3d_ewa(mean,cov3d, viewmat, fx, fy, tan_fovx, tan_fovy);
    std::cout<<"diff rast cov2d: "<<diff_rast_cov2d.x<<" "<<diff_rast_cov2d.y<<" "<<diff_rast_cov2d.z<<std::endl; 
    
    // ref rast
    float3 res = computeCov2D(mean, fx, fy, tan_fovx, tan_fovy, cov3d, viewmatColumnMajor);
    std::cout<<"ref rast cov2d:  "<<res.x<<" "<<res.y<<" "<<res.z<<std::endl;
}

void compare_scale_rot_to_cov3d(){
    // TODO: make it work with more than one point
    int num_points = 1;
    float3 scale = {random_float(), random_float(), random_float()};
    float glob_scale = random_float();

    float4 quat = random_quat();

    float* cov3d = new float[6 * num_points];

    // diff rast
    scale_rot_to_cov3d(scale, glob_scale, quat, cov3d);   
    std::cout << "diff rast cov3d: " << cov3d[0]<<" "<<cov3d[1]<<" "<<cov3d[2]<<std::endl;

    // ref rast
    float* ref_cov3d = new float[6 * num_points];
    computeCov3D(glm::vec3({scale.x,scale.y,scale.z}), glob_scale, glm::vec4({quat.x, quat.y, quat.z, quat.w}), ref_cov3d);
    std::cout << "ref rast cov3d: " <<ref_cov3d[0]<<" "<<ref_cov3d[1]<<" "<<ref_cov3d[2]<<std::endl;
}

void compare_project_preprocess(){
    // setup
    const int num_points = 1;
    const float3 mean = {1.f, 1.f, 1.f};
    float3* means3d = new float3[num_points];
    means3d[0] = mean;
    const float3 scale = {random_float(), random_float(), random_float()};
    const float3* scales = &scale;
    const float glob_scale = 1;
    float4 temp_quat = random_quat();
    float4* quats = &temp_quat;
    const float viewmat[] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 8.f,
        0.f, 0.f, 0.f, 1.f};

    const float *viewmats = viewmat;
    const float *projmats = viewmat;
    
    float *covs3d = new float[num_points*6];
    float2 *xys = new float2[num_points*2];
    float *depths = new float[num_points];
    int *radii = new int[num_points];
    float3 *conics = new float3[num_points];
    int32_t *num_tiles_hit = new int32_t[num_points];

    const float fov_x = M_PI / 2.f;
    const int W = 512;
    const int H = 512;
    const float focal = 0.5 * (float)W / tan(0.5 * fov_x);
    float tan_fovx = 0.5 * W / focal;
    float tan_fovy = 0.5 * H / focal;
    const dim3 tile_bounds = {
        (W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1};
    const dim3 img_size = {W, H, 1};
    const dim3 block = {BLOCK_X, BLOCK_Y, 1};
    
    // diff rast
    host_project_gaussians_forward_kernel(
        num_points,
        means3d,
        scales,
        glob_scale,
        quats,
        viewmats,
        projmats,
        focal,
        focal,
        img_size,
        tile_bounds,
        covs3d,
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit
    );

    float *covs3d_ref = new float[num_points*6];
    float2 *xys_ref = new float2[num_points*2];
    float *depths_ref = new float[num_points];
    int *radii_ref = new int[num_points];
    float3 *conics_ref = new float3[num_points];
    uint32_t *num_tiles_hit_ref = new uint32_t[num_points];

    glm::vec3 scales_glm = glm::vec3({scale.x, scale.y, scale.z});
    glm::vec3* scales_ref = &scales_glm;
    glm::vec4 quat_glm = glm::vec4({temp_quat.x, temp_quat.y, temp_quat.z, temp_quat.w});
    glm::vec4* quat_ref = &quat_glm;
    glm::vec3* cam_pos  = nullptr;
    
    bool clamped_val = false;
    bool* clamped = &clamped_val;
    float* colors_precomp = nullptr;
    float* rgb = nullptr;
    float4* conic_opacity = nullptr;
    float* cov3D_precomp = nullptr;
    float* opacities = new float[num_points];
    opacities[0] = 0.9f;   
    float* shs = nullptr;
    float viewmatColumnMajor[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 8.0f, 1.0f
    };
    const float *viewmats_ref = viewmatColumnMajor;
    
    // ref rast
    host_preprocessCUDA(
    num_points,
    (float*)means3d,
    scales_ref,
    glob_scale,
	quat_ref,
	opacities,
	shs,
    clamped,
	cov3D_precomp,
	colors_precomp,
	viewmats_ref,
	viewmats_ref,
	cam_pos,
    W,
    H,
	tan_fovx,
    tan_fovy,
	focal,
    focal,
	radii_ref,
	xys_ref,
	depths_ref,
	covs3d_ref,
	rgb,
	conic_opacity,
	tile_bounds,
	num_tiles_hit_ref
    );
}
int main(){
    //compare_cov2d_forward();
    //compare_scale_rot_to_cov3d();
    compare_project_preprocess();
}
