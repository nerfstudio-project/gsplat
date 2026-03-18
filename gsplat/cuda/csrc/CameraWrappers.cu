/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file CameraWrappers.cu
 * @brief C++ camera model implementations with CUDA kernels
 */

#if BUILD_CAMERA_WRAPPERS

#include "CameraWrappers.h"
#include "TensorView.h"
#include <c10/cuda/CUDAStream.h>
#include "Ops.h"

namespace gsplat {
/**
 * @brief Reshapes a tensor to match a specified shape template.
 *
/**
 * @brief Normalizes the shape of an input tensor according to a specified shape template.
 *
 * This function adjusts the dimensions of the provided tensor to match the number of dimensions
 * specified by the template parameters. If the tensor has fewer dimensions,
 * leading singleton dimensions are prepended. If it has more, leading dimensions are collapsed.
 * Optionally, expands the batch size if provided.
 *
 * @tparam SHAPE The sequence of integers representing the target shape (for checking only).
 * @param tensor Input tensor to be reshaped.
 * @param batch_size (Optional) If provided and non-negative, will expand the first dimension to this batch size.
 * @param tensor_name (Optional) Name of the tensor for error messages.
 * @return torch::Tensor Tensor reshaped to the normalized shape.
 */

template <auto... SHAPE>
torch::Tensor normalize_shape(torch::Tensor tensor, std::string_view tensor_name = "Tensor", int batch_size=-1)
{
    constexpr int expected_ndims = sizeof...(SHAPE);

    return [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
        // Reshape by prepending 1's if too few dimensions
        if(tensor.dim() < expected_ndims)
        {
            tensor = tensor.expand({((Is < expected_ndims - tensor.dim()) ? 1 : -1)...});
        }

        // Flatten leftmost dimensions if too many, or use as-is
        if(tensor.dim() > expected_ndims)
        {
            // Use view instead of flatten to make sure the tensor data won't be changed.
            tensor = tensor.view({(Is==0 ? -1 : tensor.size(tensor.dim() - expected_ndims + Is))...});
        }

        // Expand batch size if provided and needed
        if(batch_size >= 0 && (tensor.size(0) == 1 || tensor.stride(0) == 0))
        {
            tensor = tensor.expand({(Is==0 ? batch_size : -1)...});
        }

        TORCH_CHECK_ARG(tensor.dim() == expected_ndims, tensor_name, "(shape: ", torch::IntArrayRef(tensor.sizes()), "): has ", tensor.dim(), " dimensions, expected ", expected_ndims);
        TORCH_CHECK_ARG(batch_size < 0 || tensor.size(0) == batch_size, tensor_name, "(shape: ", torch::IntArrayRef(tensor.sizes()), "): batch size mismatch: expected ", batch_size, ", got ", tensor.size(0));

        return tensor;
    }(std::make_index_sequence<expected_ndims>{});
}

// Moved from TensorView.h with added batch_size validation
template<typename T, auto... SHAPE>
TensorView<T, SHAPE...> make_tensor_view(
    torch::Tensor tensor,
    int batch_size,
    std::string_view tensor_name = "Tensor")
{
    using NonConstT = std::decay_t<T>;

    // CUDA check (NEW)
    TORCH_CHECK(tensor.is_cuda(), tensor_name, " must be CUDA tensor");

    // Dtype check (EXISTING)
    TORCH_CHECK_TYPE(tensor.dtype() == torch::CppTypeToScalarType<NonConstT>::value,
        tensor_name, " dtype mismatch: expected ", torch::CppTypeToScalarType<NonConstT>::value,
        ", got ", tensor.dtype());

    constexpr int expected_ndims = TensorView<T, SHAPE...>::ndims;

    tensor = normalize_shape<SHAPE...>(tensor, tensor_name, batch_size);

    // Use the pointer overload which will perform dimension validation
    return [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
        // EXISTING: Call raw pointer overload
        return gsplat::make_tensor_view<SHAPE...>(
            const_cast<T*>(tensor.data_ptr<NonConstT>()),
            {tensor.size(Is)...},
            {tensor.stride(Is)...},
            tensor_name
        );
    }(std::make_index_sequence<expected_ndims>{});
}

// Optional tensor overload
template<typename T, auto... SHAPE>
TensorView<T, SHAPE...> make_tensor_view(
    const std::optional<torch::Tensor>& tensor,
    int batch_size,
    std::string_view tensor_name = "Tensor")
{
    return tensor
        ? make_tensor_view<T, SHAPE...>(*tensor, batch_size, tensor_name)
        : TensorView<T, SHAPE...>{};
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do \
    { \
        cudaError_t error = call; \
        if (error != cudaSuccess) \
        { \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(error) + \
                " at " + __FILE__ + ":" + std::to_string(__LINE__) \
            ); \
        } \
    } while(0)

// Dimension tags for method kernels
constexpr struct ray_tag_t {} RAY;
constexpr struct coeff_tag_t {} COEFF;
constexpr struct point_tag_t {} POINT;
constexpr struct camera_tag_t {} CAMERA;

// ==================== PyBaseCameraModel<> Implementations ====================

c10::intrusive_ptr<PyBaseCameraModel<>> PyBaseCameraModel<>::create(
    int width,
    int height,
    const std::string& camera_model,
    const torch::Tensor& principal_points,
    const std::optional<torch::Tensor> &focal_lengths,
    const std::optional<torch::Tensor> &radial_coeffs,
    const std::optional<torch::Tensor> &tangential_coeffs,
    const std::optional<torch::Tensor> &thin_prism_coeffs,
    const std::optional<c10::intrusive_ptr<FThetaCameraDistortionParameters>> &ftheta_coeffs,
    ShutterType rs_type
)
{
    if (camera_model == "pinhole")
    {
        TORCH_CHECK_ARG(!ftheta_coeffs, "ftheta_coeffs", "not allowed for pinhole camera model");
        TORCH_CHECK_ARG(focal_lengths, "focal_lengths", "required for pinhole camera model");

        if(radial_coeffs || tangential_coeffs || thin_prism_coeffs)
        {
            return c10::make_intrusive<PyOpenCVPinholeCameraModel>(
                width, height, *focal_lengths, principal_points, radial_coeffs, tangential_coeffs, thin_prism_coeffs, rs_type
            );
        }
        else
        {
            return c10::make_intrusive<PyPerfectPinholeCameraModel>(
                width, height, *focal_lengths, principal_points, rs_type
            );
        }
    }
    else if (camera_model == "fisheye")
    {
        TORCH_CHECK_ARG(!ftheta_coeffs, "ftheta_coeffs", "not allowed for pinhole camera model");
        TORCH_CHECK_ARG(!tangential_coeffs, "tangential_coeffs", "not allowed for fisheye camera model");
        TORCH_CHECK_ARG(!thin_prism_coeffs, "thin_prism_coeffs", "not allowed for fisheye camera model");
        TORCH_CHECK_ARG(focal_lengths, "focal_lengths", "required for fisheye camera model");

        return c10::make_intrusive<PyOpenCVFisheyeCameraModel>(
            width, height, *focal_lengths, principal_points, radial_coeffs, rs_type
        );
    }
    else if (camera_model == "ftheta")
    {
        TORCH_CHECK_ARG(ftheta_coeffs, "ftheta_coeffs", "required for ftheta camera model");
        TORCH_CHECK_ARG(!radial_coeffs, "radial_coeffs", "not allowed for ftheta camera model");
        TORCH_CHECK_ARG(!tangential_coeffs, "tangential_coeffs", "not allowed for ftheta camera model");
        TORCH_CHECK_ARG(!thin_prism_coeffs, "thin_prism_coeffs", "not allowed for ftheta camera model");
        TORCH_CHECK_ARG(!focal_lengths, "focal_lengths", "not allowed for ftheta camera model");

        // Get FThetaCameraDistortionParameters
        TORCH_CHECK(ftheta_coeffs.value(), "ftheta_coeffs intrusive_ptr is null");
        const FThetaCameraDistortionParameters& params = *ftheta_coeffs.value();

        // Convert polynomial arrays to tensors using from_blob
        torch::Tensor pixeldist_to_angle_poly = torch::from_blob(
            const_cast<float*>(params.pixeldist_to_angle_poly.data()),
            {static_cast<int64_t>(params.pixeldist_to_angle_poly.size())},
            torch::kFloat32
        ).clone().to(principal_points.device());

        torch::Tensor angle_to_pixeldist_poly = torch::from_blob(
            const_cast<float*>(params.angle_to_pixeldist_poly.data()),
            {static_cast<int64_t>(params.angle_to_pixeldist_poly.size())},
            torch::kFloat32
        ).clone().to(principal_points.device());

        torch::Tensor linear_cde = torch::from_blob(
            const_cast<float*>(params.linear_cde.data()),
            {static_cast<int64_t>(params.linear_cde.size())},
            torch::kFloat32
        ).clone().to(principal_points.device());

        torch::Tensor max_angle = torch::full({}, params.max_angle, torch::dtype(torch::kFloat32).device(principal_points.device()));

        return c10::make_intrusive<PyFThetaCameraModel>(
            width, height,
            principal_points,
            pixeldist_to_angle_poly,
            angle_to_pixeldist_poly,
            linear_cde,
            params.reference_poly,
            max_angle,
            rs_type
        );
    }
    // TODO: Lidar camera model is not supported through the generic create() method.
    // This needs to be added.
    // For now, use PyRowOffsetStructuredSpinningLidarModel constructor directly with all required parameters.
    else
    {
        throw std::invalid_argument("Unknown camera model: " + camera_model);
    }
}

// ==================== PyBaseCameraModel<CameraModel> Implementations ====================

template <typename CameraModel>
void PyBaseCameraModel<CameraModel>::CudaDeleter::operator()(void* ptr) const
{
    CUDA_CHECK(cudaFree(ptr));
}

template <typename CameraModel>
PyBaseCameraModel<CameraModel>::PyBaseCameraModel(int num_cameras, int width, int height, ShutterType rs_type,
                                                  const torch::Tensor &focal_lengths, const torch::Tensor &principal_points)
    : m_num_cameras(num_cameras)
    , m_width(width)
    , m_height(height)
    , m_rs_type(rs_type)
    , m_focal_lengths(focal_lengths)
    , m_principal_points(principal_points)
{
    CameraModel* d_cameras;
    CUDA_CHECK(cudaMalloc(&d_cameras,num_cameras * sizeof(CameraModel)));
    m_dev_cameras.reset(d_cameras);
}

/**
 * @brief camera_ray_to_image_point kernel (templated on CameraModel)
 *
 * Grid: (rays_per_camera/256, total_cameras, 1)
 * Block: (256, 1, 1)
 * grid.y = camera index, grid.x = ray index
 */
template<typename CameraModel>
__global__ void camera_ray_to_image_point_kernel(
    TensorView<const CameraModel, CAMERA> cameras,
    TensorView<const float, CAMERA, RAY, 3> camera_rays,
    TensorView<float, CAMERA, RAY, 2> image_points,
    TensorView<bool, CAMERA, RAY> valid,
    float margin_factor
)
{
    int camera_idx = blockIdx.y;
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (camera_idx >= cameras.shape(0) || ray_idx >= camera_rays.shape(1))
    {
        return;
    }

    const CameraModel& camera = cameras(camera_idx);

    // Construct input inline
    auto camera_ray = camera_rays(camera_idx, ray_idx);
    auto [img_pt, is_valid] = camera.camera_ray_to_image_point(
        glm::fvec3{camera_ray(0), camera_ray(1), camera_ray(2)},
        margin_factor
    );

    auto image_point = image_points(camera_idx, ray_idx);
    image_point(0) = img_pt.x;
    image_point(1) = img_pt.y;
    valid(camera_idx, ray_idx) = is_valid;
}

/**
 * @brief Template implementation for camera_ray_to_image_point
 *
 * Handles tensor reshaping, kernel launch, and result extraction.
 * Used by all 4 camera model types.
 */
template<typename CameraModel>
std::tuple<torch::Tensor, torch::Tensor> PyBaseCameraModel<CameraModel>::camera_ray_to_image_point(
    const torch::Tensor& camera_ray,
    float margin_factor) const
{
    torch::TensorOptions opt = torch::TensorOptions()
        .dtype(torch::kFloat32).device(camera_ray.device());

    // Create output tensors matching input batch shape
    std::vector<int64_t> output_shape = camera_ray.sizes().vec();
    output_shape.back() = 2;
    torch::Tensor image_points = torch::empty(output_shape, opt);

    output_shape.pop_back();
    torch::Tensor valid_result = torch::empty(output_shape, opt.dtype(torch::kBool));

    int rays_per_camera = camera_ray.size(-2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(camera_ray.device().index());

    // Launch kernel with grid.y=camera, grid.x=ray
    dim3 threads(256);
    dim3 blocks((rays_per_camera + threads.x - 1) / threads.x, m_num_cameras);

    camera_ray_to_image_point_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->const_dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, RAY, 3>(camera_ray, m_num_cameras, "camera_ray"),
        make_tensor_view<float, CAMERA, RAY, 2>(image_points, m_num_cameras, "image_points"),
        make_tensor_view<bool, CAMERA, RAY>(valid_result, m_num_cameras, "valid_result"),
        margin_factor
    );
    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(image_points, valid_result);
}

/**
 * @brief image_point_to_camera_ray kernel
 */
template<typename CameraModel>
__global__ void image_point_to_camera_ray_kernel(
    TensorView<const CameraModel, CAMERA> cameras,
    TensorView<const float, CAMERA, POINT, 2> image_points,
    TensorView<float, CAMERA, POINT, 3> camera_rays,
    TensorView<bool, CAMERA, POINT> valid
)
{
    int camera_idx = blockIdx.y;
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (camera_idx >= cameras.shape(0) || point_idx >= image_points.shape(1))
    {
        return;
    }

    const CameraModel& camera = cameras(camera_idx);

    auto cam_ray_result = camera.image_point_to_camera_ray(
        glm::fvec2{image_points(camera_idx, point_idx, 0),
                   image_points(camera_idx, point_idx, 1)}
    );

    auto camera_ray = camera_rays(camera_idx, point_idx);
    camera_ray(0) = cam_ray_result.ray_dir.x;
    camera_ray(1) = cam_ray_result.ray_dir.y;
    camera_ray(2) = cam_ray_result.ray_dir.z;

    valid(camera_idx, point_idx) = cam_ray_result.valid_flag;
}

template<typename CameraModel>
std::tuple<torch::Tensor, torch::Tensor> PyBaseCameraModel<CameraModel>::image_point_to_camera_ray(
    const torch::Tensor& image_points) const
{
    std::vector<int64_t> output_shape = image_points.sizes().vec();
    output_shape.back() = 3;

    torch::TensorOptions opt = torch::TensorOptions()
        .dtype(torch::kFloat32).device(image_points.device());
    torch::Tensor camera_rays = torch::empty(output_shape, opt);

    output_shape.pop_back();
    torch::Tensor valid_result = torch::empty(output_shape, opt.dtype(torch::kBool));

    int points_per_camera = image_points.size(-2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(image_points.device().index());

    dim3 threads(256);
    dim3 blocks((points_per_camera + threads.x - 1) / threads.x, m_num_cameras);

    image_point_to_camera_ray_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->const_dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, POINT, 2>(image_points, m_num_cameras, "image_points"),
        make_tensor_view<float, CAMERA, POINT, 3>(camera_rays, m_num_cameras, "camera_rays"),
        make_tensor_view<bool, CAMERA, POINT>(valid_result, m_num_cameras, "valid_result")
    );
    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(camera_rays, valid_result);
}

/**
 * @brief shutter_relative_frame_time kernel
 */
template<typename CameraModel>
__global__ void shutter_relative_frame_time_kernel(
    TensorView<const CameraModel, CAMERA> cameras,
    TensorView<const float, CAMERA, POINT, 2> image_points,
    TensorView<float, CAMERA, POINT> relative_times
)
{
    int camera_idx = blockIdx.y;
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (camera_idx >= cameras.shape(0) || point_idx >= image_points.shape(1))
    {
        return;
    }

    const CameraModel& camera = cameras(camera_idx);

    auto img_pt = image_points(camera_idx, point_idx);
    float relative_time = camera.shutter_relative_frame_time(glm::fvec2{img_pt(0), img_pt(1)});

    relative_times(camera_idx, point_idx) = relative_time;
}


template<typename CameraModel>
torch::Tensor PyBaseCameraModel<CameraModel>::shutter_relative_frame_time(
    const torch::Tensor& image_points) const
{
    std::vector<int64_t> output_shape = image_points.sizes().vec();
    output_shape.pop_back();

    torch::TensorOptions opt = torch::TensorOptions()
        .dtype(torch::kFloat32).device(image_points.device());
    torch::Tensor relative_times = torch::empty(output_shape, opt);

    int points_per_camera = image_points.size(-2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(image_points.device().index());

    dim3 threads(256);
    dim3 blocks((points_per_camera + threads.x - 1) / threads.x, m_num_cameras);

    shutter_relative_frame_time_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->const_dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, POINT, 2>(image_points, m_num_cameras, "image_points"),
        make_tensor_view<float, CAMERA, POINT>(relative_times, m_num_cameras, "relative_times")
    );
    CUDA_CHECK(cudaGetLastError());

    return relative_times;
}

/**
 * @brief image_point_to_world_ray_shutter_pose kernel
 */
template<typename CameraModel>
__global__ void image_point_to_world_ray_shutter_pose_kernel(
    TensorView<const CameraModel, CAMERA> cameras,
    TensorView<const float, CAMERA, POINT, 2> image_points,
    TensorView<const float, CAMERA, 7> poses_start,
    TensorView<const float, CAMERA, 7> poses_end,
    TensorView<float, CAMERA, POINT, 3> world_rays_org,
    TensorView<float, CAMERA, POINT, 3> world_rays_dir,
    TensorView<bool, CAMERA, POINT> valid
)
{
    int camera_idx = blockIdx.y;
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (camera_idx >= cameras.shape(0) || point_idx >= image_points.shape(1))
    {
        return;
    }

    const CameraModel& camera = cameras(camera_idx);

    auto pose_s = poses_start(camera_idx);
    auto pose_e = poses_end(camera_idx);

    // Create RollingShutterParameters with individual member assignments
    // Note: Input pose format is [tx, ty, tz, qw, qx, qy, qz]
    // GLM quaternion constructor (without GLM_FORCE_QUAT_DATA_XYZW) is qua(w, x, y, z)
    RollingShutterParameters rs_params;
    rs_params.t_start = glm::fvec3{pose_s(0), pose_s(1), pose_s(2)};
    rs_params.q_start = glm::fquat{pose_s(3), pose_s(4), pose_s(5), pose_s(6)};  // {qw, qx, qy, qz} -> GLM qua(w, x, y, z)
    rs_params.t_end = glm::fvec3{pose_e(0), pose_e(1), pose_e(2)};
    rs_params.q_end = glm::fquat{pose_e(3), pose_e(4), pose_e(5), pose_e(6)};    // {qw, qx, qy, qz} -> GLM qua(w, x, y, z)

    auto img_pt = image_points(camera_idx, point_idx);
    auto world_ray = camera.image_point_to_world_ray_shutter_pose(
        glm::fvec2{img_pt(0), img_pt(1)},
        rs_params
    );

    auto wr_org = world_rays_org(camera_idx, point_idx);
    wr_org(0) = world_ray.ray_org.x;
    wr_org(1) = world_ray.ray_org.y;
    wr_org(2) = world_ray.ray_org.z;

    auto wr_dir = world_rays_dir(camera_idx, point_idx);
    wr_dir(0) = world_ray.ray_dir.x;
    wr_dir(1) = world_ray.ray_dir.y;
    wr_dir(2) = world_ray.ray_dir.z;

    valid(camera_idx, point_idx) = world_ray.valid_flag;
}

template<typename CameraModel>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PyBaseCameraModel<CameraModel>::image_point_to_world_ray_shutter_pose(
    const torch::Tensor& image_points,
    const torch::Tensor& pose_start, const torch::Tensor& pose_end) const
{
    std::vector<int64_t> output_shape = image_points.sizes().vec();
    output_shape.back() = 3;

    torch::TensorOptions opt = torch::TensorOptions()
        .dtype(torch::kFloat32).device(image_points.device());
    torch::Tensor world_rays_org = torch::empty(output_shape, opt);
    torch::Tensor world_rays_dir = torch::empty(output_shape, opt);

    output_shape.pop_back();
    torch::Tensor valid_result = torch::empty(output_shape, opt.dtype(torch::kBool));

    int points_per_camera = image_points.size(-2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(image_points.device().index());

    dim3 threads(256);
    dim3 blocks((points_per_camera + threads.x - 1) / threads.x, m_num_cameras);

    image_point_to_world_ray_shutter_pose_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->const_dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, POINT, 2>(image_points, m_num_cameras, "image_points"),
        make_tensor_view<const float, CAMERA, 7>(pose_start, m_num_cameras, "pose_start"),
        make_tensor_view<const float, CAMERA, 7>(pose_end, m_num_cameras, "pose_end"),
        make_tensor_view<float, CAMERA, POINT, 3>(world_rays_org, m_num_cameras, "world_rays_org"),
        make_tensor_view<float, CAMERA, POINT, 3>(world_rays_dir, m_num_cameras, "world_rays_dir"),
        make_tensor_view<bool, CAMERA, POINT>(valid_result, m_num_cameras, "valid_result")
    );
    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(world_rays_org, world_rays_dir, valid_result);
}

/**
 * @brief world_point_to_image_point_shutter_pose kernel
 */
template<typename CameraModel>
__global__ void world_point_to_image_point_shutter_pose_kernel(
    TensorView<const CameraModel, CAMERA> cameras,
    TensorView<const float, CAMERA, POINT, 3> world_points,
    TensorView<const float, CAMERA, 7> poses_start,
    TensorView<const float, CAMERA, 7> poses_end,
    TensorView<float, CAMERA, POINT, 2> image_points,
    TensorView<bool, CAMERA, POINT> valid,
    float margin_factor
)
{
    int camera_idx = blockIdx.y;
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (camera_idx >= cameras.shape(0) || point_idx >= world_points.shape(1))
    {
        return;
    }

    const CameraModel& camera = cameras(camera_idx);

    auto pose_s = poses_start(camera_idx);
    auto pose_e = poses_end(camera_idx);

    // Create RollingShutterParameters with individual member assignments
    // Note: Input pose format is [tx, ty, tz, qw, qx, qy, qz]
    // GLM quaternion constructor (without GLM_FORCE_QUAT_DATA_XYZW) is qua(w, x, y, z)
    RollingShutterParameters rs_params;
    rs_params.t_start = glm::fvec3{pose_s(0), pose_s(1), pose_s(2)};
    rs_params.q_start = glm::fquat{pose_s(3), pose_s(4), pose_s(5), pose_s(6)};  // {qw, qx, qy, qz} -> GLM qua(w, x, y, z)
    rs_params.t_end = glm::fvec3{pose_e(0), pose_e(1), pose_e(2)};
    rs_params.q_end = glm::fquat{pose_e(3), pose_e(4), pose_e(5), pose_e(6)};    // {qw, qx, qy, qz} -> GLM qua(w, x, y, z)

    auto wp = world_points(camera_idx, point_idx);
    auto [img_pt, is_valid] = camera.world_point_to_image_point_shutter_pose(
        glm::fvec3{wp(0), wp(1), wp(2)},
        rs_params,
        margin_factor
    );

    auto ip = image_points(camera_idx, point_idx);
    ip(0) = img_pt.x;
    ip(1) = img_pt.y;
    valid(camera_idx, point_idx) = is_valid;
}

template<typename CameraModel>
std::tuple<torch::Tensor, torch::Tensor> PyBaseCameraModel<CameraModel>::world_point_to_image_point_shutter_pose(
    const torch::Tensor& world_points,
    const torch::Tensor& pose_start, const torch::Tensor& pose_end,
    float margin_factor) const
{
    torch::TensorOptions opt = torch::TensorOptions()
        .dtype(torch::kFloat32).device(world_points.device());

    std::vector<int64_t> output_shape = world_points.sizes().vec();
    output_shape.back() = 2;
    torch::Tensor image_points = torch::empty(output_shape, opt);

    output_shape.pop_back();
    torch::Tensor valid_result = torch::empty(output_shape, opt.dtype(torch::kBool));

    int points_per_camera = world_points.size(-2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(world_points.device().index());

    dim3 threads(256);
    dim3 blocks((points_per_camera + threads.x - 1) / threads.x, m_num_cameras);

    world_point_to_image_point_shutter_pose_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->const_dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, POINT, 3>(world_points, m_num_cameras, "world_points"),
        make_tensor_view<const float, CAMERA, 7>(pose_start, m_num_cameras, "pose_start"),
        make_tensor_view<const float, CAMERA, 7>(pose_end, m_num_cameras, "pose_end"),
        make_tensor_view<float, CAMERA, POINT, 2>(image_points, m_num_cameras, "image_points"),
        make_tensor_view<bool, CAMERA, POINT>(valid_result, m_num_cameras, "valid_result"),
        margin_factor
    );
    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(image_points, valid_result);
}

// ========================== PyPerfectPinholeCameraModel Implementation ==========================

/**
 * @brief Constructor kernel for perfect pinhole cameras
 *
 * Grid: (num_cameras/256, 1, 1)
 * Block: (256, 1, 1)
 * One thread per camera
 */
__global__ void construct_perfect_pinhole_cameras_kernel(
    TensorView<PerfectPinholeCameraModel, CAMERA> cameras,
    TensorView<const float, CAMERA, 2> focal_lengths,
    TensorView<const float, CAMERA, 2> principal_points,
    int width, int height, ShutterType rs_type
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cameras.shape(0))
    {
        return;
    }

    PerfectPinholeCameraModel::Parameters params;
    params.resolution = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    // Use partial indexing to get per-camera parameters
    auto focal_length = focal_lengths(idx);
    params.focal_length = {focal_length(0), focal_length(1)};

    auto principal_point = principal_points(idx);
    params.principal_point = {principal_point(0), principal_point(1)};

    params.shutter_type = rs_type;

    // Construct camera in-place
    new (&cameras(idx)) PerfectPinholeCameraModel(params);
}


PyPerfectPinholeCameraModel::PyPerfectPinholeCameraModel(
    int width, int height,
    const torch::Tensor& focal_lengths, const torch::Tensor& principal_points,
    ShutterType rs_type
) : PyBaseCameraModel(normalize_shape<CAMERA, 2>(focal_lengths).size(0), width, height, rs_type, focal_lengths, principal_points)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(focal_lengths.device().index());

    int threads = 256;
    int blocks = (m_num_cameras + threads - 1) / threads;

    construct_perfect_pinhole_cameras_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, 2>(focal_lengths, m_num_cameras, "focal_lengths"),
        make_tensor_view<const float, CAMERA, 2>(principal_points, m_num_cameras, "principal_points"),
        m_width, m_height, m_rs_type
    );
    CUDA_CHECK(cudaGetLastError());
}

// ====================== PyOpenCVPinholeCameraModel Implementation ========================

/**
 * @brief Constructor kernel for OpenCV pinhole cameras with distortion
 */
__global__ void construct_opencv_pinhole_cameras_kernel(
    TensorView<OpenCVPinholeCameraModel<>, CAMERA> cameras,
    TensorView<const float, CAMERA, 2> focal_lengths,
    TensorView<const float, CAMERA, 2> principal_points,
    TensorView<const float, CAMERA, COEFF> radial_coeffs,
    TensorView<const float, CAMERA, COEFF> tangential_coeffs,
    TensorView<const float, CAMERA, COEFF> thin_prism_coeffs,
    int width, int height, ShutterType rs_type
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cameras.shape(0))
    {
        return;
    }

    OpenCVPinholeCameraModel<>::Parameters params;
    params.resolution = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    auto focal_length = focal_lengths(idx);
    params.focal_length = {focal_length(0), focal_length(1)};

    auto principal_point = principal_points(idx);
    params.principal_point = {principal_point(0), principal_point(1)};

    params.shutter_type = rs_type;

    if (radial_coeffs)
    {
        auto radial_coeff = radial_coeffs(idx);
        int n_radial = radial_coeff.shape(0);
        for (int i = 0; i < n_radial; ++i)
        {
            params.radial_coeffs[i] = radial_coeff(i);
        }
    }

    if (tangential_coeffs)
    {
        auto tangential_coeff = tangential_coeffs(idx);
        int n_tang = tangential_coeff.shape(0);
        for (int i = 0; i < n_tang; ++i)
        {
            params.tangential_coeffs[i] = tangential_coeff(i);
        }
    }

    if (thin_prism_coeffs)
    {
        auto thin_prism_coeff = thin_prism_coeffs(idx);
        int n_tp = thin_prism_coeff.shape(0);
        for (int i = 0; i < n_tp; ++i)
        {
            params.thin_prism_coeffs[i] = thin_prism_coeff(i);
        }
    }

    new (&cameras(idx)) OpenCVPinholeCameraModel<>(params);
}

PyOpenCVPinholeCameraModel::PyOpenCVPinholeCameraModel(
    int width, int height,
    const torch::Tensor& focal_lengths, const torch::Tensor& principal_points,
    const std::optional<torch::Tensor>& radial_coeffs,
    const std::optional<torch::Tensor>& tangential_coeffs,
    const std::optional<torch::Tensor>& thin_prism_coeffs,
    ShutterType rs_type
) : PyBaseCameraModel(normalize_shape<CAMERA, 2>(focal_lengths).size(0), width, height, rs_type, focal_lengths, principal_points)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(focal_lengths.device().index());

    int threads = 256;
    int blocks = (m_num_cameras + threads - 1) / threads;

    construct_opencv_pinhole_cameras_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, 2>(focal_lengths, m_num_cameras, "focal_lengths"),
        make_tensor_view<const float, CAMERA, 2>(principal_points, m_num_cameras, "principal_points"),
        make_tensor_view<const float, CAMERA, COEFF>(radial_coeffs, m_num_cameras, "radial_coeffs"),
        make_tensor_view<const float, CAMERA, COEFF>(tangential_coeffs, m_num_cameras, "tangential_coeffs"),
        make_tensor_view<const float, CAMERA, COEFF>(thin_prism_coeffs, m_num_cameras, "thin_prism_coeffs"),
        m_width, m_height, m_rs_type
    );
    CUDA_CHECK(cudaGetLastError());
}

// ================ PyOpenCVFisheye Implementation ==============================

/**
 * @brief Constructor kernel for OpenCV fisheye cameras
 */
__global__ void construct_opencv_fisheye_cameras_kernel(
    TensorView<OpenCVFisheyeCameraModel<>, CAMERA> cameras,
    TensorView<const float, CAMERA, 2> focal_lengths,
    TensorView<const float, CAMERA, 2> principal_points,
    TensorView<const float, CAMERA, COEFF> radial_coeffs,
    int width, int height, ShutterType rs_type
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cameras.shape(0))
    {
        return;
    }

    OpenCVFisheyeCameraModel<>::Parameters params;
    params.resolution = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    auto focal_length = focal_lengths(idx);
    params.focal_length = {focal_length(0), focal_length(1)};

    auto principal_point = principal_points(idx);
    params.principal_point = {principal_point(0), principal_point(1)};

    params.shutter_type = rs_type;

    if (radial_coeffs)
    {
        auto radial_coeff = radial_coeffs(idx);
        int n_radial = radial_coeff.shape(0);
        for (int i = 0; i < n_radial; ++i)
        {
            params.radial_coeffs[i] = radial_coeff(i);
        }
    }

    new (&cameras(idx)) OpenCVFisheyeCameraModel<>(params);
}

PyOpenCVFisheyeCameraModel::PyOpenCVFisheyeCameraModel(
    int width, int height,
    const torch::Tensor& focal_lengths, const torch::Tensor& principal_points,
    const std::optional<torch::Tensor>& radial_coeffs,
    ShutterType rs_type
) : PyBaseCameraModel(normalize_shape<CAMERA, 2>(focal_lengths).size(0), width, height, rs_type, focal_lengths, principal_points)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(focal_lengths.device().index());

    int threads = 256;
    int blocks = (m_num_cameras + threads - 1) / threads;

    construct_opencv_fisheye_cameras_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, 2>(focal_lengths, m_num_cameras, "focal_lengths"),
        make_tensor_view<const float, CAMERA, 2>(principal_points, m_num_cameras, "principal_points"),
        make_tensor_view<const float, CAMERA, COEFF>(radial_coeffs, m_num_cameras, "radial_coeffs"),
        m_width, m_height, m_rs_type
    );
    CUDA_CHECK(cudaGetLastError());
}

// ================================== FTheta Implementation ==================================

/**
 * @brief Constructor kernel for F-Theta cameras
 */
__global__ void construct_ftheta_cameras_kernel(
    TensorView<FThetaCameraModel<>, CAMERA> cameras,
    TensorView<const float, CAMERA, 2> principal_points,
    TensorView<const float, CAMERA, 6> pixeldist_to_angle_poly,
    TensorView<const float, CAMERA, 6> angle_to_pixeldist_poly,
    TensorView<const float, CAMERA, 3> linear_cde,
    FThetaCameraDistortionParameters::PolynomialType reference_poly,
    TensorView<const float, CAMERA> max_angle,
    int width, int height,
    ShutterType rs_type
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cameras.shape(0))
    {
        return;
    }

    FThetaCameraModel<>::Parameters params;
    params.resolution = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    auto principal_point = principal_points(idx);
    params.principal_point = {principal_point(0), principal_point(1)};

    params.shutter_type = rs_type;

    auto p2a = pixeldist_to_angle_poly(idx);
    auto a2p = angle_to_pixeldist_poly(idx);
    auto lcd = linear_cde(idx);

    for (int i = 0; i < 6; ++i)
    {
        params.dist.pixeldist_to_angle_poly[i] = p2a(i);
        params.dist.angle_to_pixeldist_poly[i] = a2p(i);
    }

    params.dist.linear_cde[0] = lcd(0);
    params.dist.linear_cde[1] = lcd(1);
    params.dist.linear_cde[2] = lcd(2);

    params.dist.reference_poly = reference_poly;
    params.dist.max_angle = max_angle(idx);

    new (&cameras(idx)) FThetaCameraModel<>(params);
}

PyFThetaCameraModel::PyFThetaCameraModel(
    int width, int height,
    const torch::Tensor& principal_points,
    const torch::Tensor& pixeldist_to_angle_poly,
    const torch::Tensor& angle_to_pixeldist_poly,
    const torch::Tensor& linear_cde,
    FThetaCameraDistortionParameters::PolynomialType reference_poly,
    const torch::Tensor &max_angle,
    ShutterType rs_type
) : PyBaseCameraModel(normalize_shape<CAMERA, 2>(principal_points).size(0), width, height, rs_type,
                      // The linear component is a decent approximation of the focal length.
                      angle_to_pixeldist_poly.index_select(-1, torch::tensor({1, 1}, torch::kLong).to(angle_to_pixeldist_poly.device())),
                      principal_points)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(principal_points.device().index());

    int threads = 256;
    int blocks = (m_num_cameras + threads - 1) / threads;

    construct_ftheta_cameras_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->dev_cameras(), {m_num_cameras}, {1}, "cameras"),
        make_tensor_view<const float, CAMERA, 2>(principal_points, m_num_cameras, "principal_points"),
        make_tensor_view<const float, CAMERA, 6>(pixeldist_to_angle_poly, m_num_cameras, "pixeldist_to_angle_poly"),
        make_tensor_view<const float, CAMERA, 6>(angle_to_pixeldist_poly, m_num_cameras, "angle_to_pixeldist_poly"),
        make_tensor_view<const float, CAMERA, 3>(linear_cde, m_num_cameras, "linear_cde"),
        reference_poly,
        make_tensor_view<const float, CAMERA>(max_angle, m_num_cameras, "max_angle"),
        m_width, m_height, m_rs_type
    );

    CUDA_CHECK(cudaGetLastError());
}

// ================================== Lidar Implementation ==================================

__global__ void construct_lidar_cameras_kernel(
    TensorView<RowOffsetStructuredSpinningLidarModel, CAMERA> cameras, RowOffsetStructuredSpinningLidarModelParametersExtDevice lidar_params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cameras.shape(0))
    {
        return;
    }

    // Create the camera with initialized parameters
    new (&cameras(idx)) RowOffsetStructuredSpinningLidarModel(lidar_params);
}

PyRowOffsetStructuredSpinningLidarModel::PyRowOffsetStructuredSpinningLidarModel(RowOffsetStructuredSpinningLidarModelParametersExt params)
    // TODO: We're mapping n_rows->width and n_columns->height because the
    // current pipeline uses image_point = [x, y] = [row, column] = [elevation,
    // azimuth]. This should be reverted once the pipeline switches to the
    // standard [x, y] = [column, row] = [azimuth, elevation].
    : PyBaseCameraModel(1, params.n_rows(), params.n_columns(),
                      // Spinning lidars always have rolling shutter; the actual per-sigma-point
                      // timing is handled by shutter_relative_frame_time() in the lidar model.
                      // Any non-GLOBAL value activates the RS iteration loop.
                      ShutterType::ROLLING_LEFT_TO_RIGHT, torch::zeros({0}), torch::zeros({0}))
    , m_params(std::move(params))
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(m_params.row_elevations_rad.device().index());

    int threads = 256;
    int blocks = (m_num_cameras + threads - 1) / threads;

    construct_lidar_cameras_kernel<<<blocks, threads, 0, stream>>>(
        make_tensor_view<CAMERA>(this->dev_cameras(), {m_num_cameras}, {1}, "cameras"), m_params);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace gsplat

#endif
