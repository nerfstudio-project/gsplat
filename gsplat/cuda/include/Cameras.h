#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <variant>
#include <tuple>

// Include PyTorch tensors only for host code (not device code)
#ifndef __CUDA_ARCH__
#include <torch/torch.h>
#endif

// ---------------------------------------------------------------------------------------------

// Camera-specific types (camera model parameters and returns)

enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};

// ---------------------------------------------------------------------------------------------

// Gaussian-specific types
struct UnscentedTransformParameters {
    // See Gustafsson and Hendeby 2012 for sigma point parameterization - this
    // default parameter choice is based on
    //
    // - "The unscented Kalman filter for nonlinear estimation" - Wan and van
    // der Merwe 2000
    float alpha = 0.1;
    float beta = 2.f;
    float kappa = 0.f;

    // Parameters controlling validity of the unscented transform results
    float in_image_margin_factor =
        0.1f; // 10% out of bounds margin is acceptable for "valid" projection
              // state
    bool require_all_sigma_points_valid =
        false; // true: all sigma points must be valid to mark a projection as
               // "valid" false: a single valid sigma point is sufficient to
               // mark a projection as "valid"
};

// FTheta Camera Support
struct FThetaCameraDistortionParameters {
    static constexpr size_t PolynomialDegree = 6;
    enum class PolynomialType {
        PIXELDIST_TO_ANGLE,
        ANGLE_TO_PIXELDIST,
    };
    PolynomialType reference_poly;
    std::array<float, PolynomialDegree> pixeldist_to_angle_poly; // backward polynomial
    std::array<float, PolynomialDegree> angle_to_pixeldist_poly; // forward polynomial
    float max_angle;
    std::array<float, 3> linear_cde;
};

// ---------------------------------------------------------------------------------------------

namespace gsplat {

// Lidar Camera Model Support

// Spinning direction enum
enum class SpinningDirection {
    CLOCKWISE = 0,
    COUNTER_CLOCKWISE = 1
};

// Python-facing Lidar sensor parameters struct (with tensors for Python bindings)
// Only available in host code
#ifndef __CUDA_ARCH__
struct LidarSensorParameters {
    std::tuple<float, float> fov_elevation_range;
    std::tuple<float, float> fov_azimuth_range;
    SpinningDirection spin_direction;
    torch::Tensor row_elevations;
    torch::Tensor column_azimuths;
    torch::Tensor row_azimuth_offsets;
    torch::Tensor angle_to_column_map;
};
#endif

// Lidar camera parameters struct (device-side, with raw pointers)
// All parameters are provided via API as tensors
//
// NOTE: This struct supports both VREN preprocessing parameters (row_elevations, row_azimuth_offsets,
// column_azimuths) and precomputed acceleration structures (angle_to_column_map).
// - VREN-style: Provide row_elevations, row_azimuth_offsets, column_azimuths (angle_to_column_map can be computed)
// - NREND-style: Provide precomputed angle_to_column_map directly
// All tensor parameters are optional (can be null pointers).
struct LidarCameraParameters {
    // Sensor dimensions
    int n_rows;
    int n_columns;

    // FOV ranges as min/max pairs [min, max] in radians
    std::tuple<float, float> fov_elevation_range;  // [min_elevation, max_elevation]
    std::tuple<float, float> fov_azimuth_range;    // [min_azimuth, max_azimuth]

    // Spinning direction of the lidar
    SpinningDirection spin_direction;

    // Optional: VREN-style lookup tables (for future preprocessing support)
    // These can be used to compute angle_to_column_map if not provided directly
    const float* row_elevations;        // [n_rows] - elevation angle per row (optional, can be nullptr)
    const float* column_azimuths;       // [n_columns] - azimuth angle per column (optional, can be nullptr)
    const float* row_azimuth_offsets;   // [n_rows] - azimuth offset per row (optional, can be nullptr)

    // Optional: Precomputed acceleration structure for rolling shutter timing
    // Maps (elevation, azimuth) angles to actual column indices (capture time)
    const int* angle_to_column_map;     // [n_pts_vert × n_pts_horiz] int32 (optional, can be nullptr)
                                        // where n_pts_vert = n_rows * angle_to_column_map_resolution_factor
                                        // and n_pts_horiz = n_columns * angle_to_column_map_resolution_factor
    int angle_to_column_map_resolution_factor;  // Upsampling factor for lookup grid
    std::array<float, 2> map_resolution;        // Grid cell size in radians [azimuth, elevation]

    // Angle to pixel scaling factor (NREND assumes fixed LUT resolution of 1024 for both)
    static constexpr float ANGLE_TO_PIXEL_SCALING_FACTOR = 1024.f;
};

} // namespace gsplat
