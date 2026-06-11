/*
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

// Kernel parameter packs for external distortion. The coefficient tensor layout
// is shared with the CUDA helpers, so keep packing changes synchronized.

#pragma once

#include <cstdint>

// Empty by design: no-external keeps the templated launch path uniform.
struct NoExternalDistortion_KernelParameters
{
};

// Kernel-visible view of BivariateWindshieldDistortion. The coefficient buffer
// must not alias other tensors in the same launch.
struct BivariateWindshieldDistortion_KernelParameters
{
    // Packed as [h(6), v(15), h_inv(6), v_inv(15)].
    const float *__restrict__ distortion_coeffs;
    // ReferencePolynomial value: 0 uses h/v as forward and h_inv/v_inv as
    // inverse; 1 swaps those semantic roles.
    uint32_t reference_polynomial;
    // Degrees bound active triangular terms; kernels assume higher coefficients
    // are zero. Horizontal must be <= 2; vertical must be <= 4.
    uint32_t h_poly_degree;
    uint32_t v_poly_degree;
};

struct NoExternalDistortionPolicyTag
{
};

struct BivariateWindshieldPolicyTag
{
};

enum class DistortionSensor
{
    FTheta,
    OpenCVFisheye,
};

enum class DistortionOpFamily
{
    CameraRaysToImagePoints,
    ImagePointsToCameraRays,
    ProjectWorldPointsMeanPose,
    ProjectWorldPointsShutterPose,
    ImagePointsToWorldRaysStaticPose,
    ImagePointsToWorldRaysShutterPose,
};

enum class DistortionDirection
{
    Forward,
    Backward,
};

template<DistortionSensor Sensor, DistortionOpFamily Op, DistortionDirection Direction, typename PolicyTag>
struct DistortionScratchTraits;

// Per wrapper, records scratch stride, optional inverse-stash offset, and
// whether the kernel should load the undistort polynomial. A stash offset of -1
// means the op never stores an inverse-distortion primal.
#define GSPLAT_SENSOR_DISTORTION_TRAITS(sensor, op, direction, policy, stride, stash, undistort) \
    template<>                                                                                   \
    struct DistortionScratchTraits<                                                              \
        DistortionSensor::sensor,                                                                \
        DistortionOpFamily::op,                                                                  \
        DistortionDirection::direction,                                                          \
        policy                                                                                   \
    >                                                                                            \
    {                                                                                            \
        static constexpr int kScratchStride      = stride;                                       \
        static constexpr int kInverseStashOffset = stash;                                        \
        static constexpr bool kIsUndistort       = undistort;                                    \
    }

GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, CameraRaysToImagePoints, Forward, NoExternalDistortionPolicyTag, 8, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, CameraRaysToImagePoints, Backward, NoExternalDistortionPolicyTag, 8, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, CameraRaysToImagePoints, Forward, BivariateWindshieldPolicyTag, 8, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, CameraRaysToImagePoints, Backward, BivariateWindshieldPolicyTag, 8, -1, false
);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToCameraRays, Forward, NoExternalDistortionPolicyTag, 8, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToCameraRays, Backward, NoExternalDistortionPolicyTag, 8, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToCameraRays, Forward, BivariateWindshieldPolicyTag, 12, 8, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToCameraRays, Backward, BivariateWindshieldPolicyTag, 12, 8, true
);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ProjectWorldPointsMeanPose, Forward, NoExternalDistortionPolicyTag, 14, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ProjectWorldPointsMeanPose, Backward, NoExternalDistortionPolicyTag, 14, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ProjectWorldPointsMeanPose, Forward, BivariateWindshieldPolicyTag, 14, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ProjectWorldPointsMeanPose, Backward, BivariateWindshieldPolicyTag, 14, -1, false
);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ProjectWorldPointsShutterPose, Forward, NoExternalDistortionPolicyTag, 16, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ProjectWorldPointsShutterPose, Backward, NoExternalDistortionPolicyTag, 16, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ProjectWorldPointsShutterPose, Forward, BivariateWindshieldPolicyTag, 16, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ProjectWorldPointsShutterPose, Backward, BivariateWindshieldPolicyTag, 16, -1, false
);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToWorldRaysStaticPose, Forward, NoExternalDistortionPolicyTag, 8, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToWorldRaysStaticPose, Backward, NoExternalDistortionPolicyTag, 8, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToWorldRaysStaticPose, Forward, BivariateWindshieldPolicyTag, 12, 8, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToWorldRaysStaticPose, Backward, BivariateWindshieldPolicyTag, 12, 8, true
);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToWorldRaysShutterPose, Forward, NoExternalDistortionPolicyTag, 12, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToWorldRaysShutterPose, Backward, NoExternalDistortionPolicyTag, 12, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToWorldRaysShutterPose, Forward, BivariateWindshieldPolicyTag, 16, 12, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    OpenCVFisheye, ImagePointsToWorldRaysShutterPose, Backward, BivariateWindshieldPolicyTag, 16, 12, true
);

GSPLAT_SENSOR_DISTORTION_TRAITS(FTheta, CameraRaysToImagePoints, Forward, NoExternalDistortionPolicyTag, 8, -1, false);
GSPLAT_SENSOR_DISTORTION_TRAITS(FTheta, CameraRaysToImagePoints, Backward, NoExternalDistortionPolicyTag, 8, -1, false);
GSPLAT_SENSOR_DISTORTION_TRAITS(FTheta, CameraRaysToImagePoints, Forward, BivariateWindshieldPolicyTag, 8, -1, false);
GSPLAT_SENSOR_DISTORTION_TRAITS(FTheta, CameraRaysToImagePoints, Backward, BivariateWindshieldPolicyTag, 8, -1, false);

GSPLAT_SENSOR_DISTORTION_TRAITS(FTheta, ImagePointsToCameraRays, Forward, NoExternalDistortionPolicyTag, 8, -1, true);
GSPLAT_SENSOR_DISTORTION_TRAITS(FTheta, ImagePointsToCameraRays, Backward, NoExternalDistortionPolicyTag, 8, -1, true);
GSPLAT_SENSOR_DISTORTION_TRAITS(FTheta, ImagePointsToCameraRays, Forward, BivariateWindshieldPolicyTag, 12, 8, true);
GSPLAT_SENSOR_DISTORTION_TRAITS(FTheta, ImagePointsToCameraRays, Backward, BivariateWindshieldPolicyTag, 12, 8, true);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ProjectWorldPointsMeanPose, Forward, NoExternalDistortionPolicyTag, 10, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ProjectWorldPointsMeanPose, Backward, NoExternalDistortionPolicyTag, 10, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ProjectWorldPointsMeanPose, Forward, BivariateWindshieldPolicyTag, 10, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ProjectWorldPointsMeanPose, Backward, BivariateWindshieldPolicyTag, 10, -1, false
);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ProjectWorldPointsShutterPose, Forward, NoExternalDistortionPolicyTag, 11, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ProjectWorldPointsShutterPose, Backward, NoExternalDistortionPolicyTag, 11, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ProjectWorldPointsShutterPose, Forward, BivariateWindshieldPolicyTag, 11, -1, false
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ProjectWorldPointsShutterPose, Backward, BivariateWindshieldPolicyTag, 11, -1, false
);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ImagePointsToWorldRaysStaticPose, Forward, NoExternalDistortionPolicyTag, 8, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ImagePointsToWorldRaysStaticPose, Backward, NoExternalDistortionPolicyTag, 8, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ImagePointsToWorldRaysStaticPose, Forward, BivariateWindshieldPolicyTag, 12, 8, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ImagePointsToWorldRaysStaticPose, Backward, BivariateWindshieldPolicyTag, 12, 8, true
);

GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ImagePointsToWorldRaysShutterPose, Forward, NoExternalDistortionPolicyTag, 9, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ImagePointsToWorldRaysShutterPose, Backward, NoExternalDistortionPolicyTag, 9, -1, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ImagePointsToWorldRaysShutterPose, Forward, BivariateWindshieldPolicyTag, 12, 8, true
);
GSPLAT_SENSOR_DISTORTION_TRAITS(
    FTheta, ImagePointsToWorldRaysShutterPose, Backward, BivariateWindshieldPolicyTag, 12, 8, true
);

#undef GSPLAT_SENSOR_DISTORTION_TRAITS
