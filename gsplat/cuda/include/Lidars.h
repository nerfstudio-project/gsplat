#pragma once

#include <torch/custom_class.h>

namespace gsplat {

// Lidar Camera Model Support

// Spinning direction enum
enum class SpinningDirection {
    CLOCKWISE = 0,
    COUNTER_CLOCKWISE = 1
};

struct FOV : public torch::CustomClassHolder
{
    FOV(float start = 0.f, float span = 0.f) : start(start), span(span) {}

    float start;
    float span;
};

// Plain FOV for device-side structs (no CustomClassHolder overhead)
struct FOVDevice
{
    FOVDevice() = default;

    FOVDevice(const c10::intrusive_ptr<FOV> &fov)
        : start{fov->start}
        , span{fov->span}
    {
    }

    float start;
    float span;
};

} // namespace gsplat

