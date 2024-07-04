#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// Shim types to extract vector types from scalar types
// e.g. using vec3f = Float3<float>::type; yields float3
template <typename T>
struct Float3 {};
template <>
struct Float3<float> {
    typedef float3 type;
};
template <>
struct Float3<double> {
    typedef double3 type;
};

template <typename T>
struct Float4 {};
template <>
struct Float4<float> {
    typedef float4 type;
};
template <>
struct Float4<double> {
    typedef double4 type;
};

template <typename T>
struct Float2 {};
template <>
struct Float2<float> {
    typedef float2 type;
};
template <>
struct Float2<double> {
    typedef double2 type;
};



template <typename T>
using vec2 = glm::vec<2, T>;

template <typename T>
using vec3 = glm::vec<3, T>;

template <typename T>
using vec4 = glm::vec<4, T>;

template <typename T>
using mat2 = glm::mat<2, 2, T>;

template <typename T>
using mat3 = glm::mat<3, 3, T>;

template <typename T>
using mat4 = glm::mat<4, 4, T>;

template <typename T>
using mat3x2 = glm::mat<3, 2, T>;

