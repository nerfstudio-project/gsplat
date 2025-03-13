#ifndef GSPLAT_CUDA_TYPES_CUH
#define GSPLAT_CUDA_TYPES_CUH

#include <glm/glm.hpp>

namespace gsplat {

using vec2 = glm::vec<2, float>;
using vec3 = glm::vec<3, float>;
using vec4 = glm::vec<4, float>;
using mat2 = glm::mat<2, 2, float>;
using mat3 = glm::mat<3, 3, float>;
using mat4 = glm::mat<4, 4, float>;
using mat3x2 = glm::mat<3, 2, float>;

} // namespace gsplat

#endif // GSPLAT_CUDA_TYPES_CUH