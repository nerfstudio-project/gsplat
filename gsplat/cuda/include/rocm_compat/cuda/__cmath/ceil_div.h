#pragma once

#include <type_traits>

namespace cuda
{

template<typename T>
constexpr T ceil_div(T n, T d)
{
    return (n + d - 1) / d;
}

} // namespace cuda
