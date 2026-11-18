#pragma once

// HIPify rewrites <cuda/...> includes; keep ceil_div local to avoid broken paths.
namespace cuda
{

template<typename T>
constexpr T ceil_div(T n, T d)
{
    return (n + d - 1) / d;
}

} // namespace cuda
