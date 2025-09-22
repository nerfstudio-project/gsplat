"""
Setup script for stream CUDA extensions.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get CUDA toolkit path
cuda_home = torch.utils.cpp_extension.CUDA_HOME
if cuda_home is None:
    raise RuntimeError("CUDA_HOME environment variable is not set")

# Define source files
cpp_sources = [
    "ext.cpp"
]

cuda_sources = [
    "csrc/clustering.cu"
]

# Combine all sources
sources = cpp_sources + cuda_sources

# Define include directories
include_dirs = [
    "include",
    os.path.join(cuda_home, "include"),
] + torch.utils.cpp_extension.include_paths()

# Define library directories  
library_dirs = [
    os.path.join(cuda_home, "lib64"),
] + torch.utils.cpp_extension.library_paths()

# Define libraries
libraries = ["cudart", "cublas", "curand"]

# Define compiler flags
cxx_flags = ["-O3", "-std=c++17"]
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "--extended-lambda",
    "--expt-relaxed-constexpr",
    "-Xcompiler", "-fPIC",
    "-gencode", "arch=compute_70,code=sm_70",  # For V100
    "-gencode", "arch=compute_75,code=sm_75",  # For RTX 2080
    "-gencode", "arch=compute_80,code=sm_80",  # For A100
    "-gencode", "arch=compute_86,code=sm_86",  # For RTX 3090
]

# Define Debug compiler flags
# cxx_flags = ["-g", "-std=c++17"]
# nvcc_flags = [
#     "-g",
#     "-G",
#     "-std=c++17",
#     "--use_fast_math",
#     "--extended-lambda",
#     "--expt-relaxed-constexpr",
#     "-Xcompiler", "-fPIC",
#     "-gencode", "arch=compute_70,code=sm_70",  # For V100
#     "-gencode", "arch=compute_75,code=sm_75",  # For RTX 2080
#     "-gencode", "arch=compute_80,code=sm_80",  # For A100
#     "-gencode", "arch=compute_86,code=sm_86",  # For RTX 3090
# ]

# Create extension
ext_modules = [
    CUDAExtension(
        name="stream_cuda_ext",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )
]

setup(
    name="stream_cuda",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension,
    },
    zip_safe=False,
)
