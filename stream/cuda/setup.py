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

def find_boost_paths():
    """Find Boost installation paths using conda environment"""
    boost_include_dirs = []
    boost_library_dirs = []
    
    # Get conda environment paths
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        raise RuntimeError("CONDA_PREFIX not found. Make sure you're in a conda environment.")
    
    # Conda environment paths
    conda_include = os.path.join(conda_prefix, 'include')
    conda_lib = os.path.join(conda_prefix, 'lib')
    
    # Check if Boost is available in conda environment
    boost_include_path = os.path.join(conda_include, 'boost')
    if os.path.exists(boost_include_path):
        boost_include_dirs.append(conda_include)
        print(f"Found Boost headers in conda environment: {conda_include}")
    else:
        raise RuntimeError(f"Boost headers not found in conda environment at {boost_include_path}")
    
    # Check for Boost libraries in conda lib directory
    if os.path.exists(conda_lib):
        try:
            files = os.listdir(conda_lib)
            boost_libs = [f for f in files if f.startswith('libboost_')]
            if boost_libs:
                boost_library_dirs.append(conda_lib)
                print(f"Found Boost libraries in conda environment: {conda_lib}")
            else:
                raise RuntimeError(f"Boost libraries not found in conda environment at {conda_lib}")
        except OSError as e:
            raise RuntimeError(f"Cannot access conda lib directory {conda_lib}: {e}")
    else:
        raise RuntimeError(f"Conda lib directory not found: {conda_lib}")
    
    return boost_include_dirs, boost_library_dirs

# Find Boost paths
boost_include_dirs, boost_library_dirs = find_boost_paths()

# Print found paths for debugging
print(f"Boost include directories: {boost_include_dirs}")
print(f"Boost library directories: {boost_library_dirs}")

# Define source files
cpp_sources = [
    "ext.cpp",
    "timer.cpp",
    "cuda_timer.cpp"
]

cuda_sources = [
    "csrc/clustering.cu",
    "csrc/merging.cu"
]

# Combine all sources
sources = cpp_sources + cuda_sources

# Define include directories
include_dirs = [
    "include",
    os.path.join(cuda_home, "include"),
] + boost_include_dirs + torch.utils.cpp_extension.include_paths()

# Define library directories  
library_dirs = [
    os.path.join(cuda_home, "lib64"),
] + boost_library_dirs + torch.utils.cpp_extension.library_paths()

# Define libraries
libraries = ["cudart", "cublas", "curand", "boost_chrono", "boost_system", "boost_date_time"]

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

# # Define Debug compiler flags
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
