import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CSRC = os.path.abspath(f"{__file__}/../../csrc")

setup(
    name="ref_rast",
    # packages=["ref_rast"],
    # package_dir={"ref_rast": ""},
    description="reference package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    ext_modules=[
        CUDAExtension(
            name="ref_rast_cuda",
            sources=[
                f"{CSRC}/reference/cuda_rasterizer/rasterizer_impl.cu",
                f"{CSRC}/reference/cuda_rasterizer/forward.cu",
                f"{CSRC}/reference/cuda_rasterizer/backward.cu",
                f"{CSRC}/reference/rasterize_points.cu",
                f"{CSRC}/reference/ext.cpp",
            ],
            extra_compile_args={"nvcc": [f"-I {CSRC}/third_party/glm/"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
