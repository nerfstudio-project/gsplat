import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CSRC = os.path.abspath(f"{__file__}/../../csrc")

setup(
    name="diff_rast",
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    ext_modules=[
        CUDAExtension(
            name="cuda_lib",
            sources=[
                f"{CSRC}/ext.cpp",
                f"{CSRC}/rasterize.cu",
                f"{CSRC}/bindings.cu",
                f"{CSRC}/forward.cu",
                f"{CSRC}/backward.cu",
            ],
            extra_compile_args={"nvcc": [f"-I {CSRC}/third_party/glm/"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
