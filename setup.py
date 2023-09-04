import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PROJ_ROOT = os.path.abspath(f"{__file__}/..")

setup(
    name="diff_rast",
    packages=["diff_rast"],
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, slatting, cuda",
    ext_modules=[
        CUDAExtension(
            name="diff_rast.cuda_lib",
            sources=["csrc/ext.cpp", "csrc/rasterize.cu", "csrc/bindings.cu", "csrc/forward.cu"],
            extra_compile_args={"nvcc": [f"-I {PROJ_ROOT}/csrc/third_party/glm/"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
