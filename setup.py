import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PROJ_ROOT = os.path.abspath(f"{__file__}/..")

setup(
    name="diff_rast",
    ext_modules=[
        CUDAExtension(
            "diff_rast",
            ["csrc/ext.cpp", "csrc/rasterize.cu", "csrc/bindings.cu", "csrc/forward.cu"],
            extra_compile_args={"nvcc": [f"-I {PROJ_ROOT}/csrc/third_party/glm/"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
