# Installing `gsplat` on Windows

## Install using a pre-compiled wheels 

You can install gsplat from python wheels containing pre-compiled binaries for a specific pytorch and cuda version. These wheels are stored in the github releases and can be found using simple index pages under https://docs.gsplat.studio/whl. 
You obtain the wheel from this simple index page for a specific pytorch an and cuda version by appending these the version number after a + sign (part referred a *local version*). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can use
```
pip install gsplat==1.2.0+pt20cu118 --index-url https://docs.gsplat.studio/whl
```
Alternatively, you can specify the pytorch and cuda version in the index url using for example
```
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```
This has the advantage that you do not have to pin a specific version of the package and as a result get automatically the latest package version.


## Source builds

The CMake source build currently supports Linux only. On Windows, use a
precompiled wheel whose PyTorch and CUDA versions match the active environment.
