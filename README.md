# diff_rast
Our version of differentiable gaussian rasterizer

# ref_rast
Copied official version of differentiable gaussian rasterizer

# Installation 
```
python3 -m pip install --upgrade pip
cd diff_rast; pip install -e .
cd ../ref_rast; pip install -e .
```

# Brief walkthrough
The main python bindings for rasterization are found by importing diff_rast 
```
import diff_rast
help(diff_rast)
```
Additional supported cuda functions are found by importing cuda_lib
```
import cuda_lib
help(cuda_lib)
```
