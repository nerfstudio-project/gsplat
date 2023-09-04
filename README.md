# diff_rast
Python package and cuda bindings for differentiable gaussian rasterization
# Installation 
```
python3 -m pip install --upgrade pip
pip install -e .
```

# Brief walkthrough
The main python bindings for rasterization are found by importing diff_rast 
```
import diff_rast
help(diff_rast)
```
Additional supported cuda functions are found by importing cuda_lib from diff_rast
```
from diff_rast import cuda_lib
help(cuda_lib)
```
