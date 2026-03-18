#!/bin/bash

case ${1} in
  cu130)
    export CUDA_HOME=/usr/local/cuda-13.0
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;9.0a"
    ;;
  cu129)
    export CUDA_HOME=/usr/local/cuda-12.9
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;9.0a"
    ;;
  cu128)
    export CUDA_HOME=/usr/local/cuda-12.8
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;9.0a"
    ;;
  cu126)
    export CUDA_HOME=/usr/local/cuda-12.6
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
    ;;
  cu124)
    export CUDA_HOME=/usr/local/cuda-12.4
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
    ;;
  cu121)
    export CUDA_HOME=/usr/local/cuda-12.1
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
    ;;
  *)
    ;;
esac
