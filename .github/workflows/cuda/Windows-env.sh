#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu118)
    CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8
    PATH=${CUDA_HOME}/bin:$PATH
    PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu117)
    CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7
    PATH=${CUDA_HOME}/bin:$PATH
    PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu116)
    CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.6
    PATH=${CUDA_HOME}/bin:$PATH
    PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu115)
    CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.5
    PATH=${CUDA_HOME}/bin:$PATH
    PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu113)
    CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3
    PATH=${CUDA_HOME}/bin:$PATH
    PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  *)
    ;;
esac