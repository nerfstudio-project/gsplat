#!/bin/bash

case ${1} in
  cu130)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v13.0
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;9.0a"
    ;;
  cu129)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.9
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;9.0a"
    ;;
  cu128)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.8
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;9.0a"
    ;;
  cu126)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.6
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
    ;;
  cu124)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.4
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
    ;;
  *)
    ;;
esac
