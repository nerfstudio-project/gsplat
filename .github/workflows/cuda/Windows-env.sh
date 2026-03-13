#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu124)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.4
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
    ;;
  cu121)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.1
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
    ;;
  cu118)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
    ;;
  cu117)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
    ;;
  cu116)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.6
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
    ;;
  cu115)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.5
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
    ;;
  cu113)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
    ;;
  *)
    ;;
esac