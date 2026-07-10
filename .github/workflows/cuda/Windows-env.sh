#!/bin/bash

case ${1} in
  cu130)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v13.0
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    ;;
  cu129)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.9
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    ;;
  cu128)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.8
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    ;;
  cu126)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.6
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    ;;
  cu124)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.4
    export PATH=${CUDA_HOME}/bin:/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    ;;
  *)
    ;;
esac
