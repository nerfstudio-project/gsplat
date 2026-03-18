#!/bin/bash

# Install CUDA toolkit components via the NVIDIA network repository.
# This approach is more maintainable than downloading local .deb installers
# because it does not require tracking exact driver-version filenames.

set -euo pipefail

OS=ubuntu2204

case ${1} in
  cu130)
    CUDA_MAJOR_MINOR=13.0
    ;;
  cu129)
    CUDA_MAJOR_MINOR=12.9
    ;;
  cu128)
    CUDA_MAJOR_MINOR=12.8
    ;;
  cu126)
    CUDA_MAJOR_MINOR=12.6
    ;;
  cu124)
    CUDA_MAJOR_MINOR=12.4
    ;;
  cu121)
    CUDA_MAJOR_MINOR=12.1
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${1}"
    exit 1
    ;;
esac

CUDA_DASHED=${CUDA_MAJOR_MINOR/./-}

# Install the CUDA keyring for repository authentication
wget -nv "https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb

sudo apt-get -qq update
sudo apt-get -qq install -y \
  cuda-nvcc-${CUDA_DASHED} \
  cuda-libraries-dev-${CUDA_DASHED} \
  cuda-command-line-tools-${CUDA_DASHED}
sudo apt-get clean
