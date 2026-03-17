#!/bin/bash

# Install NVIDIA driver stub DLLs (needed for linking, not for runtime).
# CUDA 13+ no longer bundles the driver, but we still need the stubs for building.
curl -k -L "https://drive.google.com/u/0/uc?id=1injUyo3lnarMgWyRcXqKg4UGnN0ysmuq&export=download" --output "/tmp/gpu_driver_dlls.zip"
7z x "/tmp/gpu_driver_dlls.zip" -o"/c/Windows/System32"

case ${1} in
  cu130)
    CUDA_SHORT=13.0
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
    # CUDA 13+ no longer bundles the driver, so the filename has no driver version.
    CUDA_FILE=cuda_${CUDA_SHORT}.0_windows.exe
    ;;
  cu129)
    CUDA_SHORT=12.9
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.0_576.02_windows.exe
    ;;
  cu128)
    CUDA_SHORT=12.8
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.1/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.1_572.61_windows.exe
    ;;
  cu126)
    CUDA_SHORT=12.6
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.3/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.3_561.17_windows.exe
    ;;
  cu124)
    CUDA_SHORT=12.4
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.1/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.1_551.78_windows.exe
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${1}"
    exit 1
    ;;
esac

curl -k -L "${CUDA_URL}/${CUDA_FILE}" --output "${CUDA_FILE}"
echo ""
echo "Installing from ${CUDA_FILE}..."

# Base components needed for building CUDA extensions.
CUDA_COMPONENTS="nvcc_${CUDA_SHORT} cuobjdump_${CUDA_SHORT} nvprune_${CUDA_SHORT} cupti_${CUDA_SHORT} cublas_dev_${CUDA_SHORT} cudart_${CUDA_SHORT} cufft_dev_${CUDA_SHORT} curand_dev_${CUDA_SHORT} cusolver_dev_${CUDA_SHORT} cusparse_dev_${CUDA_SHORT} thrust_${CUDA_SHORT} npp_dev_${CUDA_SHORT} nvrtc_dev_${CUDA_SHORT} nvml_dev_${CUDA_SHORT}"

# CUDA 13+ split several components out of nvcc into their own subpackages:
#   crt       - compiler headers (crt/host_config.h)
#   nvvm      - CUDA IR compiler (cicc)
#   nvfatbin  - fatbinary combiner
#   nvptxcompiler - PTX compiler
#   visual_studio_integration - MSVC integration for --use-local-env
CUDA_MAJOR=${CUDA_SHORT%%.*}
if [ "${CUDA_MAJOR}" -ge 13 ]; then
    CUDA_COMPONENTS="${CUDA_COMPONENTS} crt_${CUDA_SHORT} nvvm_${CUDA_SHORT} nvfatbin_${CUDA_SHORT} nvptxcompiler_${CUDA_SHORT} visual_studio_integration_${CUDA_SHORT}"
fi

PowerShell -Command "Start-Process -FilePath \"${CUDA_FILE}\" -ArgumentList \"-s ${CUDA_COMPONENTS}\" -Wait -NoNewWindow"
echo "Done!"
rm -f "${CUDA_FILE}"

echo Installing NvToolsExt...
curl -k -L https://ossci-windows.s3.us-east-1.amazonaws.com/builder/NvToolsExt.7z --output /tmp/NvToolsExt.7z
7z x /tmp/NvToolsExt.7z -o"/tmp/NvToolsExt"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/include"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64"
cp -r /tmp/NvToolsExt/bin/x64/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64"
cp -r /tmp/NvToolsExt/include/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/include"
cp -r /tmp/NvToolsExt/lib/x64/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64"
