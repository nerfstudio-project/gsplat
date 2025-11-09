#!/bin/bash
# Installation script for PCL, Draco, and Open3D with GPU support
# Installs to ~/tools/

set -e  # Exit on error

TOOLS_DIR="$HOME/tools"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_CORES=$(nproc)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Installing PCL, Draco, and Open3D ===${NC}"
echo "Installation directory: $TOOLS_DIR"
echo "Using $NUM_CORES CPU cores"

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}CUDA found: version $CUDA_VERSION${NC}"
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
else
    echo -e "${YELLOW}CUDA not found - GPU support will be limited${NC}"
fi

# Version numbers
DRACO_VERSION="1.5.7"
PCL_VERSION="1.15.1"
OPEN3D_VERSION="v0.19.0"

# Create tools directory structure
mkdir -p "$TOOLS_DIR/"

# Export paths (will be updated as libraries are installed)
export PATH="$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"

cd "$TOOLS_DIR"

echo -e "\n${GREEN}=== Installing dependencies ===${NC}"
# Install system dependencies
sudo apt-get update
sudo apt-get install \
    build-essential cmake git \
    libeigen3-dev libflann-dev libboost-all-dev \
    libvtk7-dev libvtk7-qt-dev \
    libusb-1.0-0-dev \
    python3-dev python3-pip \
    libglfw3-dev libgl1-mesa-dev libxrandr-dev libxi-dev libxxf86vm-dev

echo -e "\n${GREEN}=== 1. Installing Draco ===${NC}"
cd "$TOOLS_DIR/"
DRACO_DIR="draco-${DRACO_VERSION}"
# if [ ! -d "$DRACO_DIR" ]; then
#     git clone https://github.com/google/draco.git "$DRACO_DIR"
# fi
# cd "$DRACO_DIR"
# git fetch --tags
# git checkout tags/${DRACO_VERSION}  # Use stable version tag

# mkdir -p build install
# cd build
# cmake .. \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX="../install" \
#     -DDRACO_BUILD_SHARED_LIBS=ON \
#     -DDRACO_BUILD_MAYA_PLUGIN=OFF \
#     -DDRACO_BUILD_UNITY_PLUGIN=OFF
# make -j$NUM_CORES
# make install

# Update environment paths
DRACO_INSTALL="$TOOLS_DIR/$DRACO_DIR/install"
export PATH="$DRACO_INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$DRACO_INSTALL/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$DRACO_INSTALL/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$DRACO_INSTALL:$CMAKE_PREFIX_PATH"

echo -e "\n${GREEN}=== 2. Installing PCL ===${NC}"
cd "$TOOLS_DIR/"
PCL_DIR="pcl-${PCL_VERSION}"
# if [ ! -d "$PCL_DIR" ]; then
#     git clone https://github.com/PointCloudLibrary/pcl.git "$PCL_DIR"
# fi
# cd "$PCL_DIR"
# git fetch --tags
# git checkout tags/pcl-${PCL_VERSION}  # Use latest stable version tag

# mkdir -p build install
# cd build
# cmake .. \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX="../install" \
#     -DBUILD_SHARED_LIBS=ON \
#     -DWITH_CUDA=ON \
#     -DWITH_OPENGL=ON \
#     -DWITH_QT=OFF \
#     -DWITH_VTK=ON \
#     -DBUILD_GPU=ON \
#     -DBUILD_examples=OFF \
#     -DBUILD_tools=OFF \
#     -DBUILD_tests=OFF
# make -j$NUM_CORES
# make install

# Update environment paths
PCL_INSTALL="$TOOLS_DIR/$PCL_DIR/install"
export PATH="$PCL_INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$PCL_INSTALL/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PCL_INSTALL/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$PCL_INSTALL:$CMAKE_PREFIX_PATH"

echo -e "\n${GREEN}=== 3. Installing Open3D ===${NC}"
cd "$TOOLS_DIR/"
OPEN3D_DIR="open3d-${OPEN3D_VERSION}"
if [ ! -d "$OPEN3D_DIR" ]; then
    git clone --recursive https://github.com/isl-org/Open3D.git "$OPEN3D_DIR"
fi
cd "$OPEN3D_DIR"
git fetch --tags
git checkout tags/${OPEN3D_VERSION}  # Use latest stable version tag
git submodule update --init --recursive

mkdir -p build install
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="../install" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_PYTHON_MODULE=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GUI=ON \
    -DBUILD_WEBRTC=OFF \
    -DUSE_SYSTEM_EIGEN3=OFF \
    -DBUILD_CUDA_MODULE=ON \
    -DUSE_CUDA=ON
make -j$NUM_CORES
make install

# Update environment paths
OPEN3D_INSTALL="$TOOLS_DIR/$OPEN3D_DIR/install"
export PATH="$OPEN3D_INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$OPEN3D_INSTALL/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$OPEN3D_INSTALL/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$OPEN3D_INSTALL:$CMAKE_PREFIX_PATH"

echo -e "\n${GREEN}=== Installation Complete! ===${NC}"
echo ""
echo "Add these to your ~/.bashrc or ~/.zshrc:"
echo ""
echo "export PCL_ROOT=\"$PCL_INSTALL\""
echo "export DRACO_ROOT=\"$DRACO_INSTALL\""
echo "export Open3D_ROOT=\"$OPEN3D_INSTALL\""
echo "export PATH=\"\$PCL_INSTALL/bin:\$DRACO_INSTALL/bin:\$OPEN3D_INSTALL/bin:\$PATH\""
echo "export LD_LIBRARY_PATH=\"\$PCL_INSTALL/lib:\$DRACO_INSTALL/lib:\$OPEN3D_INSTALL/lib:\$LD_LIBRARY_PATH\""
echo "export PKG_CONFIG_PATH=\"\$PCL_INSTALL/lib/pkgconfig:\$DRACO_INSTALL/lib/pkgconfig:\$OPEN3D_INSTALL/lib/pkgconfig:\$PKG_CONFIG_PATH\""
echo "export CMAKE_PREFIX_PATH=\"\$PCL_INSTALL:\$DRACO_INSTALL:\$OPEN3D_INSTALL:\$CMAKE_PREFIX_PATH\""

