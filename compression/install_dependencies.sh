#!/bin/bash
# Smart installation script for PCL, Draco, and Open3D
# Checks existing installations and installs only what's needed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

INSTALL_PREFIX="$HOME/.local"
NUM_CORES=$(nproc)

echo -e "${BLUE}=== Dependency Installation Script ===${NC}"
echo "Installation prefix: $INSTALL_PREFIX"
echo "Using $NUM_CORES CPU cores"
echo ""

#########################################
# 1. Check and install system dependencies
#########################################
echo -e "${BLUE}=== Checking System Dependencies ===${NC}"

SYSTEM_PACKAGES=(
    build-essential cmake git
    libeigen3-dev libflann-dev libboost-all-dev
    libvtk7-dev libvtk7-qt-dev
    libusb-1.0-0-dev
    python3-dev python3-pip
    libglfw3-dev libgl1-mesa-dev libxrandr-dev libxi-dev libxxf86vm-dev
    libjsoncpp-dev pkg-config
)

MISSING_PACKAGES=()

for pkg in "${SYSTEM_PACKAGES[@]}"; do
    if dpkg -s $pkg >/dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} $pkg installed"
    else
        echo -e "${YELLOW}[MISSING]${NC} $pkg not installed"
        MISSING_PACKAGES+=($pkg)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Missing packages: ${MISSING_PACKAGES[*]}${NC}"
    echo "Installing missing packages (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PACKAGES[@]}"
    echo -e "${GREEN}System dependencies installed!${NC}"
else
    echo -e "${GREEN}All system dependencies are installed!${NC}"
fi

echo ""

#########################################
# 2. Check for PCL
#########################################
echo -e "${BLUE}=== Checking PCL ===${NC}"

PCL_FOUND=false
PCL_ROOT=""

# Check via pkg-config
if pkg-config --exists pcl_common 2>/dev/null; then
    PCL_VERSION=$(pkg-config --modversion pcl_common)
    PCL_ROOT=$(pkg-config --variable=prefix pcl_common)
    echo -e "${GREEN}[FOUND]${NC} PCL $PCL_VERSION at $PCL_ROOT"
    PCL_FOUND=true
else
    # Check common locations
    for location in "$HOME/.local" "$HOME/tools/pcl-1.15.1/install" "/usr/local" "/usr"; do
        if [ -f "$location/lib/pkgconfig/pcl_common.pc" ] || [ -f "$location/lib/cmake/pcl/PCLConfig.cmake" ]; then
            PCL_ROOT="$location"
            echo -e "${GREEN}[FOUND]${NC} PCL at $PCL_ROOT"
            PCL_FOUND=true
            break
        fi
    done
fi

if [ "$PCL_FOUND" = false ]; then
    echo -e "${YELLOW}[NOT FOUND]${NC} Installing PCL 1.15.1 to $INSTALL_PREFIX"

    mkdir -p ~/build-libs
    cd ~/build-libs

    if [ ! -d "pcl" ]; then
        git clone https://github.com/PointCloudLibrary/pcl.git pcl
    fi
    cd pcl
    git fetch --tags
    git checkout pcl-1.15.1

    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DBUILD_SHARED_LIBS=ON \
        -DWITH_CUDA=ON \
        -DWITH_OPENGL=ON \
        -DWITH_QT=OFF \
        -DWITH_VTK=ON \
        -DBUILD_GPU=ON \
        -DBUILD_examples=OFF \
        -DBUILD_tools=OFF \
        -DBUILD_tests=OFF

    make -j$NUM_CORES
    make install

    PCL_ROOT="$INSTALL_PREFIX"
    echo -e "${GREEN}[INSTALLED]${NC} PCL to $PCL_ROOT"
fi

echo ""

#########################################
# 3. Check for Draco
#########################################
echo -e "${BLUE}=== Checking Draco ===${NC}"

DRACO_FOUND=false
DRACO_ROOT=""

# Check common locations
for location in "$HOME/.local" "$HOME/tools/draco-1.5.7/install" "/usr/local" "/usr"; do
    if [ -f "$location/lib/cmake/draco/dracoConfig.cmake" ] || [ -f "$location/lib/libdraco.so" ]; then
        DRACO_ROOT="$location"
        echo -e "${GREEN}[FOUND]${NC} Draco at $DRACO_ROOT"
        DRACO_FOUND=true
        break
    fi
done

if [ "$DRACO_FOUND" = false ]; then
    echo -e "${YELLOW}[NOT FOUND]${NC} Installing Draco 1.5.7 to $INSTALL_PREFIX"

    mkdir -p ~/build-libs
    cd ~/build-libs

    if [ ! -d "draco" ]; then
        git clone https://github.com/google/draco.git draco
    fi
    cd draco
    git fetch --tags
    git checkout 1.5.7

    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DBUILD_SHARED_LIBS=ON \
        -DDRACO_BUILD_MAYA_PLUGIN=OFF \
        -DDRACO_BUILD_UNITY_PLUGIN=OFF

    make -j$NUM_CORES
    make install

    DRACO_ROOT="$INSTALL_PREFIX"
    echo -e "${GREEN}[INSTALLED]${NC} Draco to $DRACO_ROOT"
fi

echo ""

#########################################
# 4. Check for Open3D
#########################################
echo -e "${BLUE}=== Checking Open3D ===${NC}"

OPEN3D_FOUND=false
OPEN3D_ROOT=""

# Check common locations
for location in "$HOME/.local" "$HOME/tools/open3d-v0.19.0/install" "/usr/local" "/usr"; do
    if [ -f "$location/lib/cmake/Open3D/Open3DConfig.cmake" ] || [ -f "$location/lib/libOpen3D.so" ]; then
        OPEN3D_ROOT="$location"
        echo -e "${GREEN}[FOUND]${NC} Open3D at $OPEN3D_ROOT"
        OPEN3D_FOUND=true
        break
    fi
done

if [ "$OPEN3D_FOUND" = false ]; then
    echo -e "${YELLOW}[NOT FOUND]${NC} Installing Open3D v0.19.0 to $INSTALL_PREFIX"

    mkdir -p ~/build-libs
    cd ~/build-libs

    if [ ! -d "Open3D" ]; then
        git clone --recursive https://github.com/isl-org/Open3D.git Open3D
    fi
    cd Open3D
    git fetch --tags
    git checkout v0.19.0
    git submodule update --init --recursive

    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
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

    OPEN3D_ROOT="$INSTALL_PREFIX"
    echo -e "${GREEN}[INSTALLED]${NC} Open3D to $OPEN3D_ROOT"
fi

echo ""

#########################################
# 5. Summary and Environment Setup
#########################################
echo -e "${GREEN}=== Installation Summary ===${NC}"
echo "PCL:    $PCL_ROOT"
echo "Draco:  $DRACO_ROOT"
echo "Open3D: $OPEN3D_ROOT"
echo ""

# Save paths to a config file for CMake
CONFIG_FILE="$(dirname "$0")/dependency_paths.cmake"
cat > "$CONFIG_FILE" << EOF
# Auto-generated by install_dependencies.sh
set(PCL_ROOT "$PCL_ROOT" CACHE PATH "PCL installation directory")
set(DRACO_ROOT "$DRACO_ROOT" CACHE PATH "Draco installation directory")
set(Open3D_ROOT "$OPEN3D_ROOT" CACHE PATH "Open3D installation directory")
EOF

echo -e "${GREEN}Dependency paths saved to: $CONFIG_FILE${NC}"
echo ""

echo -e "${BLUE}Add these to your ~/.bashrc (if needed):${NC}"
echo ""
echo "export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
echo "export LD_LIBRARY_PATH=\"$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:\$LD_LIBRARY_PATH\""
echo "export PKG_CONFIG_PATH=\"$INSTALL_PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH\""
echo "export CMAKE_PREFIX_PATH=\"$INSTALL_PREFIX:\$CMAKE_PREFIX_PATH\""
echo ""
echo -e "${GREEN}=== All Done! ===${NC}"
