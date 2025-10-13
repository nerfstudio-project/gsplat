# gsplat

## Setup Environment

```bash
# Ubuntu 18.04/20.04 and CUDA 11.8
conda create -n gsplat python=3.10
conda activate gsplat
python -m pip install --upgrade pip
pip install numpy==1.26.4
pip install setuptools==78.1.1
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install -e . --no-build-isolation

# Alternative Method 2: Use legacy mode
# pip install -e . --use-pep517=false

pip install -r examples/requirements.txt
pip install jaxtyping
pip install toml ipykernel jupyter wandb
pip install open3d
conda install -c conda-forge libboost-devel
```

## Installing COLMAP 3.12.0

```bash
cd ~/tools
git clone --recurse-submodules https://github.com/colmap/colmap.git
cd colmap
git checkout tags/3.12.0 -b origin/main

# Dependencies Ubuntu 18.04
sudo apt-get update
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libcurl4-openssl-dev

sudo apt-get install libmkl-full-dev libgmock-dev # These libraries are not available in Ubuntu 18.04
cmake .. -GNinja -DBLA_VENDOR=Intel10_64lp -DCMAKE_CUDA_ARCHITECTURES=native # Throws Blas error

cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release # Works
ninja
sudo ninja install
```

## Training and Evaluation

```bash
cd examples

# download mipnerf_360 benchmark data
python datasets/download_dataset.py

# run batch evaluation
cd ../gsplat
bash examples/benchmarks/basic.sh
```