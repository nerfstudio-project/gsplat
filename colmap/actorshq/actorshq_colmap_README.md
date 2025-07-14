# ActorsHQ Colmap Preprocessing

## Installing COLMAP 3.12.0

```bash
cd ~/tools
git clone --recurse-submodules https://github.com/colmap/colmap.git
cd colmap
git checkout tags/3.12.1 -b origin/main

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
mkdir build && cd build
cmake .. -GNinja -DBLA_VENDOR=Intel10_64lp -DCMAKE_CUDA_ARCHITECTURES=native # Throws Blas error

cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release # Works, built with CUDA 11.8
ninja
sudo ninja install
```

## Dataset Structure

### ActorsHQ Format
```
actorshq/
├── Actor01/
│   └── Sequence1/
│       ├── scene.json                          # {"num_frames": 2214}
│       ├── aabbs.csv                           # AABB of the scene
│       ├── occupancy_grids/
│       │   ├── occupancy_grid000000.npz
│       │   └── ...
│       ├── 4x/                                 # Quarter resolution
│       │   ├── calibration.csv                 # Camera intrinsics & extrinsics (162 cameras)
│       │   ├── light_annotations.csv
│       │   ├── rgbs/
│       │   │   ├── Cam001/                     # RGB images per camera
│       │   │   │   ├── Cam001_rgb000000.jpg
│       │   │   │   ├── Cam001_rgb000001.jpg
│       │   │   │   └── ...
│       │   │   ├── Cam002/
│       │   │   │   ├── Cam002_rgb000000.jpg
│       │   │   │   └── ...
│       │   │   └── ...                         # Up to 160 cameras
│       │   └── masks/                          # Segmentation masks
│       │       ├── Cam001/
│       │       │   ├── Cam001_mask000000.png
│       │       │   └── ...
│       │       ├── Cam002/
│       │       │   └── ...
│       │       └── ...                         # Up to 160 cameras
│       ├── 2x/                                 # Half resolution
│       │   ├── calibration.csv
│       │   ├── light_annotations.csv
│       │   ├── rgbs/
│       │   │   ├── Cam001/
│       │   │   │   └── ...
│       │   │   └── ...
│       │   └── masks/
│       │       ├── Cam001/
│       │       │   └── ...
│       │       └── ...
│       └── 1x/                                 # Full resolution
│           ├── calibration.csv
│           ├── light_annotations.csv
│           ├── rgbs/
│           │   ├── Cam001/
│           │   │   └── ...
│           │   └── ...
│           └── masks/
│               ├── Cam001/
│               │   └── ...
│               └── ...
└── ...                             # Additional actors follow same structure
```

### Convert ActorsHQ to COLMAP input format

```
actorshq/
└── colmap/
   ├── Actor01/
   │   ├── Sequence1/
   │   │   ├── metadata/                          
   │   │   │   ├── scene.json                      # {"num_frames": 2214}
   │   │   │   └── aabbs.csv                       # AABB of the scene
   │   │   ├── calibration_gt_4/                   # GT Calibration and light for quarter resolution
   │   │   │   ├── calibration.csv                 # Camera intrinsics & extrinsics (160 cameras)
   │   │   │   └── light_annotations.csv           # Light annotations
   │   │   ├── calibration_gt_2/                   # GT Calibration and light for half resolution
   │   │   │   ├── calibration.csv                 # Camera intrinsics & extrinsics (160 cameras)
   │   │   │   └── light_annotations.csv           # Light annotations
   │   │   ├── calibration_gt/                     # GT Calibration and light for full resolution
   │   │   │   ├── calibration.csv                 # Camera intrinsics & extrinsics (160 cameras)
   │   │   │   └── light_annotations.csv           # Light annotations
   │   │   └── frames/
   │   │       ├── frame0/
   │   │       │   ├── occupancy_grid000000.npz
   │   │       │   ├── images_4/                       # Quarter resolution
   │   │       │   │   ├── Cam001_rgb000000.png
   │   │       │   │   ├── Cam002_rgb000000.png
   │   │       │   │   └── ...
   │   │       │   │   └── Cam160_rgb000000.png
   │   │       │   ├── images_2/                       # Half resolution
   │   │       │   │   ├── Cam001_rgb000000.png
   │   │       │   │   ├── Cam002_rgb000000.png
   │   │       │   │   └── ...
   │   │       │   │   └── Cam160_rgb000000.png
   │   │       │   ├── images/                         # Full resolution
   │   │       │   │   ├── Cam001_rgb000000.png
   │   │       │   │   ├── Cam002_rgb000000.png
   │   │       │   │   └── ...
   │   │       │   │   └── Cam160_rgb000000.png
   │   │       │   ├── masks_4/                        # Quarter resolution
   │   │       │   │   ├── Cam001_mask000000.png
   │   │       │   │   ├── Cam002_mask000000.png
   │   │       │   │   └── ...
   │   │       │   │   └── Cam160_mask000000.png
   │   │       │   ├── masks_2/                        # Half resolution
   │   │       │   │   ├── Cam001_mask000000.png
   │   │       │   │   ├── Cam002_mask000000.png
   │   │       │   │   └── ...
   │   │       │   │   └── Cam160_mask000000.png
   │   │       │   └── masks/                          # Full resolution
   │   │       │       ├── Cam001_mask000000.png
   │   │       │       ├── Cam002_mask000000.png
   │   │       │       └── ...
   │   │       │       └── Cam160_mask000000.png
   │   │       └── frame1/
   │   │       │   └── ...
   │   │       └── frame2/
   │   │       │   └── ...
   │   │       └── ...
   │   └── Sequence2/
   │   │   └── ...
   └── Actor02/
   │   └── Sequence1/
   │   │   └── ...
   └── ...
```

```bash
cd colmap/
pip install -r requirements.txt

cd actorshq/

# Convert All Actors from Custom Input Path
python convert_actorshq_to_colmap_input_format.py --input /bigdata2/rajrup/datasets/actorshq/ --output /bigdata2/rajrup/datasets/actorshq/

# Convert Specific Actor
python convert_actorshq_to_colmap_input_format.py --actor Actor01 --input /bigdata2/rajrup/datasets/actorshq/ --output /bigdata2/rajrup/datasets/actorshq/

# Convert Specific Actor and Sequence
python convert_actorshq_to_colmap_input_format.py --actor Actor01 --sequence Sequence1 --input /bigdata2/rajrup/datasets/actorshq/ --output /bigdata2/rajrup/datasets/actorshq/
``` 

### Convert ActorsHQ camera parameters to COLMAP format

```bash
cd colmap/actorshq/
bash export_actorshq_to_colmap.sh # From https://github.com/synthesiaresearch/humanrf/issues/32
# Change parameters to export_actorshq_to_colmap.sh to export the correct resolution.
## Note that camera id is 1-indexed in COLMAP. Check the camera.txt file to see the camera ids start from 1. Also, check the images.txt file to see the camera ids are 1-indexed.
## Otherwise, the point triangulation will fail.
```

### Run COLMAP

- Running on one frame of ActorsHQ dataset. Example: Actor01, Sequence1, frame0.
- Run COLMAP with without known poses. This can run for several minutes. It can produce several sparse models without taking all the cameras into account.

```bash
cd /bigdata2/rajrup/datasets/actorshq/colmap/Actor01/Sequence1/frames/frame0/

## Full Colmap without known poses -> This works best with parameters suggested here: https://github.com/synthesiaresearch/humanrf/issues/35
colmap feature_extractor \
    --database_path ./database_4.db \
    --image_path ./images_4 \
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.peak_threshold 0.001 \
    --SiftExtraction.edge_threshold 80 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index 0

colmap exhaustive_matcher \
    --database_path ./database_4.db \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index 0

colmap mapper \
    --database_path ./database_4.db \
    --image_path ./images_4 \
    --output_path ./sparse_4 \
    --Mapper.multiple_models 0

colmap model_converter \
    --input_path ./sparse_4 \
    --output_path ./sparse_4 \
    --output_type TXT

# This created a sparse model with 159 cameras and 35420 points.
```

- Run COLMAP with known poses. This can run for several seconds. It can produce a sparse model with all the cameras. Refer to [COLMAP known poses](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses) for more details.

```bash
cd /bigdata2/rajrup/datasets/actorshq/colmap/Actor01/Sequence1/frames/frame0/

## Note that camera id is 1-indexed in COLMAP. Check the camera.txt file to see the camera ids start from 1. Also, check the images.txt file to see the camera ids are 1-indexed.
## Otherwise, the point triangulation will fail.

# Run COLMAP with known poses for 4x resolution.
# Test 1: This works best with parameters suggested here: https://github.com/synthesiaresearch/humanrf/issues/35
# This created a sparse model with 160 cameras and 26157 points.
colmap feature_extractor \
    --database_path ./database_gt_4.db \
    --image_path ./images_4 \
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.peak_threshold 0.001 \
    --SiftExtraction.edge_threshold 80

colmap exhaustive_matcher \
    --database_path ./database_gt_4.db \
    --SiftMatching.guided_matching 1
#--SequentialMatching.vocab_tree_path ""

mkdir -p ./triangulated_gt_4/
colmap point_triangulator \
    --database_path ./database_gt_4.db \
    --image_path ./images_4 \
    --input_path ./sparse_gt_4 \
    --output_path ./triangulated_gt_4 \
    --log_level 3

colmap model_converter \
    --input_path ./triangulated_gt_4 \
    --output_path ./triangulated_gt_4 \
    --output_type TXT

# Test 2 (Bad): Using default parameters.
# This created a sparse model with 160 cameras and 2255 points.
colmap feature_extractor \
    --database_path ./database_gt_4_test2.db \
    --image_path ./images_4 \
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index 0

colmap exhaustive_matcher \
    --database_path ./database_gt_4_test2.db \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index 0

mkdir -p ./triangulated_gt_4_test2/
colmap point_triangulator \
    --database_path ./database_gt_4_test2.db \
    --image_path ./images_4 \
    --input_path ./sparse_gt_4_test2 \
    --output_path ./triangulated_gt_4_test2 \
    --log_level 3

colmap model_converter \
    --input_path ./triangulated_gt_4_test2 \
    --output_path ./triangulated_gt_4_test2 \
    --output_type TXT

## Running COLMAP with known poses for 1x resolution
cd colmap/actorshq/
bash export_actorshq_to_colmap.sh # Change parameters to export_actorshq_to_colmap.sh to export the correct resolution

cd /bigdata2/rajrup/datasets/actorshq/colmap/Actor01/Sequence1/frames/frame0/
colmap feature_extractor \
    --database_path ./database_gt.db \
    --image_path ./images \
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.peak_threshold 0.001 \
    --SiftExtraction.edge_threshold 80

colmap exhaustive_matcher \
    --database_path ./database_gt.db \
    --SiftMatching.guided_matching 1

mkdir -p ./triangulated_gt/
colmap point_triangulator \
    --database_path ./database_gt.db \
    --image_path ./images \
    --input_path ./sparse_gt \
    --output_path ./triangulated_gt

colmap model_converter \
    --input_path ./triangulated_gt \
    --output_path ./triangulated_gt \
    --output_type TXT

## Running COLMAP with known poses for 2x resolution
cd colmap/actorshq/
bash export_actorshq_to_colmap.sh # Change parameters to export_actorshq_to_colmap.sh to export the correct resolution

cd /bigdata2/rajrup/datasets/actorshq/colmap/Actor01/Sequence1/frames/frame0/
colmap feature_extractor \
    --database_path ./database_gt_2.db \
    --image_path ./images_2 \
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.peak_threshold 0.001 \
    --SiftExtraction.edge_threshold 80

colmap exhaustive_matcher \
    --database_path ./database_gt_2.db \
    --SiftMatching.guided_matching 1

mkdir -p ./triangulated_gt_2/
colmap point_triangulator \
    --database_path ./database_gt_2.db \
    --image_path ./images_2 \
    --input_path ./sparse_gt_2 \
    --output_path ./triangulated_gt_2

colmap model_converter \
    --input_path ./triangulated_gt_2 \
    --output_path ./triangulated_gt_2 \
    --output_type TXT
```

### View COLMAP

- Run `colmap gui`
- Import model - `File -> Import Model -> Select folder that contains camera.txt, images.txt, and points3D.txt`
- Example: `File -> Import Model -> /bigdata2/rajrup/datasets/actorshq/colmap/Actor01/Sequence1/frames/frame0/triangulated_gt_4`
- Using Open3D to view the model.

```bash
cd colmap/actorshq/

python visualize_model.py --input_model /main/rajrup/Dropbox/Project/GsplatStream/gsplat/data/Actor01/Sequence1/frames/frame0/resolution_4/sparse/0 --input_format .bin
```