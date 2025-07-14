#!/bin/bash

# Initialize conda for bash script
source ~/miniconda3/etc/profile.d/conda.sh
# Alternative: source $(conda info --base)/etc/profile.d/conda.sh

cd /main/rajrup/Dropbox/Project/GsplatStream/humanrf
conda activate humanrf

dataset_path=/bigdata2/rajrup/datasets/actorshq/colmap/Actor01/Sequence1/
start_frame=0
end_frame=0

# Use seq or C-style for loop instead of brace expansion with variables
for frame in $(seq $start_frame $end_frame)
do
    echo "Processing frame $frame"
    python actorshq/toolbox/export_colmap.py \
        --csv ${dataset_path}/calibration_gt_2/calibration.csv \
        --output_dir ${dataset_path}/${frame}/sparse_gt_2/ \
        --frame_id $frame
done

conda deactivate
