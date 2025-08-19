#!/bin/bash

# Initialize conda for bash script
source ~/miniconda3/etc/profile.d/conda.sh
# Alternative: source $(conda info --base)/etc/profile.d/conda.sh

cd /main/rajrup/Dropbox/Project/GsplatStream/humanrf
conda activate humanrf

actor_name=Actor08
seq_name=Sequence1
resolution=4
start_frame=0
end_frame=0
dataset_path=/bigdata2/rajrup/datasets/actorshq/colmap/${actor_name}/${seq_name}/


# Use seq or C-style for loop instead of brace expansion with variables
for frame in $(seq $start_frame $end_frame)
do
    echo "Processing frame $frame"
    python actorshq/toolbox/export_colmap.py \
        --csv ${dataset_path}/calibration_gt_${resolution}/calibration.csv \
        --output_dir ${dataset_path}/frames/frame${frame}/sparse_gt_${resolution}/ \
        --frame_id $frame
done

conda deactivate
