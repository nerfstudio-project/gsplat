#!/bin/bash

actor_name=Actor08
seq_name=Sequence1
resolution=4
frame_id=0

bash export_actorshq_to_colmap.sh # Change parameters to export_actorshq_to_colmap.sh to export the correct resolution

cd /bigdata2/rajrup/datasets/actorshq/colmap/${actor_name}/${seq_name}/frames/frame${frame_id}/
colmap feature_extractor \
    --database_path ./database_${resolution}.db \
    --image_path ./images_${resolution} \
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.peak_threshold 0.001 \
    --SiftExtraction.edge_threshold 80

colmap exhaustive_matcher \
    --database_path ./database_${resolution}.db \
    --SiftMatching.guided_matching 1

mkdir -p ./triangulated_${resolution}/
colmap point_triangulator \
    --database_path ./database_${resolution}.db \
    --image_path ./images_${resolution} \
    --input_path ./sparse_gt_${resolution} \
    --output_path ./triangulated_${resolution}

colmap model_converter \
    --input_path ./triangulated_${resolution} \
    --output_path ./triangulated_${resolution} \
    --output_type TXT