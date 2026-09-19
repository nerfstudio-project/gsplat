#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NHT split-strategy benchmark on Mip-NeRF 360 (paper configuration).
# Indoor and outdoor scenes use different cap-max, max-steps, and view encoding.
# Mirrors the M360 portion of benchmarks/nht/benchmark_nht_split.sh (Table 1 of the NHT paper).
#
#   M360 outdoor: 5M cap, 25k steps, per-ray view encoding
#   M360 indoor:  2M cap, 45k steps, center-ray view encoding

SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark_nht_split"
M360_INDOOR="bonsai counter kitchen room"
M360_OUTDOOR="garden bicycle stump" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

COMMON_ARGS=(
    --disable_viewer
    --render_traj_path $RENDER_TRAJ_PATH
    --ssim_lambda 0.1
    --lpips_net vgg
)

run_scene() {
    local scene=$1 factor=$2 cap_max=$3 max_steps=$4 group=$5
    shift 5
    local extra_args=("$@")
    local result_dir="$RESULT_DIR/$scene"

    echo ">>> [$group] Training $scene (factor=$factor, cap=$cap_max, steps=$max_steps)"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py default --eval_steps -1 \
        "${COMMON_ARGS[@]}" \
        --data_factor $factor \
        --max_steps $max_steps \
        --strategy.cap-max $cap_max \
        "${extra_args[@]}" \
        --data_dir $SCENE_DIR/$scene/ \
        --result_dir $result_dir/

    # run eval and render
    for CKPT in $result_dir/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py default \
            "${COMMON_ARGS[@]}" \
            --data_factor $factor \
            --max_steps $max_steps \
            --strategy.cap-max $cap_max \
            "${extra_args[@]}" \
            --data_dir $SCENE_DIR/$scene/ \
            --result_dir $result_dir/ \
            --ckpt $CKPT
    done
}

# Outdoor: per-ray (no --deferred_opt_center_ray_encoding), 5M cap, 25k steps, factor=4
for SCENE in $M360_OUTDOOR;
do
    run_scene "$SCENE" 4 5000000 25000 "m360-outdoor"
done

# Indoor: center-ray encoding, 2M cap, 45k steps, factor=2
for SCENE in $M360_INDOOR;
do
    run_scene "$SCENE" 2 2000000 45000 "m360-indoor" --deferred_opt_center_ray_encoding
done


for SCENE in $M360_OUTDOOR $M360_INDOOR;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
    do
        echo $STATS
        cat $STATS;
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do
        echo $STATS
        cat $STATS;
        echo
    done
done
