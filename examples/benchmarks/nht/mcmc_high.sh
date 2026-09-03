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
# NHT MCMC benchmark with per-scene high primitive counts on Mip-NeRF 360.
# Per-scene caps match standard 3DGS-MCMC budgets. Mirrors the M360 portion of
# benchmarks/nht/benchmark_nht_high.sh (Table 7 of the NHT paper).

SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark_nht_high"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

MAX_STEPS=30000

NHT_ARGS=(
    --deferred_opt_feature_dim 64
    --deferred_opt_view_encoding_type sh
    --deferred_opt_center_ray_encoding
    --deferred_mlp_hidden_dim 128
    --deferred_mlp_num_layers 3
    --deferred_mlp_ema
    --deferred_features_lr 0.015
    --deferred_mlp_lr 0.0072
    --deferred_features_lr_decay_final 0.1
    --deferred_mlp_lr_decay_final 0.1
    --opacity_reg 0.02
    --scale_reg 0.01
    --color_refine_steps 3000
)

get_cap_max() {
    case "$1" in
        bonsai)   echo 1300000 ;;
        counter)  echo 1200000 ;;
        kitchen)  echo 1800000 ;;
        room)     echo 1500000 ;;
        garden)   echo 5200000 ;;
        bicycle)  echo 5900000 ;;
        stump)    echo 4750000 ;;
        treehill) echo 3500000 ;;
        flowers)  echo 3000000 ;;
        *)        echo 1000000 ;;
    esac
}

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    CAP_MAX=$(get_cap_max "$SCENE")
    echo "Running $SCENE (cap=$CAP_MAX)"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --max_steps $MAX_STEPS \
        --strategy.cap-max $CAP_MAX \
        --lpips_net vgg \
        --render_traj_path $RENDER_TRAJ_PATH \
        "${NHT_ARGS[@]}" \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py default --disable_viewer --data_factor $DATA_FACTOR \
            --max_steps $MAX_STEPS \
            --strategy.cap-max $CAP_MAX \
            --lpips_net vgg \
            --render_traj_path $RENDER_TRAJ_PATH \
            "${NHT_ARGS[@]}" \
            --data_dir $SCENE_DIR/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done


for SCENE in $SCENE_LIST;
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
