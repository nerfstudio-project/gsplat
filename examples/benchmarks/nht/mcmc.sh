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
# NHT unified MCMC benchmark on Mip-NeRF 360 (1M primitives, 30k steps, 64-D features).
# Matches the unified config in benchmarks/nht/benchmark_nht.sh (Table 2 of the NHT paper).
# LPIPS is reported with VGG (normalize=True), matching the NHT paper. Pass `--lpips_net alex`
# to switch backbones; the trainer always normalizes inputs.

SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark_nht_mcmc_1M"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

CAP_MAX=1000000
MAX_STEPS=30000
FEATURE_DIM=64

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --max_steps $MAX_STEPS \
        --strategy.cap-max $CAP_MAX \
        --deferred_opt_feature_dim $FEATURE_DIM \
        --ssim_lambda 0.1 \
        --lpips_net vgg \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py default --disable_viewer --data_factor $DATA_FACTOR \
            --max_steps $MAX_STEPS \
            --strategy.cap-max $CAP_MAX \
            --deferred_opt_feature_dim $FEATURE_DIM \
            --ssim_lambda 0.1 \
            --lpips_net vgg \
            --render_traj_path $RENDER_TRAJ_PATH \
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
