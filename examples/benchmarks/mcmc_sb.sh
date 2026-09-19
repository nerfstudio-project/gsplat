#!/bin/sh

# MCMC with Spherical Beta appearance. Identical to mcmc.sh apart from
# --sh_degree/--sb_number, so the numbers are directly comparable.
#
# Spherical Beta is from "Deformable Beta Splatting" (SIGGRAPH 2025) by Rong Liu,
# Dylan Sun, Meida Chen, Yue Wang and Andrew Feng:
# https://rongliu-leo.github.io/beta-splatting/
#
# The default here is the Beta Splatting configuration: SH degree 0 supplies the
# diffuse base and two beta lobes model the speculars, for 15 appearance values
# per Gaussian against the 48 that SH degree 3 needs. Override SH_DEGREE and
# SB_NUMBER to explore other combinations, e.g. SH_DEGREE=3 SB_NUMBER=2 to model
# low- and high-frequency view dependence together.

SDIR=$(cd -- "$(dirname "$0")" && pwd -P)

EXAMPLES_DIR=$SDIR/..
SCENE_DIR="data/360_v2"
SH_DEGREE=${SH_DEGREE:-0}
SB_NUMBER=${SB_NUMBER:-2}
RESULT_DIR="results/benchmark_mcmc_sb_1M_sh${SH_DEGREE}_sb${SB_NUMBER}"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

CAP_MAX=1000000

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python $EXAMPLES_DIR/simple_trainer.py mcmc --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --render_traj_path $RENDER_TRAJ_PATH \
        --sh_degree $SH_DEGREE \
        --sb_number $SB_NUMBER \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python $EXAMPLES_DIR/simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
            --strategy.cap-max $CAP_MAX \
            --render_traj_path $RENDER_TRAJ_PATH \
            --sh_degree $SH_DEGREE \
            --sb_number $SB_NUMBER \
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
