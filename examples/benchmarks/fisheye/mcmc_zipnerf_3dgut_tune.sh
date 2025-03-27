SCENE_DIR="data/zipnerf"
SCENE_LIST="nyc alameda berlin london"
DATA_FACTOR=4
RENDER_TRAJ_PATH="ellipse"

# RESULT_DIR="results/benchmark_mcmc_1M_zipnerf_3dgut_perfect"
# CAP_MAX=1000000

RESULT_DIR="results/benchmark_mcmc_4M_zipnerf_3dgut_perfect_iscale0.5"
CAP_MAX=4000000

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=5 python simple_trainer.py mcmc  --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --with_eval3d --with_ut --perfect_camera --init_scale 0.5 \
        --strategy.cap-max $CAP_MAX \
        --opacity_reg 0.001 \
        --use_bilateral_grid \
        --render_traj_path $RENDER_TRAJ_PATH \
        --camera_model fisheye \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=5 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
            --with_eval3d --with_ut --perfect_camera --init_scale 0.5 \
            --strategy.cap-max $CAP_MAX \
            --opacity_reg 0.001 \
            --use_bilateral_grid \
            --render_traj_path $RENDER_TRAJ_PATH \
            --camera_model fisheye \
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