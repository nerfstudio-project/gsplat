SCENE_DIR="data/zipnerf_undistorted"
SCENE_LIST="berlin london nyc alameda"
DATA_FACTOR=4
RENDER_TRAJ_PATH="ellipse"

RESULT_DIR="results/benchmark_mcmc_2M_zipnerf_undistorted"
CAP_MAX=2000000

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --opacity_reg 0.001 \
        --init_scale 0.5 \
        --post_processing bilateral_grid \
        --render_traj_path $RENDER_TRAJ_PATH \
        --camera_model pinhole \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
            --strategy.cap-max $CAP_MAX \
            --opacity_reg 0.001 \
            --init_scale 0.5 \
            --post_processing bilateral_grid \
            --render_traj_path $RENDER_TRAJ_PATH \
            --camera_model pinhole \
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