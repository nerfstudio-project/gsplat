# SCENE_DIR="data/zipnerf/undistort"
# RESULT_DIR="results/benchmark_zipnerf/undistort"
# CAMERA_MODEL="pinhole"
SCENE_DIR="data/zipnerf/fisheye"
RESULT_DIR="results/benchmark_zipnerf/fisheye"
CAMERA_MODEL="fisheye"
SCENE_LIST="berlin" # alameda
RENDER_TRAJ_PATH="interp"

CAP_MAX=2000000
DATA_FACTOR=4

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --opacity_reg 0.001 \
        --camera_model $CAMERA_MODEL \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --ckpt "results/benchmark_zipnerf/undistort/$SCENE/ckpts/ckpt_29999_rank0.pt"

    # run eval and render
    # for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    # do
    #     CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
    #         --strategy.cap-max $CAP_MAX \
    #         --camera_model $CAMERA_MODEL \
    #         --render_traj_path $RENDER_TRAJ_PATH \
    #         --data_dir $SCENE_DIR/$SCENE/ \
    #         --result_dir $RESULT_DIR/$SCENE/ \
    #         --ckpt $CKPT
    # done
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