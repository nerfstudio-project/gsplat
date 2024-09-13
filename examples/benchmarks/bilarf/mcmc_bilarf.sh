SCENE_DIR="data/bilarf/bilarf_data/testscenes"
SCENE_LIST="chinesearch lionpavilion pondbike statue strat building"

# SCENE_DIR="data/bilarf/bilarf_data/editscenes"
# SCENE_LIST="rawnerf_windowlegovary rawnerf_sharpshadow scibldg"

RESULT_DIR="results/benchmark_bilarf"
CAP_MAX=1000000
DATA_FACTOR=4
RENDER_TRAJ_PATH="spiral"

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --use_bilateral_grid \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done
