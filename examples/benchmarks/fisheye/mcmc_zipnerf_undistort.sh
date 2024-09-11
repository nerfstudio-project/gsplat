SCENE_DIR="data/zipnerf_undistort"
SCENE_LIST="berlin london nyc alameda"
DATA_FACTOR=2

RESULT_DIR="results/benchmark_mcmc_2M_zipnerf_undistort"
CAP_MAX=2000000

# RESULT_DIR="results/benchmark_mcmc_4M_zipnerf_undistort"
# CAP_MAX=4000000

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train and eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --opacity_reg 0.001 \
        --camera_model pinhole \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done
