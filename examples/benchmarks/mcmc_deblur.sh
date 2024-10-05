SCENE_DIR="data/deblur_dataset/real_defocus_blur"
SCENE_LIST="defocuscake defocuscaps defocuscisco defocuscoral defocuscupcake defocuscups defocusdaisy defocussausage defocusseal defocustools"

DATA_FACTOR=4
RENDER_TRAJ_PATH="spiral"

RESULT_DIR="results/benchmark_mcmc_deblur"
CAP_MAX=100000

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train and eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --blur_opt \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done
