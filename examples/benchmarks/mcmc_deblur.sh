SCENE_DIR="data/deblur_dataset/real_defocus_blur"
SCENE_LIST="defocuscake defocuscaps defocuscisco defocuscoral defocuscupcake defocuscups defocusdaisy defocussausage defocusseal defocustools"
# RETRY_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
# for RETRY in $RETRY_LIST;
    # SCENE="defocuscaps"
        # --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_29999_rank0.pt

DATA_FACTOR=4
RENDER_TRAJ_PATH="spiral"

RESULT_DIR="results/benchmark_mcmc_deblur_cheating3"
CAP_MAX=250000

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train and eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --blur_opt \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE
done

# Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
else
    echo "zip command not found, skipping zipping"
fi
