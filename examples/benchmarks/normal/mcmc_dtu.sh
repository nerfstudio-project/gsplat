SCENE_DIR="data/DTU"
SCENE_LIST="scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122"
RENDER_TRAJ_PATH="ellipse"

RESULT_DIR="results/benchmark_dtu_mcmc_0.25M_normal"
CAP_MAX=250000

# RESULT_DIR="results/benchmark_dtu_mcmc_0.5M_normal"
# CAP_MAX=500000

# RESULT_DIR="results/benchmark_dtu_mcmc_1M_normal"
# CAP_MAX=1000000

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train and eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor 1 \
        --strategy.cap-max $CAP_MAX \
        --normal_consistency_loss \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done

echo "Summarizing results"
python benchmarks/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST --stage val
