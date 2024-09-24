SCENE_DIR="data/360_v2"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"

RESULT_DIR="results/benchmark_normal"
RENDER_TRAJ_PATH="ellipse"

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --normal_consistency_loss \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done

echo "Summarizing results"
python benchmarks/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST --stage val
