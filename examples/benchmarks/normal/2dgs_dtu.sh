SCENE_DIR="data/DTU"
SCENE_LIST="scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122"

RESULT_DIR="results/benchmark_dtu_2dgs"

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train and eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_2dgs.py --disable_viewer --data_factor 1 \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done

echo "Summarizing results"
python benchmarks/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST --stage val
