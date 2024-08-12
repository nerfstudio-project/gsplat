SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark_mcmc_1M_png_compression"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers

CAP_MAX=1000000

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    # compress and eval at the --eval_steps 
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --eval_steps 30000 --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --compression png

done

# see if zip command is available
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/compression/summarize_stats.py 
else
    echo "zip command not found, skipping zipping"
fi