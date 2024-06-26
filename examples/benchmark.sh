RESULT_DIR=results/benchmark

for SCENE in bicycle bonsai counter garden kitchen room stump;
do
    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ]; then
        DATA_FACTOR=4
    else
        DATA_FACTOR=2
    fi

    echo "Running $SCENE"

    # train without eval
    python simple_trainer.py --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in results/benchmark/$SCENE/ckpts/*;
    do
        python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done


for SCENE in bicycle bonsai counter garden kitchen room stump;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done