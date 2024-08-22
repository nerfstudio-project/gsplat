RESULT_DIR=results/gsplat-packed

for SCENE in bicycle bonsai counter garden kitchen room stump;
do
    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ]; then
        DATA_FACTOR=4
    else
        DATA_FACTOR=2
    fi

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_2dgs.py --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir data/360_v2/$SCENE/ \
        --model_type 2dgs \
        --result_dir $RESULT_DIR/$SCENE/ \
        --packed

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python simple_trainer_2dgs.py --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir data/360_v2/$SCENE/ \
            --model_type 2dgs \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT \
            --packed

    done
done


for SCENE in bicycle bonsai counter garden kitchen room stump;
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