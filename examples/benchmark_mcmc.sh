RESULT_DIR=results/2dgs_mcmc_sfm

# for SCENE in bicycle bonsai counter garden kitchen room stump;
for SCENE in garden treehill bonsai counter kitchen room bicycle stump flowers;
do
    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ] || [ "$SCENE" = "treehill" ] || [ "$SCENE" = "flowers" ]; then
        DATA_FACTOR=4
        DIST_LAMBDA=100
    else
        DATA_FACTOR=2
        DIST_LAMBDA=1000
    fi
    
    if [ "$SCENE" = "bonsai" ]; then
        CAP_MAX=1300000
    elif [ "$SCENE" = "counter" ]; then
        CAP_MAX=1200000
    elif [ "$SCENE" = "kitchen" ]; then
        CAP_MAX=1800000
    elif [ "$SCENE" = "room" ]; then
        CAP_MAX=1500000
    else
        CAP_MAX=3000000
    fi

    echo "Running $SCENE"

    # train without eval
    python simple_trainer_mcmc.py --disable_viewer --data_factor $DATA_FACTOR \
        --model_type 2dgs \
        --dist_lambda $DIST_LAMBDA \
        --init_type sfm \
        --cap_max $CAP_MAX \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # # run eval and render
    # for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    # do
    #     python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
    #         --data_dir data/360_v2/$SCENE/ \
    #         --result_dir $RESULT_DIR/$SCENE/ \
    #         --ckpt $CKPT
    # done
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