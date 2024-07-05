# for SCENE in bicycle bonsai counter garden kitchen room stump;
for SCENE in garden treehill;
do
    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ] || [ "$SCENE" = "treehill" ] || [ "$SCENE" = "flowers" ]; then
        DATA_FACTOR=4
    else
        DATA_FACTOR=2
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

    python simple_trainer_mcmc.py --disable_viewer --data_factor $DATA_FACTOR \
        --model_type 2dgs \
        --init_type sfm \
        --eval_steps 1000 7000 15000 30000 \
        --cap_max $CAP_MAX \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir results/2dgs_mcmc_sfm/$SCENE/

done
