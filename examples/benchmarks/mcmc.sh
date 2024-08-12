RESULT_DIR=results/benchmark_mcmc_1M
CAP_MAX=1000000

# for SCENE in bicycle bonsai counter garden kitchen room stump;
# do
#     if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ]; then
#         DATA_FACTOR=4
#     else
#         DATA_FACTOR=2
#     fi

#     echo "Running $SCENE"

#     # train without eval
#     CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
#         --strategy.cap-max $CAP_MAX \
#         --data_dir data/360_v2/$SCENE/ \
#         --result_dir $RESULT_DIR/$SCENE/

#     # run eval and render
#     for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
#     do
#         CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
#             --strategy.cap-max $CAP_MAX \
#             --data_dir data/360_v2/$SCENE/ \
#             --result_dir $RESULT_DIR/$SCENE/ \
#             --ckpt $CKPT
#     done
# done


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