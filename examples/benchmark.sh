for SCENE in bicycle bonsai counter garden kitchen room stump;
do
    echo "Running $SCENE"

    # train without eval
    python simple_trainer.py --eval_steps -1 --exit_once_done \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir results/benchmark/$SCENE/

    # run eval and render
    for CKPT in results/benchmark/$SCENE/ckpts/*;
    do
        python simple_trainer.py --exit_once_done \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir results/benchmark/$SCENE/ \
            --ckpt $CKPT
    done

    echo "=== Eval Stats ==="

    for STATS in results/benchmark/$SCENE/stats/val*;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in results/benchmark/$SCENE/stats/train*;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

done