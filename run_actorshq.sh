DATA_DIR="/main/rajrup/Dropbox/Project/GsplatStream/gsplat/data/Actor01/Sequence1/"
SCENE_DIR="$DATA_DIR/0"
RESULT_DIR="results/actorshq/0/"
SCENE="resolution_4" # bicycle stump bonsai counter kitchen room treehill flowers
RENDER_TRAJ_PATH="interp"
DATA_FACTOR=1

echo "Running $SCENE"

# train without eval
CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py default --eval_steps -1 --data_factor $DATA_FACTOR \
    --render_traj_path $RENDER_TRAJ_PATH \
    --data_dir $SCENE_DIR/$SCENE/ \
    --result_dir $RESULT_DIR/$SCENE/

# # run eval and render
# for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
# do
#     # run eval and render
#     CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py default --disable_viewer --data_factor $DATA_FACTOR \
#         --render_traj_path $RENDER_TRAJ_PATH \
#         --data_dir $SCENE_DIR/$SCENE/ \
#         --result_dir $RESULT_DIR/$SCENE/ \
#         --ckpt $CKPT
# done

# # run eval and view
# CKPT=$RESULT_DIR/$SCENE/ckpts/ckpt_29999_rank0.pt
# CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py default --data_factor $DATA_FACTOR \
#     --render_traj_path $RENDER_TRAJ_PATH \
#     --data_dir $SCENE_DIR/$SCENE/ \
#     --result_dir $RESULT_DIR/$SCENE/ \
#     --ckpt $CKPT


# echo "=== Eval Stats ==="

# for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
# do  
#     echo $STATS
#     cat $STATS; 
#     echo
# done

# echo "=== Train Stats ==="

# for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
# do  
#     echo $STATS
#     cat $STATS; 
#     echo
# done