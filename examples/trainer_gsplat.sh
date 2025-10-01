export FRAME=00000000
echo "Training frame $FRAME"

conda init
conda activate gsplat

echo "RGBA without alpha loss, random background, MCMC strategy..."
CUDA_VISIBLE_DEVICES=0 python simple_trainer_rgba.py mcmc \
    --data_dir /home/minhtran/Code/data/vocap/minh_2/frames/$FRAME/train/rgba \
    --data_factor 1 \
    --result_dir /home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/$FRAME/rgba/mcmc_random_bkgd \
    --test_every 6 \
    --no_normalize_world_space \
    --random_bkgd \
    --disable_viewer

echo "RGBA with alpha loss, MCMC strategy..."
CUDA_VISIBLE_DEVICES=0 python simple_trainer_rgba.py mcmc \
    --data_dir /home/minhtran/Code/data/vocap/minh_2/frames/$FRAME/train/rgba \
    --data_factor 1 \
    --result_dir /home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/$FRAME/rgba/mcmc_alpha_loss \
    --test_every 6 \
    --no_normalize_world_space \
    --disable_viewer

echo "RGBA with alpha loss, default strategy..."
CUDA_VISIBLE_DEVICES=0 python simple_trainer_rgba.py default \
    --data_dir /home/minhtran/Code/data/vocap/minh_2/frames/$FRAME/train/rgba \
    --data_factor 1 \
    --result_dir /home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/$FRAME/rgba/default_alpha_loss \
    --test_every 6 \
    --no_normalize_world_space \
    --disable_viewer

echo "RGBA without alpha loss, random background, default strategy..."
CUDA_VISIBLE_DEVICES=0 python simple_trainer_rgba.py default \
    --data_dir /home/minhtran/Code/data/vocap/minh_2/frames/$FRAME/train/rgba \
    --data_factor 1 \
    --result_dir /home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/$FRAME/rgba/default_random_bkgd \
    --test_every 6 \
    --no_normalize_world_space \
    --random_bkgd \
    --disable_viewer

echo "RGB, default strategy..."
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir /home/minhtran/Code/data/vocap/minh_2/frames/$FRAME/train/rgb \
    --data_factor 1 \
    --result_dir /home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/$FRAME/rgb/default \
    --test_every 6 \
    --no_normalize_world_space \
    --disable_viewer

echo "RGB, MCMC strategy..."
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc \
    --data_dir /home/minhtran/Code/data/vocap/minh_2/frames/$FRAME/train/rgb \
    --data_factor 1 \
    --result_dir /home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/$FRAME/rgb/mcmc \
    --test_every 6 \
    --no_normalize_world_space \
    --disable_viewer


# CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
#     --ckpt /home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/$FRAME/rgb/ckpts/ckpt_29999_rank0.pt \
#     --port 8080