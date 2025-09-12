export FRAME=00000000

CUDA_VISIBLE_DEVICES=0 python -m simple_viewer \
        --ckpt /home/minhtran/Code/data/vocap/minh_2/textured_gaussians_results/frames/$FRAME/ckpt_29999.pt \
        --port 8081
