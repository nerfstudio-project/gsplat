SCENE_DIR="data/DeepBlending"

SCENE_LIST="drjohnson playroom"

# # 0.36M GSs
# RESULT_DIR="results/benchmark_db_mcmc_0_36M_png_compression"
# CAP_MAX=360000

# # 0.49M GSs
# RESULT_DIR="results/benchmark_db_mcmc_0_49M_png_compression"
# CAP_MAX=490000

# 1M GSs
RESULT_DIR="results/benchmark_db_mcmc_1M_png_compression"
CAP_MAX=1000000

# # 4M GSs
# RESULT_DIR="results/benchmark_db_mcmc_4M_png_compression"
# CAP_MAX=4000000

# Override default values if provided as arguments
[ ! -z "$1" ] && RESULT_DIR="$1"
[ ! -z "$2" ] && CAP_MAX="$2"

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --eval_steps -1 --disable_viewer --data_factor 1 \
        --strategy.cap-max $CAP_MAX \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # eval: use vgg for lpips to align with other benchmarks
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor 1 \
        --strategy.cap-max $CAP_MAX \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --lpips_net vgg \
        --compression png \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_29999_rank0.pt
done

# Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
else
    echo "zip command not found, skipping zipping"
fi