#!/bin/bash

data_dir="../../playaround_gaussian_platting/dataset_cvpr/smerf/nyc/"
# Define hyperparameters
iresnet_lrs=(1e-8)
strategy_cap_maxs=(2000000 3000000)
strategy_min_opacities=(0.0001 0.0005)
step_ranges=(
    "4000 14000"
    "6000 16000"
    "8000 18000"
    "4000 30000"
)
eval_steps=(1 10000 20000 30000 40000)

# Trap for graceful exit
trap 'echo "Stopping..."; kill $(jobs -p); exit' SIGINT

# Create a directory for GPU lock files
lock_dir="/tmp/gpu_locks"
mkdir -p "$lock_dir"

# Function to acquire a lock on a GPU
acquire_gpu_lock() {
    while true; do
        for gpu_id in {0..7}; do
            if mkdir "$lock_dir/gpu_$gpu_id" 2>/dev/null; then
                echo $gpu_id
                return
            fi
        done
        sleep 1
    done
}

# Function to release a GPU lock
release_gpu_lock() {
    local gpu_id=$1
    rmdir "$lock_dir/gpu_$gpu_id"
}

# Iterate through combinations of hyperparameters
for iresnet_lr in "${iresnet_lrs[@]}"; do
    for strategy_cap_max in "${strategy_cap_maxs[@]}"; do
        for strategy_min_opacity in "${strategy_min_opacities[@]}"; do
            for step_range in "${step_ranges[@]}"; do
                # Extract step2start and step2end
                step2start=$(echo $step_range | cut -d' ' -f1)
                step2end=$(echo $step_range | cut -d' ' -f2)

                # Acquire a GPU lock
                gpu_id=$(acquire_gpu_lock)

                # Define result directory
                result_dir="results/zipnerf/scale3.5_2/nyc_lr${iresnet_lr}_num${strategy_cap_max}_${step2start}-${step2end}"

                # Log start of the job
                echo "Starting job: iresnet_lr=${iresnet_lr}, cap-max=${strategy_cap_max}, min_opacity=${strategy_min_opacity}, steps=${step2start}-${step2end} on GPU=${gpu_id}"

                # Run the command
                (
                    CUDA_VISIBLE_DEVICES=$gpu_id python simple_trainer_single.py mcmc \
                        --disable_viewer \
                        --data_factor 1 \
                        --steps_scaler 1 \
                        --packed \
                        --render_traj_path ellipse \
                        --antialiased \
                        --use_bilateral_grid \
                        --data_dir "$data_dir" \
                        --result_dir "$result_dir" \
                        --eval_steps "${eval_steps[@]}" \
                        --iresnet_lr "$iresnet_lr" \
                        --step2start "$step2start" \
                        --step2end "$step2end" \
                        --strategy.cap-max "$strategy_cap_max" \
                        --strategy.min_opacity "$strategy_min_opacity" \
                        --control_point_sample_scale 16 \
                        --control_point_sample_scale_eval 4 \
                        --scale_x 3.5 \
                        --scale_y 2. \
                        --scale_x_eval 3.5 \
                        --scale_y_eval 2. \
                        --when2addmask 0 \
                        --init_scale 0.3 \
                        --update_min_opacity 0.001 \
                        --max_steps 40000
                    # Release the GPU lock after the job is finished
                    release_gpu_lock $gpu_id
                ) &

            done
        done
    done
done

# Wait for all jobs to complete
wait

echo "All jobs completed!"

