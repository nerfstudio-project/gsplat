# #!/bin/bash

# # Function to run the tile_trainer.py script
# run_tile_trainer() {
#     iterations=$1
#     num_points=$2
#     gpu=$3
#     echo "Running on GPU $gpu: iterations=$iterations, num_points=$num_points"
#     CUDA_VISIBLE_DEVICES=$gpu python examples/tile_trainer.py --weights 0.25 0.25 0.25 0.25 --iterations $iterations --num_points $num_points
# }

# # Function to find the next available GPU
# find_available_gpu() {
#     for gpu in 0 1 2; do
#         if [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu) -lt 80 ]; then
#             echo $gpu
#             return
#         fi
#     done
#     echo "-1"
# }

# # Generate all combinations of iterations and num_points
# combinations=()
# for iterations in $(seq 500 500 9500); do
#     for num_points in $(seq 100 100 4900); do
#         combinations+=("$iterations $num_points")
#     done
# done

# # Process all combinations
# for combination in "${combinations[@]}"; do
#     read iterations num_points <<< "$combination"
    
#     # Find an available GPU
#     while true; do
#         gpu=$(find_available_gpu)
#         if [ "$gpu" != "-1" ]; then
#             run_tile_trainer $iterations $num_points $gpu &
#             sleep 1  # Short delay to allow the job to start
#             break
#         fi
#         sleep 5  # Wait before checking GPUs again
#     done
# done

# # Wait for all background processes to finish
# wait

# echo "All jobs completed."




#!/bin/bash

# Initialize an array to track GPU processes
gpu_pids=()

# Set the ranges for iterations and num_points
for iterations in $(seq 500 500 1500); do
  for num_points in $(seq 100 100 300); do
    # Calculate the GPU to assign (cycles through 0, 1, 2 for 3 GPUs)
    gpu=$(( (counter % 3) ))

    # Wait for the GPU to finish its previous task if it's still running
    if [ ! -z "${gpu_pids[$gpu]}" ]; then
      wait ${gpu_pids[$gpu]}
    fi

    # Run the tile_trainer.py command and store the process ID (PID)
    CUDA_VISIBLE_DEVICES=$gpu python examples/tile_trainer.py --random --weights 0.4 0.1 0.1 0.4 \
      --iterations $iterations --num_points $num_points &

    # Store the PID of the background process
    gpu_pids[$gpu]=$!

    # Increment the counter
    counter=$((counter + 1))

    # Small delay to avoid overwhelming the system
    sleep 0.2
  done
done

# Wait for all remaining GPU jobs to finish
wait
