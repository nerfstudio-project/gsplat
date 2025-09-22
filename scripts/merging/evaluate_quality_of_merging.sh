#!/bin/bash

# pixel_thresholds=(0.0 0.01 0.05 0.1 0.5 1.0 2.0)
pixel_thresholds=(0.01 0.1 0.5 1.0 2.0)
# Sort in descending order
pixel_thresholds=($(for i in "${pixel_thresholds[@]}"; do echo $i; done | sort -nr))

# echo "Starting Gaussian merging quality evaluation..."
# echo "Pixel thresholds to test: ${pixel_thresholds[@]}"

# for pixel_threshold in "${pixel_thresholds[@]}"; do
#     echo ""
#     echo "======================================================"
#     echo "Running evaluation with pixel threshold: $pixel_threshold"
#     echo "======================================================"
#     python scripts/merging/evaluate_static_merging.py --pixel-threshold $pixel_threshold --save-images #--lpips
#     # if [ $pixel_threshold == 0.01 ] || [ $pixel_threshold == 2.0 ]; then
#     #     python scripts/merging/evaluate_static_merging.py --pixel-threshold $pixel_threshold --save-images #--lpips
#     # else
#     #     python scripts/merging/evaluate_static_merging.py --pixel-threshold $pixel_threshold --no-save-images #--lpips
#     # fi 
# done

# echo ""
# echo "======================================================"
# echo "All merging evaluations completed!"
# echo "======================================================"


displacement=0.1
inward_angle=0.05

echo "Starting viewpoint robustness evaluation for Gaussian merging..."
echo "Pixel thresholds to test: ${pixel_thresholds[@]}"
echo "Displacement: $displacement"
echo "Inward angle: $inward_angle"

for pixel_threshold in "${pixel_thresholds[@]}"; do
    echo ""
    echo "======================================================"
    echo "Running viewpoint robustness with pixel threshold: $pixel_threshold"
    echo "======================================================"
    python scripts/merging/evaluate_static_merging_surr_views.py --pixel-threshold $pixel_threshold --save-images --displacement $displacement --inward-angle $inward_angle
done

echo ""
echo "======================================================"
echo "All viewpoint robustness evaluations completed!"
echo "======================================================"