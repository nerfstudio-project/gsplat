#!/bin/bash

pixel_thresholds=(0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 1.0 2.0)
# pixel_thresholds=(0.0)

# for pixel_threshold in "${pixel_thresholds[@]}"; do
#     python scripts/culling/evaluate_static_size_based_culling.py --pixel-threshold $pixel_threshold --no-save-images --lpips  
# done

for pixel_threshold in "${pixel_thresholds[@]}"; do
    python scripts/culling/evaluate_static_area_based_culling.py --area-threshold $pixel_threshold --save-images --lpips  
done