#!/bin/bash

# Evaluate dynamic distance culling quality

pixel_thresholds=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 1.0 2.0)

for pixel_threshold in "${pixel_thresholds[@]}"; do
    python scripts/ablation/evaluate_quality_of_dynamic_culling.py --pixel-threshold $pixel_threshold --no-save-images #--lpips  
done