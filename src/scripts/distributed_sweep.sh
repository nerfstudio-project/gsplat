SCRIPT="src/scripts/distr_wb_ppo_lr_predictor.py"
PROJECT_NAME="ppo_lr_predictor_distr_entropy_sched_debug"
ENTITY="rl_gsplat"
COMMAND="python $SCRIPT --sweep --project_name $PROJECT_NAME --entity $ENTITY"
$COMMAND
SWEEP_ID=$(cat ./temp/sweep_id.txt)

echo "================================================"
echo "Sweep ID: $SWEEP_ID"
echo "================================================"

for GPU_ID in 0 1 2 3; do
    echo "GPU ID: $GPU_ID"
    python $SCRIPT --sweep --sweep_id $SWEEP_ID --project_name $PROJECT_NAME --entity $ENTITY --gpu_id $GPU_ID &
done
