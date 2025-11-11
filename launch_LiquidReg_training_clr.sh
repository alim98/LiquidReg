#!/bin/bash
# Auto-restart launcher for train_LiquidReg_clr.sh

log_file="slurm/launch_orchestrator.log"
exec > >(tee -a "$log_file") 2>&1

echo "=========================================="
echo "LiquidReg Multi-GPU Training Launcher"
echo "Timestamp: $(date)"
echo "=========================================="

job_script="train_LiquidReg_clr.sh"

# Training arguments
CONFIG="configs/default_4gpu.yaml"
TRAIN_PAIRS="data/OASIS_train/pairs_train.csv"
VAL_PAIRS="data/OASIS_val/pairs_val.csv"
WORK_DIR="runs/liquidreg_4gpu"

TRAIN_ARGS="--config $CONFIG --train_pairs $TRAIN_PAIRS --val_pairs $VAL_PAIRS --work_dir $WORK_DIR --ddp"

# First submission
output=$(sbatch "$job_script" $TRAIN_ARGS)
echo "$output"
job_id=$(echo "$output" | awk '{print $4}')
echo "Submitted Training Job ID: $job_id with 4 GPUs (DDP enabled)"

# Trap for clean exit
cleanup() {
    echo "Received signal, cleaning up..."
    echo "Launcher stopped at $(date)"
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "Starting monitoring loop..."
while true; do
    sleep 10
    job_status=$(sacct -j "$job_id" --format=State --noheader | head -n 1 | tr -d '[:space:]')

    while [[ $job_status == "RUNNING" || $job_status == "PENDING" ]]; do
        sleep 60
        job_status=$(sacct -j "$job_id" --format=State --noheader | head -n 1 | tr -d '[:space:]')
    done

    if [[ $job_status == "TIMEOUT" || $job_status == "COMPLETED" ]]; then
        echo "Job $job_id ended with $job_status. Restarting..."
        
        # Find latest checkpoint for resume
        LATEST_CKPT=$(find "$WORK_DIR" -name "*.pth" -type f | grep -E "(best|last|final)" | sort -V | tail -1)
        if [[ -n "$LATEST_CKPT" ]]; then
            echo "Resuming from: $LATEST_CKPT"
            TRAIN_ARGS="$TRAIN_ARGS --resume $LATEST_CKPT"
        fi
        
        output=$(sbatch "$job_script" $TRAIN_ARGS)
        echo "$output"
        job_id=$(echo "$output" | awk '{print $4}')
        echo "Restarted Job ID: $job_id with 4 GPUs (DDP enabled)"
    else
        echo "Job $job_id finished with status: $job_status"
        break
    fi
done
