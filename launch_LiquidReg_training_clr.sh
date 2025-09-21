#!/bin/bash
# Auto-restart launcher for train_LiquidReg_clr.sh

log_file="slurm/launch_orchestrator.log"
exec > >(tee -a "$log_file") 2>&1

echo "=========================================="
echo "Orchestrator Launcher Started"
echo "Timestamp: $(date)"
echo "=========================================="

job_script="train_LiquidReg_clr.sh"

# First submission
output=$(sbatch "$job_script")
echo "$output"
job_id=$(echo "$output" | awk '{print $4}')
echo "Submitted Orchestrator Job ID: $job_id"

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
        output=$(sbatch "$job_script")
        echo "$output"
        job_id=$(echo "$output" | awk '{print $4}')
        echo "Restarted Job ID: $job_id"
    else
        echo "Job $job_id finished with status: $job_status"
        break
    fi
done
