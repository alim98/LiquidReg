#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o ./slurm/output_liquid/%j.out
#SBATCH -J liquid_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=300000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --mail-user=your_email@domain
module purge
module load anaconda/3/2023.03
module load cuda/11.6
module load gcc/11
module load openmpi_gpu/4.1

# Activate your conda environment first
source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate LiquidReg_env2

# Load PyTorch distributed module (must load after anaconda)
module load pytorch-distributed/gpu-cuda-11.6/2.1.0

# Quick sanity check
python - <<'PY'
try:
    import torch
    print("[SANITY] torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())
except Exception as e:
    import sys
    print("[SANITY] torch import failed:", e, file=sys.stderr)
    sys.exit(1)
PY

# Prevent picking up ~/.local site-packages
export PYTHONNOUSERSITE=1

# GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Fix scipy/GCC library compatibility
export LD_LIBRARY_PATH=/mpcdf/soft/SLE_15/packages/x86_64/gcc/12.2.0/lib64:$LD_LIBRARY_PATH

NP=${SLURM_NTASKS_PER_NODE:-4}
echo "Running with $NP GPUs using torch.distributed.run"
python -m torch.distributed.run --standalone --nproc_per_node="$NP" scripts/train.py "$@"
