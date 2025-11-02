#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o ./slurm/output_liquid/%j.out
#SBATCH -J liquid_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=300000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --mail-user=your_email@domain
module purge
# Compiler and Python stack required to expose ML modules
module load gcc/12 anaconda/3/2023.03

# Activate your conda environment first
source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate LiquidReg_env2

# Then load CUDA 11.6 + cuDNN (cudnn 8.x) and PyTorch module matching it
module load cuda/11.6 || module load cuda/11.6.2 || module load cuda
module load cudnn/8.9.2 || module load cudnn/8.8.1 || true
module load pytorch/gpu-cuda-11.6/2.1.0 || module load pytorch/gpu-cuda-11.6/2.0.0 || true

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

srun -n 1 bash run_all.sh "$@"
