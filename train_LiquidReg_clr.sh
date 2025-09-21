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
module load intel/21.2.0 impi/2021.2

# Activate the conda environment
source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate cephclr

srun bash run_all.sh "$@"
