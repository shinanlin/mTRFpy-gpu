#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --output=log/compare_models.out
#SBATCH --error=log/compare_models.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-common

module purge
module load CUDA/11.4
#module load CUDA/11.4  # CUDA not available or not needed for CPU test
source /hpc/home/ns458/miniconda3/bin/activate mtrf-gpu

mkdir -p log
export PYTHONPATH=/hpc/group/coganlab/nanlinshi/mTRFpy-gpu:$PYTHONPATH
cd /hpc/group/coganlab/nanlinshi/mTRFpy-gpu

python tests/compare_models.py
