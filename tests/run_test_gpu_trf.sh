#!/bin/bash
#SBATCH --job-name=test_gpu_trf
#SBATCH --output=log/test_gpu_trf.out
#SBATCH --error=log/test_gpu_trf.err
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2

# DCC GPU manual: https://dcc.duke.edu/dcc/partitions/?h=gpu#gpus

module purge
module load CUDA/11.4
#module load CUDA/11.4  # CUDA not available or not needed for CPU test
source /hpc/home/ns458/miniconda3/bin/activate mtrf-gpu

mkdir -p log
export PYTHONPATH=/hpc/group/coganlab/nanlinshi/mTRFpy-gpu:$PYTHONPATH
cd /hpc/group/coganlab/nanlinshi/mTRFpy-gpu
python tests/test_gpu_trf.py