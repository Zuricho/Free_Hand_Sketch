#!/bin/bash
#SBATCH --job-name=model_cnn
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH -x gpu01,gpu03,gpu05,gpu13,gpu19,gpu21
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=task_file/%x_%j.out
#SBATCH --error=task_file/%x_%j.err

module purge
module load miniconda3
source activate torch

nvidia-smi

python model_cnn.py

