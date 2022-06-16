#!/bin/bash
#SBATCH --job-name=data_transform
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --output=task_file/%x_%j.out
#SBATCH --error=task_file/%x_%j.err


python data_transform.py

