#!/bin/bash
#SBATCH --job-name=train_sdxl_loss_2_2_with_bptt_scale
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --account=stf                                                                                                             
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00-5:00:00
#SBATCH --signal=SIGUSR1@90

# Activate your conda
source $(conda info --base)/etc/profile.d/conda.sh                                                                                                                                                   
conda activate /gpfs/projects/shlneuroai/caleb/miniconda3/envs/sdib

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --save_dir results_sdxl --cfg configs/sdxl22.yaml