#!/bin/bash
#SBATCH --job-name=flux_retrain_lora
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --account=stf                                                                                                             
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00-15:00:00
#SBATCH --signal=SIGUSR1@90

# Activate your conda
source $(conda info --base)/etc/profile.d/conda.sh                                                                                                                                                   
conda activate /gpfs/projects/shlneuroai/caleb/miniconda3/envs/sdib
# Get pruning ratio from command line argument
PRUNING_RATIO=20
echo "Using pruning ratio: $PRUNING_RATIO"


export MODEL_NAME="/gpfs/projects/shlneuroai/caleb/hf_cache/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9"
export DATASET_NAME="/gpfs/projects/shlneuroai/caleb/dataset/flux_dev_finetune_1000"
export MODEL_DIR="/gpfs/projects/shlneuroai/caleb/projects/EcoDiff_Caleb/results/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_16_lr_0.051.00.5_batch_size_4_loss_21_regex_.*_masking_hard_discrete/pruned/flux_demo_pruned_20_16_epochs"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/retraining/train_text_to_image_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pruned_transformer_model_path=$MODEL_DIR"/pruned_model_20.pkl" \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --rank=256 \
  --train_batch_size=4 \
  --max_train_steps=10000 \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="bf16" \
  --checkpoints_total_limit=10 \
  --report_to="wandb" \
  --validation_prompt "A clock tower floating in a sea of clouds" "A cozy library with a roaring fireplace" "an astronaut riding a rainbow unicorn" \
  --validation_epochs 20 \
  --checkpointing_steps=1000 \
  --output_dir=$MODEL_DIR"/retrain_results/flux_lora/" \
  --allow_tf32 \
  --gradient_accumulation_steps 2
