#!/bin/bash
#SBATCH --job-name=semantic_eval_flux_schnell_4steps_pruned_10
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=ckpt-all
#SBATCH --account=stf                                                                                                             
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00-5:00:00
#SBATCH --signal=SIGUSR1@90

# Activate your conda
source $(conda info --base)/etc/profile.d/conda.sh                                                                                                                                                   
conda activate /mmfs1/gscratch/shlneuroai/zheng94/envs/sdib
# Run your Python script
# python scripts/inference_pruned_model.py --model flux --num_intervention_steps 5 --dst results/flux_demo_pr20/ --pruned_model_pt ../../hf_cache/ecodiff_flux_prune/schnell/pruned_model_20.pkl 
# python scripts/evaluation/semantic_eval.py --model flux --pruned_model_pt ../../hf_cache/ecodiff_flux_prune/schnell/pruned_model_20.pkl --task fid

# accelerate launch --num_processes 1 --main_process_port 29501 scripts/evaluation/semantic_eval_dataset.py \
# --save_dir results/generated_images_flux_schnell_4steps_pruned_10/ \
# --num_intervention_steps 4 --model flux --dataset_name flickr --max_size 100 --image_size 512 \
# --pruned_model_pt /mmfs1/gscratch/shlneuroai/zheng94/hf_cache/ecodiff_flux_prune/schnell/pruned_model_20.pkl

# python scripts/load_pruned_model.py --model flux --num_intervention_steps 4 --dst results/pruned_images \
# --save_pt /home/jovyan/projects/EcoDiff_Caleb/results/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_4_lr_0.051.00.5_batch_size_4_loss_21_regex_.*_masking_hard_discrete/lambda/epoch_3_step_100_attn.pt \
# --scope "global" --ratio 0.8 --save_pruned_model

# python scripts/inference_pruned_model.py --model flux --num_intervention_steps 4 --dst results \
# --pruned_model_pt /home/jovyan/projects/EcoDiff_Caleb/results/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_4_lr_0.051.00.5_batch_size_4_loss_21_regex_.*_masking_hard_discrete/results/pruned_images/pruned_model_20.pkl

# python scripts/evaluation/image_quality_eval.py --task gen --model flux --dataset_name flickr --data_size 5000

accelerate launch --num_processes 1 --main_process_port 29501 scripts/evaluation/semantic_eval_dataset.py \
--save_dir results/generated_images_flux_schnell_4steps_pruned_20_caleb/ \
--num_intervention_steps 4 --model flux --dataset_name flickr --max_size 10 --image_size 512 \
--pruned_model_pt /home/jovyan/projects/EcoDiff_Caleb/results/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_4_lr_0.051.00.5_batch_size_4_loss_21_regex_.*_masking_hard_discrete/results/pruned_images/pruned_model_20.pkl