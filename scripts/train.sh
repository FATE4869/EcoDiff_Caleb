CUDA_VISIBLE_DEVICES=0 python scripts/load_pruned_model.py --model flux \
--num_intervention_steps 4 --dst ../pruned/pruned_10 \
--scope "global" --ratio 0.9 --save_pruned_model \
--save_pt results/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_16_lr_0.051.00.5_batch_size_4_loss_21_regex_.\*_masking_hard_discrete/lambda/epoch_15_step_400_attn.pt


# accelerate launch --num_processes 1 --main_process_port 29501 scripts/evaluation/semantic_eval_dataset.py \
# --save_dir results/generated_images_flux_schnell_4steps_pruned_10_epochs_16/ \
# --num_intervention_steps 4 --model flux --dataset_name flickr --max_size 100 --image_size 512 \
# --pruned_model_pt /gpfs/projects/shlneuroai/caleb/projects/EcoDiff_Caleb/results/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_16_lr_0.051.00.5_batch_size_4_loss_21_regex_.*_masking_hard_discrete/lambda/results/flux_demo_pruned_20_16_epochs/pruned_model_10.pkl

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --save_dir results --cfg configs/flux.yaml

CUDA_VISIBLE_DEVICES=0 python scripts/get_retrained_model.py --model flux --num_intervention_steps 4 \
--dst results/retrained/retrained_20 --full_lora \
--pruned_model_pt results/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_16_lr_0.051.00.5_batch_size_4_loss_21_regex_.*_masking_hard_discrete/pruned/flux_demo_pruned_20_16_epochs/pruned_model_20.pkl \
--lora_pt results/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_16_lr_0.051.00.5_batch_size_4_loss_21_regex_.*_masking_hard_discrete/pruned/flux_demo_pruned_20_16_epochs/retrain_results/flux_lora/checkpoint-4000
