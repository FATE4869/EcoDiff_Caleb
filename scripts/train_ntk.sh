# CUDA_VISIBLE_DEVICES=0 python scripts/train_ntk.py --save_dir results_ntk --cfg configs/flux_ntk.yaml --ntk_lambda 0.1

# python scripts/load_pruned_model.py --model flux --dst results_flux/prune_results --num_intervention_steps 4 \
# --save_pt results_ntk/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_4_lr_0.051.00.5_batch_size_4_loss_21_regex_.\*_masking_hard_discrete/lambda/epoch_3_step_20_attn.pt --scope "global" --ratio 0.8 --save_pruned_model


accelerate launch --num_processes 1 scripts/evaluation/semantic_eval_dataset.py \
--save_dir results_ntk/generated_images_flux_schnell_4steps_pruned_10/ \
--num_intervention_steps 4 --model flux --dataset_name flickr --max_size 50 --image_size 512 \
--pruned_model_pt /gpfs/projects/shlneuroai/caleb/projects/EcoDiff_Caleb/results_ntk/flux/model_flux_eps_0.1_sample_100_beta_0.1_epochs_4_lr_0.051.00.5_batch_size_4_loss_21_regex_.*_masking_hard_discrete/lambda/results_flux/prune_results_0.9/pruned_model_10.pkl