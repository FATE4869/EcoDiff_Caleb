#!/bin/bash
# Hyperparameter search over beta x ff_lr for baseline pruning (train.py).
# Searches: beta in {0.1, 0.01}, ff_lr in {1.0, 0.1} → 4 runs total.

TUNING_CFG_DIR=configs/sdxl_hp_tuning

python scripts/utils/hyperparameter_tuning.py \
    --task gen \
    --config configs/sdxl.yaml \
    --output_dir "$TUNING_CFG_DIR" \
    --beta 0.5 \
    --ffn_learning_rate 0.15 \
    --attn_learning_rate 0.15 \
    --masking hard_discrete \
    --regex ".*" \
    --eps 0.5 \
    --loss_reg 2 1 0 \
    --loss_recons 2 1 \
    --data_size 100 \
    --num_intervention 50 \
    --device 0 \
    --project_name sdxl_hp \
    --prompts dummy \
    --train_task general \
    --results_output_dir results_sdxl_hp_search

# for cfg in "$TUNING_CFG_DIR"/*.yaml; do
#     echo "=== Running: $cfg ==="
#     CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
#         --save_dir results_sdxl_hp_search \
#         --cfg "$cfg"
# done
