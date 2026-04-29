#!/bin/bash
# Hyperparameter search over beta x ff_lr for baseline pruning (train.py).
# Searches: beta in {0.1, 0.01}, ff_lr in {1.0, 0.1} → 4 runs total.

TUNING_CFG_DIR=configs/flux_schnell_hp_tuning

python scripts/utils/hyperparameter_tuning.py \
    --task gen \
    --config configs/flux.yaml \
    --output_dir "$TUNING_CFG_DIR" \
    --beta 0.1 0.01 \
    --ffn_learning_rate 1.0 0.1 \
    --attn_learning_rate 0.5 \
    --n_learning_rate 0.5 \
    --masking hard_discrete \
    --eps 0.1 \
    --regex ".*" \
    --loss_reg 1 0 \
    --loss_recons 2 \
    --data_size 100 \
    --num_intervention 5 \
    --device 0 \
    --project_name flux_hp \
    --prompts dummy \
    --train_task general

for cfg in "$TUNING_CFG_DIR"/*.yaml; do
    echo "=== Running: $cfg ==="
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --save_dir results_flux_schnell_hp_search \
        --cfg "$cfg"
done
