#!/bin/bash
set -e

BATCH_SIZE=${1:-8}
LR=${2:-3e-5}
EPOCHS=${3:-3}

echo "=== Flow 2 (mT5): Noisy Aug ==="
echo "Train google/mt5-base on clean + augmented data. Eval on clean + 14 noises @ L2."
echo "Batch size: $BATCH_SIZE, LR: $LR, Epochs: $EPOCHS"
echo ""

python scripts/train_noisy_aug_mt5.py \
    --config configs/noisy_aug_mt5.yaml \
    --skip-final-eval \
    --batch-size $BATCH_SIZE \
    --learning-rate $LR \
    --num-epochs $EPOCHS

echo ""
echo "Evaluating noisy_aug_mt5 on clean + 14 noises @ L2"
python scripts/eval_noise_grid_mt5.py \
    --config configs/noisy_aug_mt5.yaml \
    --model-dir outputs/models/noisy_aug_mt5 \
    --model-tag mt5_aug \
    --levels 2 \
    --output-csv outputs/results/flow2_mt5_aug_noise_l2.csv

echo ""
echo "Plot flow 2 (mt5)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow2_mt5_aug_noise_l2.csv \
    --output-prefix outputs/results/flow2_mt5_aug_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow2_mt5_aug_noise_l2.csv"
echo "  outputs/results/flow2_mt5_aug_noise_l2_anls.png"
echo "  outputs/results/flow2_mt5_aug_noise_l2_drop.png"
