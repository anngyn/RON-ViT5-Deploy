#!/bin/bash
set -e

BATCH_SIZE=${1:-8}
LR=${2:-3e-5}
EPOCHS=${3:-3}

echo "=== Flow 1 (mT5): Baseline clean ==="
echo "Train google/mt5-base on clean data. Eval on clean + 14 noises @ L2."
echo "Batch size: $BATCH_SIZE, LR: $LR, Epochs: $EPOCHS"
echo ""

python scripts/train_baseline_mt5.py \
    --config configs/baseline_mt5.yaml \
    --skip-final-eval \
    --batch-size $BATCH_SIZE \
    --learning-rate $LR \
    --num-epochs $EPOCHS

echo ""
echo "Evaluating baseline_mt5 on clean + 14 noises @ L2"
python scripts/eval_noise_grid_mt5.py \
    --config configs/baseline_mt5.yaml \
    --model-dir outputs/models/baseline_mt5 \
    --model-tag mt5_clean \
    --levels 2 \
    --output-csv outputs/results/flow1_mt5_clean_noise_l2.csv

echo ""
echo "Plot flow 1 (mt5)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow1_mt5_clean_noise_l2.csv \
    --output-prefix outputs/results/flow1_mt5_clean_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow1_mt5_clean_noise_l2.csv"
echo "  outputs/results/flow1_mt5_clean_noise_l2_anls.png"
echo "  outputs/results/flow1_mt5_clean_noise_l2_drop.png"
