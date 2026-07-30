#!/bin/bash
set -e

BATCH_SIZE=${1:-8}
LR=${2:-3e-5}
EPOCHS=${3:-3}

echo "=== Flow 3 (mT5): Consistency ==="
echo "Train google/mt5-base on paired clean/noisy data with consistency loss. Eval on clean + 14 noises @ L2."
echo "Batch size: $BATCH_SIZE, LR: $LR, Epochs: $EPOCHS"
echo ""

python scripts/train_consistency_mt5.py \
    --config configs/consistency_mt5.yaml \
    --skip-final-eval \
    --batch-size $BATCH_SIZE \
    --learning-rate $LR \
    --num-epochs $EPOCHS

echo ""
echo "Evaluating consistency_mt5 on clean + 14 noises @ L2"
python scripts/eval_noise_grid_mt5.py \
    --config configs/consistency_mt5.yaml \
    --model-dir outputs/models/consistency_mt5 \
    --model-tag mt5_consistency \
    --levels 2 \
    --output-csv outputs/results/flow3_mt5_consistency_noise_l2.csv

echo ""
echo "Plot flow 3 (mt5)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow3_mt5_consistency_noise_l2.csv \
    --output-prefix outputs/results/flow3_mt5_consistency_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow3_mt5_consistency_noise_l2.csv"
echo "  outputs/results/flow3_mt5_consistency_noise_l2_anls.png"
echo "  outputs/results/flow3_mt5_consistency_noise_l2_drop.png"
