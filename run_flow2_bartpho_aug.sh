#!/bin/bash
set -e

BATCH_SIZE=${1:-8}  # default 8 if not provided

echo "=== Flow 2 (BARTpho): Noisy Aug ==="
echo "Train BARTpho-syllable-base on clean + augmented data. Eval on clean + 14 noises @ L2."
echo "Batch size: $BATCH_SIZE"
echo ""

python scripts/train_noisy_aug_bartpho.py --config configs/noisy_aug_bartpho.yaml --skip-final-eval --batch-size $BATCH_SIZE

echo ""
echo "Evaluating noisy_aug_bartpho on clean + 14 noises @ L2"
python scripts/eval_noise_grid_bartpho.py \
    --config configs/noisy_aug_bartpho.yaml \
    --model-dir outputs/models/noisy_aug_bartpho \
    --model-tag bartpho_aug \
    --levels 2 \
    --output-csv outputs/results/flow2_bartpho_aug_noise_l2.csv

echo ""
echo "Plot flow 2 (bartpho)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow2_bartpho_aug_noise_l2.csv \
    --output-prefix outputs/results/flow2_bartpho_aug_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow2_bartpho_aug_noise_l2.csv"
echo "  outputs/results/flow2_bartpho_aug_noise_l2_anls.png"
echo "  outputs/results/flow2_bartpho_aug_noise_l2_drop.png"
