#!/bin/bash
set -e

BATCH_SIZE=${1:-8}  # default 8 if not provided

echo "=== Flow 1 (BARTpho): Baseline clean ==="
echo "Train BARTpho-syllable-base on clean data. Eval on clean + 14 noises @ L2."
echo "Batch size: $BATCH_SIZE"
echo ""

python scripts/train_baseline_bartpho.py --config configs/baseline_bartpho.yaml --skip-final-eval --batch-size $BATCH_SIZE

echo ""
echo "Evaluating baseline_bartpho on clean + 14 noises @ L2"
python scripts/eval_noise_grid_bartpho.py \
    --config configs/baseline_bartpho.yaml \
    --model-dir outputs/models/baseline_bartpho \
    --model-tag bartpho_clean \
    --levels 2 \
    --output-csv outputs/results/flow1_bartpho_clean_noise_l2.csv

echo ""
echo "Plot flow 1 (bartpho)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow1_bartpho_clean_noise_l2.csv \
    --output-prefix outputs/results/flow1_bartpho_clean_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow1_bartpho_clean_noise_l2.csv"
echo "  outputs/results/flow1_bartpho_clean_noise_l2_anls.png"
echo "  outputs/results/flow1_bartpho_clean_noise_l2_drop.png"
