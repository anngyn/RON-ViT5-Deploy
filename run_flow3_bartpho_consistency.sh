#!/bin/bash
set -e

BATCH_SIZE=${1:-8}  # default 8 if not provided

echo "=== Flow 3 (BARTpho): Consistency ==="
echo "Train BARTpho-syllable-base on paired clean/noisy data with consistency loss. Eval on clean + 14 noises @ L2."
echo "Batch size: $BATCH_SIZE"
echo ""

python scripts/train_consistency_bartpho.py --config configs/consistency_bartpho.yaml --skip-final-eval --batch-size $BATCH_SIZE

echo ""
echo "Evaluating consistency_bartpho on clean + 14 noises @ L2"
python scripts/eval_noise_grid_bartpho.py \
    --config configs/consistency_bartpho.yaml \
    --model-dir outputs/models/consistency_bartpho \
    --model-tag bartpho_consistency \
    --levels 2 \
    --output-csv outputs/results/flow3_bartpho_consistency_noise_l2.csv

echo ""
echo "Plot flow 3 (bartpho)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow3_bartpho_consistency_noise_l2.csv \
    --output-prefix outputs/results/flow3_bartpho_consistency_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow3_bartpho_consistency_noise_l2.csv"
echo "  outputs/results/flow3_bartpho_consistency_noise_l2_anls.png"
echo "  outputs/results/flow3_bartpho_consistency_noise_l2_drop.png"
