#!/bin/bash
set -e

echo "=== Flow 3: Consistency ==="
echo "Train on paired clean/noisy data with consistency loss. Eval on clean + 14 noises @ L2."
echo ""

python scripts/train_consistency.py --config configs/consistency_16gb.yaml --skip-final-eval

echo ""
echo "Evaluating consistency model on clean + 14 noises @ L2"
python scripts/eval_noise_grid.py \
    --config configs/consistency_16gb.yaml \
    --model-dir outputs/models/consistency_only \
    --model-tag consistency \
    --levels 2 \
    --output-csv outputs/results/flow3_consistency_noise_l2.csv

echo ""
echo "Plot flow 3"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow3_consistency_noise_l2.csv \
    --output-prefix outputs/results/flow3_consistency_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow3_consistency_noise_l2.csv"
echo "  outputs/results/flow3_consistency_noise_l2_anls.png"
echo "  outputs/results/flow3_consistency_noise_l2_drop.png"
