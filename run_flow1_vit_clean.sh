#!/bin/bash
set -e

echo "=== Flow 1: ViT_clean ==="
echo "Train on clean data. Eval on clean + 14 noises @ L2."
echo ""

python scripts/train_baseline.py --config configs/baseline.yaml

echo ""
echo "Evaluating baseline on clean + 14 noises @ L2"
python scripts/eval_noise_grid.py \
    --config configs/baseline.yaml \
    --model-dir outputs/models/baseline \
    --model-tag vit_clean \
    --levels 2 \
    --output-csv outputs/results/flow1_vit_clean_noise_l2.csv

echo ""
echo "Plot flow 1"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow1_vit_clean_noise_l2.csv \
    --output-prefix outputs/results/flow1_vit_clean_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow1_vit_clean_noise_l2.csv"
echo "  outputs/results/flow1_vit_clean_noise_l2_anls.png"
echo "  outputs/results/flow1_vit_clean_noise_l2_drop.png"
