#!/bin/bash
set -e

echo "=== Q1 Robustness Flows ==="
echo "Flow 1: ViT_clean"
echo "Flow 2: ViT_aug"
echo "Flow 3: ViT_consistency"
echo ""

if ! command -v python &> /dev/null; then
    echo "Error: python not found"
    exit 1
fi

echo "1/3 Train baseline"
python scripts/train_baseline.py --config configs/baseline.yaml --skip-final-eval

echo ""
echo "1/3 Eval baseline on clean + 14 noises @ L2"
python scripts/eval_noise_grid.py \
    --config configs/baseline.yaml \
    --model-dir outputs/models/baseline \
    --model-tag vit_clean \
    --levels 2 \
    --output-csv outputs/results/flow1_vit_clean_noise_l2.csv

echo ""
echo "2/3 Train noisy augmentation"
python scripts/train_noisy_aug.py --config configs/noisy_aug.yaml --skip-final-eval

echo ""
echo "2/3 Eval noisy augmentation on clean + 14 noises @ L2"
python scripts/eval_noise_grid.py \
    --config configs/noisy_aug.yaml \
    --model-dir outputs/models/noisy_aug \
    --model-tag vit_aug \
    --levels 2 \
    --output-csv outputs/results/flow2_vit_aug_noise_l2.csv

echo ""
echo "3/3 Train consistency"
python scripts/train_consistency.py --config configs/consistency_16gb.yaml --skip-final-eval

echo ""
echo "3/3 Eval consistency on clean + 14 noises @ L2"
python scripts/eval_noise_grid.py \
    --config configs/consistency_16gb.yaml \
    --model-dir outputs/models/consistency_only \
    --model-tag consistency \
    --levels 2 \
    --output-csv outputs/results/flow3_consistency_noise_l2.csv

echo ""
echo "Plot combined charts"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow1_vit_clean_noise_l2.csv outputs/results/flow2_vit_aug_noise_l2.csv outputs/results/flow3_consistency_noise_l2.csv \
    --output-prefix outputs/results/q1_noise_l2_compare

echo ""
echo "Done."
echo "CSV:"
echo "  outputs/results/flow1_vit_clean_noise_l2.csv"
echo "  outputs/results/flow2_vit_aug_noise_l2.csv"
echo "  outputs/results/flow3_consistency_noise_l2.csv"
echo "Plots:"
echo "  outputs/results/q1_noise_l2_compare_anls.png"
echo "  outputs/results/q1_noise_l2_compare_drop.png"
