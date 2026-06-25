#!/bin/bash
set -e

echo "=== Flow 2: ViT_aug ==="
echo "Train on clean + augmented data. Eval on clean + 14 noises @ L2."
echo ""

python scripts/train_noisy_aug.py --config configs/noisy_aug.yaml

echo ""
echo "Evaluating noisy augmentation on clean + 14 noises @ L2"
python scripts/eval_noise_grid.py \
    --config configs/noisy_aug.yaml \
    --model-dir outputs/models/noisy_aug \
    --model-tag vit_aug \
    --levels 2 \
    --output-csv outputs/results/flow2_vit_aug_noise_l2.csv

echo ""
echo "Plot flow 2"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow2_vit_aug_noise_l2.csv \
    --output-prefix outputs/results/flow2_vit_aug_noise_l2

echo ""
echo "Done:"
echo "  outputs/results/flow2_vit_aug_noise_l2.csv"
echo "  outputs/results/flow2_vit_aug_noise_l2_anls.png"
echo "  outputs/results/flow2_vit_aug_noise_l2_drop.png"
