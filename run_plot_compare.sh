#!/bin/bash
set -e

echo "=== Plot Compare: Flow 1 vs Flow 2 vs Flow 3 ==="
echo "Input CSV:"
echo "  outputs/results/flow1_vit_clean_noise_l2.csv"
echo "  outputs/results/flow2_vit_aug_noise_l2.csv"
echo "  outputs/results/flow3_consistency_noise_l2.csv"
echo ""

for file in \
    outputs/results/flow1_vit_clean_noise_l2.csv \
    outputs/results/flow2_vit_aug_noise_l2.csv \
    outputs/results/flow3_consistency_noise_l2.csv
do
    if [ ! -f "$file" ]; then
        echo "Error: missing $file"
        echo "Run training/eval flows first."
        exit 1
    fi
done

python scripts/plot_noise_results.py \
    --csv outputs/results/flow1_vit_clean_noise_l2.csv outputs/results/flow2_vit_aug_noise_l2.csv outputs/results/flow3_consistency_noise_l2.csv \
    --output-prefix outputs/results/q1_noise_l2_compare

echo ""
echo "Done:"
echo "  outputs/results/q1_noise_l2_compare_anls.png"
echo "  outputs/results/q1_noise_l2_compare_drop.png"
