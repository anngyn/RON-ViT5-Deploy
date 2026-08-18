#!/bin/bash
set -e

BATCH_SIZE=${1:-2}
LR=${2:-1e-4}
EPOCHS=${3:-3}

echo "=== Flow 2 (Vintern-1B, text-only): Noisy augmentation ==="
echo "LoRA-fine-tune on randomly noised OCR context. Eval on clean + 14 noises @ L2."
echo "Batch: $BATCH_SIZE, LR: $LR, Epochs: $EPOCHS"

pip install -q "transformers==4.37.2" "peft==0.10.0" accelerate timm einops sentencepiece

python scripts/train_vintern.py \
    --config configs/noisy_aug_vintern.yaml \
    --flow noisy_aug \
    --skip-final-eval \
    --batch-size $BATCH_SIZE \
    --learning-rate $LR \
    --num-epochs $EPOCHS

echo ""
echo "Evaluating noisy_aug_vintern on clean + 14 noises @ L2"
python scripts/eval_noise_grid_vintern.py \
    --config configs/noisy_aug_vintern.yaml \
    --model-dir outputs/models/noisy_aug_vintern \
    --model-tag vintern_aug \
    --levels 2 \
    --output-csv outputs/results/flow2_vintern_aug_noise_l2.csv

echo ""
echo "Plot flow 2 (vintern)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow2_vintern_aug_noise_l2.csv \
    --output-prefix outputs/results/flow2_vintern_aug_noise_l2

echo ""
echo "Done: outputs/results/flow2_vintern_aug_noise_l2.csv (+ _anls.png, _drop.png)"
