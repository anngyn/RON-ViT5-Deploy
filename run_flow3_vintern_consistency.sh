#!/bin/bash
set -e

BATCH_SIZE=${1:-1}
LR=${2:-1e-4}
EPOCHS=${3:-3}

echo "=== Flow 3 (Vintern-1B, text-only): Consistency ==="
echo "LoRA-fine-tune on clean/noisy pairs + hidden-state consistency. Eval on clean + 14 noises @ L2."
echo "Batch: $BATCH_SIZE, LR: $LR, Epochs: $EPOCHS"

pip install -q "transformers==4.37.2" "peft==0.10.0" accelerate timm einops sentencepiece

python scripts/train_vintern.py \
    --config configs/consistency_vintern.yaml \
    --flow consistency \
    --skip-final-eval \
    --batch-size $BATCH_SIZE \
    --learning-rate $LR \
    --num-epochs $EPOCHS

echo ""
echo "Evaluating consistency_vintern on clean + 14 noises @ L2"
python scripts/eval_noise_grid_vintern.py \
    --config configs/consistency_vintern.yaml \
    --model-dir outputs/models/consistency_vintern \
    --model-tag vintern_consistency \
    --levels 2 \
    --max-eval-samples 1000 \
    --output-csv outputs/results/flow3_vintern_consistency_noise_l2.csv

echo ""
echo "Plot flow 3 (vintern)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow3_vintern_consistency_noise_l2.csv \
    --output-prefix outputs/results/flow3_vintern_consistency_noise_l2

echo ""
echo "Done: outputs/results/flow3_vintern_consistency_noise_l2.csv (+ _anls.png, _drop.png)"
