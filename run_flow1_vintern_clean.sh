#!/bin/bash
set -e

BATCH_SIZE=${1:-2}
LR=${2:-1e-4}
EPOCHS=${3:-3}

echo "=== Flow 1 (Vintern-1B, text-only): Baseline clean ==="
echo "LoRA-fine-tune Qwen2 LLM inside Vintern-1B on clean OCR context. Eval on clean + 14 noises @ L2."
echo "Batch: $BATCH_SIZE, LR: $LR, Epochs: $EPOCHS"

pip install -q "transformers==4.37.2" "peft==0.10.0" accelerate timm einops sentencepiece

python scripts/train_vintern.py \
    --config configs/baseline_vintern.yaml \
    --flow baseline \
    --skip-final-eval \
    --batch-size $BATCH_SIZE \
    --learning-rate $LR \
    --num-epochs $EPOCHS

echo ""
echo "Evaluating baseline_vintern on clean + 14 noises @ L2"
python scripts/eval_noise_grid_vintern.py \
    --config configs/baseline_vintern.yaml \
    --model-dir outputs/models/baseline_vintern \
    --model-tag vintern_clean \
    --levels 2 \
    --output-csv outputs/results/flow1_vintern_clean_noise_l2.csv

echo ""
echo "Plot flow 1 (vintern)"
python scripts/plot_noise_results.py \
    --csv outputs/results/flow1_vintern_clean_noise_l2.csv \
    --output-prefix outputs/results/flow1_vintern_clean_noise_l2

echo ""
echo "Done: outputs/results/flow1_vintern_clean_noise_l2.csv (+ _anls.png, _drop.png)"
