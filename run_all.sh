#!/bin/bash
set -e

echo "=== RON-ViT5 Training Pipeline ==="
echo "This will train 5 methods sequentially (~5-6 hours total on A100)"
echo ""

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
fi

# Check data
if [ ! -d "data/ReceiptVQA-Dataset" ]; then
    echo "❌ Error: Dataset not found at data/ReceiptVQA-Dataset"
    echo "Run setup.sh first and upload dataset."
    exit 1
fi

# Run training
echo ""
echo "1/5: Training Baseline (clean only)..."
python scripts/train_baseline.py

echo ""
echo "2/5: Training Noisy Augmentation..."
python scripts/train_noisy_aug.py

echo ""
echo "3/5: Training Adapter Only..."
python scripts/train_adapter.py

echo ""
echo "4/5: Training Consistency Only..."
python scripts/train_consistency.py

echo ""
echo "5/5: Training RON-NACA (full method)..."
python scripts/train_ron_naca.py

echo ""
echo "=== Training Complete ==="
echo ""
echo "Results saved to:"
echo "  - outputs/models/"
echo "  - outputs/results/"
echo "  - outputs/logs/"
echo ""
echo "Compare results:"
ls -lh outputs/results/*.csv
