#!/bin/bash
set -e

echo "=== RON-ViT5: Baseline + Noisy Augmentation ==="
echo "Training 2 methods sequentially (~2 hours total)"
echo ""

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
else
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Check data
if [ ! -d "data/ReceiptVQA-Dataset" ]; then
    echo "❌ Error: Dataset not found at data/ReceiptVQA-Dataset"
    echo "Run setup.sh first and upload dataset."
    exit 1
fi

# Method 1: Baseline
echo "1/2: Training Baseline (clean only)..."
echo "Expected time: ~1 hour"
echo "Expected GPU memory: ~4-7GB"
echo ""
python scripts/train_baseline.py

# Method 2: Noisy Augmentation
echo ""
echo "2/2: Training Noisy Augmentation (clean + noisy)..."
echo "Expected time: ~1.2 hours"
echo "Expected GPU memory: ~5-8GB"
echo ""
python scripts/train_noisy_aug.py

echo ""
echo "=== Training Complete ==="
echo ""
echo "Results:"
ls -lh outputs/results/*.csv

echo ""
echo "Compare ANLS scores:"
echo ""
echo "Baseline:"
cat outputs/results/baseline_results.csv
echo ""
echo "Noisy Aug:"
cat outputs/results/noisy_aug_results.csv

echo ""
echo "Expected improvement on noisy test:"
echo "  Mixed noise: +0.02 to +0.04 ANLS"
echo ""
echo "Next steps:"
echo "  - Download results: outputs/results/"
echo "  - Check logs: outputs/logs/"
echo "  - Run remaining methods: bash run_all.sh"
