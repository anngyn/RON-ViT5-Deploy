#!/bin/bash
set -e

echo "=== RON-ViT5 Training Pipeline (16GB GPU) ==="
echo "Modified configs for 16GB memory limit"
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

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Run training with optimized configs
echo "1/5: Training Baseline (batch=8, ~4GB)..."
python scripts/train_baseline.py

echo ""
echo "2/5: Training Noisy Augmentation (batch=8, ~5GB)..."
python scripts/train_noisy_aug.py

echo ""
echo "3/5: Training Adapter Only (batch=8, ~3GB)..."
python scripts/train_adapter.py

echo ""
echo "4/5: Training Consistency Only (batch=4, ~7GB)..."
# Use 16GB config
python scripts/train_consistency.py --config configs/consistency_16gb.yaml

echo ""
echo "5/5: Training RON-NACA (batch=4, ~6GB)..."
# Use 16GB config
python scripts/train_ron_naca.py --config configs/ron_naca_16gb.yaml

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
