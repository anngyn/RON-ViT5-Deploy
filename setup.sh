#!/bin/bash
set -e

echo "=== RON-ViT5 Setup ==="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup data directory
mkdir -p data

# Check if dataset already exists
if [ -d "data/ReceiptVQA-Dataset" ]; then
    echo "✓ Dataset already exists in data/"
else
    echo "Downloading ReceiptVQA dataset..."

    # Try Kaggle API first
    if [ -f "$HOME/.kaggle/kaggle.json" ]; then
        echo "Using Kaggle API..."
        pip install -q kaggle
        kaggle datasets download -d anlgyn/receiptvqa -p data/

        echo "Extracting dataset..."
        cd data
        unzip -q receiptvqa.zip
        rm receiptvqa.zip
        cd ..
        echo "✓ Dataset downloaded from Kaggle"
    else
        echo ""
        echo "⚠️  Kaggle credentials not found"
        echo ""
        echo "Setup options:"
        echo ""
        echo "Option 1: Kaggle API (recommended)"
        echo "  1. Get API token from: https://www.kaggle.com/settings"
        echo "  2. Create ~/.kaggle/kaggle.json:"
        echo "     mkdir -p ~/.kaggle"
        echo "     nano ~/.kaggle/kaggle.json"
        echo "  3. Paste token JSON and save"
        echo "  4. chmod 600 ~/.kaggle/kaggle.json"
        echo "  5. Re-run: bash setup.sh"
        echo ""
        echo "Option 2: Manual upload"
        echo "  1. Download from: https://www.kaggle.com/datasets/anlgyn/receiptvqa"
        echo "  2. Upload to server and extract to: ./data/ReceiptVQA-Dataset/"
        echo ""
        echo "Option 3: Google Drive (if you have backup)"
        echo "  pip install gdown"
        echo "  gdown <DRIVE_FILE_ID> -O data/receiptvqa.zip"
        echo "  cd data && unzip receiptvqa.zip && cd .."
        echo ""
    fi
fi

# Verify dataset structure
if [ -d "data/ReceiptVQA-Dataset/ReceiptVQA_annotations" ] && [ -d "data/ReceiptVQA-Dataset/features/google_ocr" ]; then
    echo "✓ Dataset structure verified"
else
    echo "⚠️  Warning: Dataset structure incomplete"
    echo "Expected:"
    echo "  data/ReceiptVQA-Dataset/"
    echo "    ├── ReceiptVQA_annotations/ReceiptVQA_annotations/"
    echo "    └── features/google_ocr/google_ocr/"
fi

# Create output directories
mkdir -p outputs/models
mkdir -p outputs/results
mkdir -p outputs/logs

echo "✓ Setup complete"
echo ""
echo "Next steps:"
echo "  1. Ensure dataset is in ./data/"
echo "  2. Run: bash run_all.sh"
