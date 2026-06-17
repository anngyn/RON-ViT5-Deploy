#!/bin/bash
set -e

echo "=== RON-ViT5 Setup ==="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download dataset from Kaggle (requires kaggle.json in ~/.kaggle/)
echo "Downloading ReceiptVQA dataset..."
if [ ! -d "data" ]; then
    mkdir -p data

    # Option 1: Kaggle API (uncomment if using)
    # pip install kaggle
    # kaggle datasets download -d anlgyn/receiptvqa -p data/
    # unzip data/receiptvqa.zip -d data/

    # Option 2: Google Drive (provide your own link)
    # gdown <YOUR_GDRIVE_FILE_ID> -O data/receiptvqa.zip
    # unzip data/receiptvqa.zip -d data/

    echo "⚠️  Manual step required:"
    echo "   Upload ReceiptVQA dataset to ./data/"
    echo "   Expected structure:"
    echo "     data/ReceiptVQA-Dataset/"
    echo "       ├── ReceiptVQA_annotations/"
    echo "       └── features/google_ocr/google_ocr/"
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
