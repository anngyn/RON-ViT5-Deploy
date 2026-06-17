#!/bin/bash
set -e

echo "=== Download ReceiptVQA Dataset ==="
echo ""

# Check if already exists
if [ -d "data/ReceiptVQA-Dataset" ]; then
    echo "Dataset already exists. Remove data/ to re-download."
    exit 0
fi

mkdir -p data

# Choose download method
echo "Select download method:"
echo "1. Kaggle API (recommended)"
echo "2. Google Drive (if you have backup link)"
echo "3. Manual (download separately)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Using Kaggle API..."

        # Check credentials
        if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
            echo ""
            echo "Kaggle credentials not found!"
            echo ""
            echo "Setup steps:"
            echo "1. Go to: https://www.kaggle.com/settings"
            echo "2. Scroll to 'API' section"
            echo "3. Click 'Create New Token'"
            echo "4. Save kaggle.json to ~/.kaggle/"
            echo ""
            echo "Run these commands:"
            echo "  mkdir -p ~/.kaggle"
            echo "  nano ~/.kaggle/kaggle.json"
            echo "  # Paste token JSON and save (Ctrl+O, Enter, Ctrl+X)"
            echo "  chmod 600 ~/.kaggle/kaggle.json"
            echo ""
            echo "Then re-run this script."
            exit 1
        fi

        # Install kaggle
        pip install -q kaggle

        # Download
        echo "Downloading from Kaggle..."
        kaggle datasets download -d anlgyn/receiptvqa -p data/

        # Extract
        echo "Extracting..."
        cd data
        unzip -q receiptvqa.zip
        rm receiptvqa.zip
        cd ..

        echo "✓ Download complete"
        ;;

    2)
        echo ""
        echo "Using Google Drive..."
        read -p "Enter Google Drive file ID: " file_id

        if [ -z "$file_id" ]; then
            echo "Error: File ID cannot be empty"
            exit 1
        fi

        # Install gdown
        pip install -q gdown

        # Download
        echo "Downloading from Google Drive..."
        gdown "$file_id" -O data/receiptvqa.zip

        # Extract
        echo "Extracting..."
        cd data
        unzip -q receiptvqa.zip
        rm receiptvqa.zip
        cd ..

        echo "✓ Download complete"
        ;;

    3)
        echo ""
        echo "Manual download steps:"
        echo ""
        echo "1. Download dataset:"
        echo "   - Kaggle: https://www.kaggle.com/datasets/anlgyn/receiptvqa"
        echo "   - Or from your backup source"
        echo ""
        echo "2. Upload to server (if remote):"
        echo "   scp -r path/to/receiptvqa.zip user@server:~/RON-ViT5-Deploy/data/"
        echo ""
        echo "3. Extract:"
        echo "   cd data"
        echo "   unzip receiptvqa.zip"
        echo "   cd .."
        echo ""
        echo "4. Verify structure:"
        echo "   ls -lh data/ReceiptVQA-Dataset/"
        echo ""
        exit 0
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Verify
echo ""
echo "Verifying dataset structure..."

if [ -d "data/ReceiptVQA-Dataset/ReceiptVQA_annotations/ReceiptVQA_annotations" ]; then
    echo "✓ Annotations found"
else
    echo "✗ Annotations missing"
fi

if [ -d "data/ReceiptVQA-Dataset/features/google_ocr/google_ocr" ]; then
    echo "✓ OCR features found"
else
    echo "✗ OCR features missing"
fi

# Check file counts
if [ -d "data/ReceiptVQA-Dataset" ]; then
    csv_count=$(find data/ReceiptVQA-Dataset -name "*.csv" | wc -l)
    npy_count=$(find data/ReceiptVQA-Dataset -name "*.npy" 2>/dev/null | wc -l)

    echo ""
    echo "Dataset statistics:"
    echo "  CSV files: $csv_count (expected: 3)"
    echo "  NPY files: $npy_count (expected: ~9,769)"

    if [ "$csv_count" -eq 3 ] && [ "$npy_count" -gt 9000 ]; then
        echo ""
        echo "✓ Dataset complete!"
        echo ""
        echo "Next steps:"
        echo "  bash run_baseline_and_noisy.sh"
    else
        echo ""
        echo "⚠️  Dataset may be incomplete"
    fi
fi
