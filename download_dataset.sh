#!/bin/bash
set -euo pipefail

echo "=== Prepare ReceiptVQA Dataset ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

extract_nested_archives() {
    local base="$1"

    if [ -f "$base/ReceiptVQA_annotations.zip" ] && [ ! -d "$base/ReceiptVQA_annotations" ]; then
        echo "Extracting nested annotations zip..."
        unzip -q "$base/ReceiptVQA_annotations.zip" -d "$base"
    fi

    if [ -f "$base/features/google_ocr.zip" ] && [ ! -d "$base/features/google_ocr" ]; then
        echo "Extracting nested OCR zip..."
        mkdir -p "$base/features"
        unzip -q "$base/features/google_ocr.zip" -d "$base/features"
    fi
}

verify_dataset() {
    local base="$1"
    local ann_dir=""
    local ocr_dir=""

    if [ -d "$base/ReceiptVQA_annotations/ReceiptVQA_annotations" ]; then
        ann_dir="$base/ReceiptVQA_annotations/ReceiptVQA_annotations"
    elif [ -d "$base/ReceiptVQA_annotations" ]; then
        ann_dir="$base/ReceiptVQA_annotations"
    fi

    if [ -d "$base/features/google_ocr/google_ocr" ]; then
        ocr_dir="$base/features/google_ocr/google_ocr"
    elif [ -d "$base/features/google_ocr" ]; then
        ocr_dir="$base/features/google_ocr"
    fi

    if [ -z "$ann_dir" ] || [ -z "$ocr_dir" ]; then
        return 1
    fi

    [ -f "$ann_dir/ReceiptVQA_train.csv" ] || return 1
    [ -f "$ann_dir/ReceiptVQA_dev.csv" ] || return 1
    [ -f "$ann_dir/ReceiptVQA_test.csv" ] || return 1

    local npy_count
    npy_count=$(find "$ocr_dir" -maxdepth 1 -name "*.npy" | wc -l)
    [ "$npy_count" -gt 9000 ] || return 1

    echo "Dataset OK:"
    echo "  base:        $base"
    echo "  annotations: $ann_dir"
    echo "  OCR npy:     $ocr_dir ($npy_count files)"
}

if verify_dataset "data/ReceiptVQA-Dataset"; then
    exit 0
fi

if verify_dataset "ReceiptVQA-Dataset"; then
    echo "Training scripts support this root-level dataset path."
    exit 0
fi

mkdir -p data

zip_path=""
if [ -f "ReceiptVQA-Dataset-20260617T063456Z-3-003.zip" ]; then
    zip_path="ReceiptVQA-Dataset-20260617T063456Z-3-003.zip"
else
    zip_path=$(find . -maxdepth 1 \( -name "*ReceiptVQA*.zip" -o -name "*receiptvqa*.zip" \) | head -n 1 || true)
fi

if [ -n "$zip_path" ]; then
    echo "Extracting $zip_path..."
    unzip -q "$zip_path" -d data
else
    echo "No local ReceiptVQA zip found. Trying Kaggle API..."
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo "Missing Kaggle credentials: $HOME/.kaggle/kaggle.json"
        echo "Place ReceiptVQA-Dataset in this folder or add Kaggle credentials."
        exit 1
    fi
    pip install -q kaggle
    kaggle datasets download -d anlgyn/receiptvqa -p data/
    unzip -q data/receiptvqa.zip -d data
    rm -f data/receiptvqa.zip
fi

extract_nested_archives "data/ReceiptVQA-Dataset"
if [ -d "ReceiptVQA-Dataset" ]; then
    extract_nested_archives "ReceiptVQA-Dataset"
fi

if verify_dataset "data/ReceiptVQA-Dataset"; then
    exit 0
fi

if verify_dataset "ReceiptVQA-Dataset"; then
    exit 0
fi

echo "Dataset incomplete after extraction."
echo "Expected CSV under ReceiptVQA_annotations and OCR NPY under features/google_ocr."
exit 1
