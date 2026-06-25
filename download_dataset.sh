#!/bin/bash
set -euo pipefail

echo "=== Prepare ReceiptVQA Dataset ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DEFAULT_DRIVE_FILE_URL="https://drive.google.com/file/d/1-LJyQ27HNmfCCuE1AAnwF2WYm84VlWPS/view?usp=drive_link"
DEFAULT_DRIVE_FILE_ID="1-LJyQ27HNmfCCuE1AAnwF2WYm84VlWPS"
DEFAULT_DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1uw7v_tVAaszkDSvt5MdtpgWAImoy4Tq1?usp=drive_link"
DEFAULT_DRIVE_FOLDER_ID="1uw7v_tVAaszkDSvt5MdtpgWAImoy4Tq1"

normalize_folder_layout() {
    local base="$1"

    if [ -d "$base/ReceiptVQA-Dataset" ] && [ ! -d "$base/ReceiptVQA_annotations" ] && [ ! -d "$base/features" ]; then
        echo "Flattening nested ReceiptVQA-Dataset directory..."
        mv "$base/ReceiptVQA-Dataset"/* "$base"/
        rmdir "$base/ReceiptVQA-Dataset"
    fi
}

download_from_drive() {
    local drive_url="${RECEIPTVQA_DRIVE_URL:-$DEFAULT_DRIVE_FILE_URL}"
    local drive_id="${RECEIPTVQA_DRIVE_ID:-$DEFAULT_DRIVE_FILE_ID}"
    local folder_url="${RECEIPTVQA_DRIVE_FOLDER_URL:-$DEFAULT_DRIVE_FOLDER_URL}"
    local folder_id="${RECEIPTVQA_DRIVE_FOLDER_ID:-$DEFAULT_DRIVE_FOLDER_ID}"
    local output_zip="data/receiptvqa_drive.zip"
    local folder_tmp="data/receiptvqa_drive_folder"

    pip install -q gdown

    echo "Downloading ReceiptVQA zip from Google Drive..."
    rm -f "$output_zip"
    if gdown --fuzzy "$drive_url" -O "$output_zip"; then
        unzip -q "$output_zip" -d data
        rm -f "$output_zip"
        return 0
    fi

    echo "Primary Drive file URL failed. Retrying with file id..."
    rm -f "$output_zip"
    if gdown "$drive_id" -O "$output_zip"; then
        unzip -q "$output_zip" -d data
        rm -f "$output_zip"
        return 0
    fi

    echo "Drive zip download failed. Trying shared folder..."
    rm -rf "$folder_tmp"
    mkdir -p "$folder_tmp"
    if ! gdown --folder "$folder_url" -O "$folder_tmp"; then
        echo "Folder URL failed. Retrying with folder id..."
        rm -rf "$folder_tmp"
        mkdir -p "$folder_tmp"
        gdown --folder "https://drive.google.com/drive/folders/$folder_id" -O "$folder_tmp" || return 1
    fi

    local folder_zip
    folder_zip=$(find "$folder_tmp" -maxdepth 2 -type f \( -name "*ReceiptVQA*.zip" -o -name "*receiptvqa*.zip" \) | head -n 1 || true)
    if [ -n "$folder_zip" ]; then
        echo "Extracting dataset zip from shared folder..."
        unzip -q "$folder_zip" -d data
        rm -rf "$folder_tmp"
        return 0
    fi

    if [ -d "$folder_tmp/ReceiptVQA-Dataset" ]; then
        echo "Moving ReceiptVQA-Dataset from shared folder..."
        rm -rf data/ReceiptVQA-Dataset
        mv "$folder_tmp/ReceiptVQA-Dataset" data/
        rm -rf "$folder_tmp"
        return 0
    fi

    rm -rf "$folder_tmp"
    return 1
}

extract_nested_archives() {
    local base="$1"

    normalize_folder_layout "$base"

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
    echo "No local ReceiptVQA zip found."
    if download_from_drive; then
        echo "Google Drive download complete."
    else
        echo "Trying Kaggle API..."
        if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
            echo "Missing Kaggle credentials: $HOME/.kaggle/kaggle.json"
            echo "Google Drive file default: $DEFAULT_DRIVE_FILE_URL"
            echo "Google Drive folder default: $DEFAULT_DRIVE_FOLDER_URL"
            echo "Or place ReceiptVQA-Dataset zip in repo root."
            exit 1
        fi
        pip install -q kaggle
        kaggle datasets download -d anlgyn/receiptvqa -p data/
        unzip -q data/receiptvqa.zip -d data
        rm -f data/receiptvqa.zip
    fi
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
