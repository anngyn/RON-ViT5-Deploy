#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RECEIPTVQA_DRIVE_URL="${RECEIPTVQA_DRIVE_URL:-https://drive.google.com/file/d/1-LJyQ27HNmfCCuE1AAnwF2WYm84VlWPS/view?usp=drive_link}"
export RECEIPTVQA_DRIVE_ID="${RECEIPTVQA_DRIVE_ID:-1-LJyQ27HNmfCCuE1AAnwF2WYm84VlWPS}"
export RECEIPTVQA_DRIVE_FOLDER_URL="${RECEIPTVQA_DRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1uw7v_tVAaszkDSvt5MdtpgWAImoy4Tq1?usp=drive_link}"
export RECEIPTVQA_DRIVE_FOLDER_ID="${RECEIPTVQA_DRIVE_FOLDER_ID:-1uw7v_tVAaszkDSvt5MdtpgWAImoy4Tq1}"
exec bash "$SCRIPT_DIR/download_dataset.sh"
