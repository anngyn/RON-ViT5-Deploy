# Quick Start: Baseline + Noisy Aug Only

Train 2 methods đầu trước (baseline và noisy aug) - ~2 giờ total.

## 1. Setup Vast.ai

```bash
# Search RTX 5070 Ti / 4070 Ti (16GB, cheapest)
vastai search offers 'gpu_ram>=16 reliability>0.95 disk_space>50' --order 'dph_total+'

# Rent (replace <ID> with cheapest offer)
vastai create instance <ID> \
  --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
  --disk 60 \
  --direct

# SSH
vastai ssh-url <INSTANCE_ID>
ssh root@<IP> -p <PORT>
```

**Cost:** ~$0.22/h × 2h = **$0.44 total**

## 2. Clone & Install

```bash
# Update & clone
apt-get update && apt-get install -y git wget
git clone https://github.com/YOUR_USERNAME/RON-ViT5.git
cd RON-ViT5

# Install dependencies
bash setup.sh
```

## 3. Download Dataset

**Option A: Interactive script (easiest)**
```bash
bash download_dataset.sh
# Select option 1 (Kaggle) or 2 (Google Drive)
# Follow prompts
```

**Option B: Kaggle API (manual)**
```bash
# Install kaggle
pip install kaggle

# Setup credentials
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# Paste: {"username":"YOUR_USERNAME","key":"YOUR_KEY"}
# Get from: https://www.kaggle.com/settings (API section)
# Save: Ctrl+O, Enter, Ctrl+X

chmod 600 ~/.kaggle/kaggle.json

# Download & extract
kaggle datasets download -d anlgyn/receiptvqa -p data/
cd data && unzip receiptvqa.zip && cd ..
```

## 4. Verify Setup

```bash
# Check GPU
nvidia-smi

# Check dataset
ls -lh data/ReceiptVQA-Dataset/ReceiptVQA_annotations/
ls -lh data/ReceiptVQA-Dataset/features/google_ocr/google_ocr/ | head

# Expected output:
# -rw-r--r-- 1 root root  5.2M  ReceiptVQA_train.csv
# -rw-r--r-- 1 root root  1.5K  1.npy
```

## 5. Run Training

```bash
# Train baseline + noisy aug (~2 hours)
bash run_baseline_and_noisy.sh
```

### Monitor Progress

Open another SSH session:

```bash
# GPU utilization
watch -n 5 nvidia-smi

# Training logs
tail -f outputs/logs/baseline.log

# Current status
ls -lh outputs/models/
```

### Expected Output

```
Epoch 1/3: 100%|██████████| 1299/1299 [23:45<00:00, loss=2.1234]
Dev ANLS: 0.6520
✓ Best model saved (ANLS: 0.6520)

Epoch 2/3: 100%|██████████| 1299/1299 [23:42<00:00, loss=1.8765]
Dev ANLS: 0.7015
✓ Best model saved (ANLS: 0.7015)

Epoch 3/3: 100%|██████████| 1299/1299 [23:40<00:00, loss=1.6543]
Dev ANLS: 0.7233
✓ Best model saved (ANLS: 0.7233)
```

## 6. Check Results

```bash
# View results
cat outputs/results/baseline_results.csv
cat outputs/results/noisy_aug_results.csv

# Compare
grep "clean" outputs/results/*.csv
grep "mixed" outputs/results/*.csv
```

### Expected Results

**Baseline:**
```csv
noise_type,anls
clean,0.7233
mixed,0.6810
char,0.6950
money,0.7100
date,0.7050
accent,0.6500
```

**Noisy Aug:**
```csv
noise_type,anls
clean,0.7250         # Similar to baseline
mixed,0.7020         # +0.02 improvement
char,0.7100          # +0.015
money,0.7200         # +0.01
date,0.7180          # +0.013
accent,0.6650        # +0.015
```

**Improvement:** Noisy aug giảm performance drop trên noisy test ~20-30%.

## 7. Download Results

```bash
# On local machine
scp -P <PORT> -r root@<IP>:~/RON-ViT5/outputs ./

# Results in:
#   outputs/results/baseline_results.csv
#   outputs/results/noisy_aug_results.csv
#   outputs/models/baseline/
#   outputs/models/noisy_aug/
```

## 8. Stop Instance

```bash
# Exit SSH
exit

# Stop billing
vastai stop instance <INSTANCE_ID>

# Check cost
vastai show instance <INSTANCE_ID>
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
nano configs/baseline.yaml
# Change: batch_size: 4  (default 8)

nano configs/noisy_aug.yaml
# Change: batch_size: 4
```

### Training hangs

```bash
# Check GPU
nvidia-smi

# Kill and restart
pkill -f python
bash run_baseline_and_noisy.sh
```

### Dataset not found

```bash
# Check extraction
ls -lh data/
unzip -l data/receiptvqa.zip | head

# Re-extract
rm -rf data/ReceiptVQA-Dataset
unzip data/receiptvqa.zip -d data/
```

## Timeline

| Step | Time | Cumulative |
|------|------|------------|
| Setup | 5 min | 5 min |
| Download dataset | 3 min | 8 min |
| Baseline training | 60 min | 68 min |
| Noisy Aug training | 70 min | 138 min |
| **Total** | **~2.3h** | **~2.3h** |

**Cost:** $0.22/h × 2.3h = **$0.51**

## Next Steps

### If results good → Run full pipeline

```bash
# Continue with remaining 3 methods (~4 hours more)
bash run_all.sh
```

### If results bad → Debug first

```bash
# Check sample predictions
python -c "
import pandas as pd
df = pd.read_csv('outputs/logs/baseline.log')
print(df.head(20))
"

# Re-train with smaller subset
nano configs/baseline.yaml
# Change: subset_ratio: 0.1  (10% instead of 20%)
```

## Quick Commands Reference

```bash
# Search GPU
vastai search offers 'gpu_ram>=16 reliability>0.95' --order 'dph_total+'

# Rent
vastai create instance <ID> --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# SSH
vastai ssh-url <ID>

# Monitor
nvidia-smi
tail -f outputs/logs/*.log

# Stop
vastai stop instance <ID>
```
