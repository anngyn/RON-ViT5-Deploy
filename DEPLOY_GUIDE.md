# Deployment Guide: Training RON-ViT5 on Rented GPU

Quick guide để train RON-ViT5 trên GPU thuê (Vast.ai/Runpod/Colab).

## Prerequisites

- Kaggle account (download dataset)
- Vast.ai account (hoặc Runpod/Colab Pro)
- Git/SSH basic knowledge

---

## Option 1: Vast.ai (Cheapest - $1.8 total)

### Step 1: Search & Rent GPU

```bash
# Install vast CLI
pip install vastai

# Login
vastai set api-key YOUR_API_KEY

# Search RTX 4090 with 50GB disk
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95 disk_space>50' --order 'dph_total+'

# Rent (replace 12345 with cheapest offer ID)
vastai create instance 12345 \
  --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
  --disk 60 \
  --direct

# Get SSH command
vastai ssh-url INSTANCE_ID
```

### Step 2: SSH & Setup

```bash
# SSH into instance (from vastai ssh-url output)
ssh root@<IP> -p <PORT>

# Update system
apt-get update && apt-get install -y git wget unzip

# Clone repo
git clone https://github.com/YOUR_USERNAME/RON-ViT5.git
cd RON-ViT5

# Install dependencies
bash setup.sh
```

### Step 3: Upload Dataset

**Method A: Kaggle API (recommended)**
```bash
# Install kaggle CLI
pip install kaggle

# Upload kaggle.json (from Kaggle Account -> API -> Create New Token)
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# Paste token, save (Ctrl+O, Enter, Ctrl+X)
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d anlgyn/receiptvqa -p data/
unzip data/receiptvqa.zip -d data/
```

**Method B: Upload from local**
```bash
# On local machine (Windows)
scp -P <PORT> -r "path/to/ReceiptVQA-Dataset" root@<IP>:~/RON-ViT5/data/

# Or use SFTP client like WinSCP
```

### Step 4: Run Training

```bash
# Verify dataset
ls -lh data/ReceiptVQA-Dataset/

# Run all methods (5-6 hours)
bash run_all.sh

# Monitor progress (in another SSH session)
tail -f outputs/logs/baseline.log
watch -n 5 nvidia-smi  # GPU utilization
```

### Step 5: Download Results

```bash
# On local machine
scp -P <PORT> -r root@<IP>:~/RON-ViT5/outputs ./

# Results in:
#   outputs/results/*.csv
#   outputs/models/*
```

### Step 6: Stop Instance

```bash
# Stop billing
vastai stop instance INSTANCE_ID

# Or destroy completely
vastai destroy instance INSTANCE_ID
```

---

## Option 2: Runpod (GUI-based - $6 total)

### Step 1: Create Instance
1. Go to [runpod.io](https://runpod.io)
2. Select GPU: A100 (80GB) or RTX 4090
3. Template: PyTorch 2.0
4. Disk: 60GB
5. Deploy

### Step 2: Open Terminal
Click "Connect" → "Start Jupyter" → Open Terminal

### Step 3: Same as Vast.ai Steps 2-5

---

## Option 3: Colab Pro ($10/month)

### Step 1: Setup Colab
1. Subscribe Colab Pro
2. Upload `setup.sh`, `run_all.sh` to Google Drive
3. Create new notebook

### Step 2: Mount Drive & Setup

```python
# Cell 1: Mount drive
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/RON-ViT5

# Cell 2: Install
!bash setup.sh

# Cell 3: Download dataset (kaggle API)
!pip install kaggle
# Upload kaggle.json to Colab Files, then:
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d anlgyn/receiptvqa -p data/
!unzip data/receiptvqa.zip -d data/
```

### Step 3: Run Training

```python
# Cell 4: Train (will take 5-6 hours)
!bash run_all.sh
```

⚠️ **Colab timeout:** If session disconnects:
- Results save to `outputs/` automatically
- Re-run from checkpoint (models auto-save each epoch)

---

## Monitoring Training

### Check GPU Usage
```bash
watch -n 5 nvidia-smi
```

Expected:
- GPU Util: 80-95%
- Memory: 12-16GB / 24GB (RTX 4090) or 16GB / 40GB (A100)

### Check Progress
```bash
# Current epoch
tail -f outputs/logs/baseline.log

# All methods progress
ls -lh outputs/results/
```

### Estimated Times (RTX 4090)
- Baseline: 1.0h
- Noisy Aug: 1.2h (2x data)
- Adapter: 1.0h
- Consistency: 1.3h (paired forward)
- RON-NACA: 1.2h

**Total:** ~5.7 hours

---

## Troubleshooting

### Out of Memory
```bash
# Edit config before running
nano configs/baseline.yaml
# Change: batch_size: 4  (default 8)
```

### Dataset not found
```bash
# Verify structure
ls -lh data/ReceiptVQA-Dataset/ReceiptVQA_annotations/
ls -lh data/ReceiptVQA-Dataset/features/google_ocr/google_ocr/
```

### Training hangs
```bash
# Check GPU
nvidia-smi

# Kill and restart
pkill -f python
bash run_all.sh
```

### SSH disconnected (Vast.ai)
```bash
# Reconnect
vastai ssh-url INSTANCE_ID

# Check if training still running
ps aux | grep python

# Resume logs
tail -f outputs/logs/*.log
```

---

## Cost Comparison

| Platform | GPU | $/hour | 6h Total | Setup |
|----------|-----|--------|----------|-------|
| Vast.ai | RTX 4090 | $0.30 | **$1.80** | CLI |
| Vast.ai | A100 | $0.80 | $4.80 | CLI |
| Runpod | RTX 4090 | $0.50 | $3.00 | GUI |
| Runpod | A100 | $1.00 | **$6.00** | GUI |
| Colab Pro | A100 | $10/mo | **$10** | Web |

**Recommendation:** Vast.ai RTX 4090 nếu familiar với CLI, Runpod nếu prefer GUI.

---

## After Training

### Compare Results
```bash
cd outputs/results/
cat *_results.csv | grep "clean\|mixed"
```

Expected ANLS improvements:
- Baseline (clean): ~0.72
- Noisy Aug (mixed): +0.02-0.04
- RON-NACA (mixed): +0.04-0.06

### Create Comparison Table
```python
import pandas as pd
import glob

results = []
for f in glob.glob("outputs/results/*_results.csv"):
    method = f.split("/")[-1].replace("_results.csv", "")
    df = pd.read_csv(f)
    df['method'] = method
    results.append(df)

comparison = pd.concat(results)
pivot = comparison.pivot(index='noise_type', columns='method', values='anls')
print(pivot.round(4))
```

---

## Quick Reference

```bash
# Vast.ai commands
vastai search offers 'gpu_name=RTX_4090 reliability>0.95'
vastai create instance <ID> --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
vastai ssh-url <INSTANCE_ID>
vastai stop instance <INSTANCE_ID>

# Training
bash setup.sh               # Install deps
bash run_all.sh             # Train all (5-6h)
python scripts/train_baseline.py  # Train one method

# Monitoring
tail -f outputs/logs/*.log  # Training logs
nvidia-smi                  # GPU usage
ls -lh outputs/results/     # Check outputs
```
