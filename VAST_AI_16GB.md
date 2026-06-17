# Vast.ai Setup for 16GB GPU (RTX 5070 Ti)

## Search & Rent

```bash
# Install vast CLI
pip install vastai

# Login
vastai set api-key YOUR_API_KEY

# Search RTX 5070 Ti / RTX 4070 Ti (16GB)
vastai search offers 'gpu_name=RTX_5070_Ti num_gpus=1 reliability>0.95 disk_space>50' --order 'dph_total+'

# Or RTX 4070 Ti (same 16GB)
vastai search offers 'gpu_name=RTX_4070_Ti num_gpus=1 reliability>0.95 disk_space>50' --order 'dph_total+'

# Rent cheapest (replace 12345 with offer ID)
vastai create instance 12345 \
  --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
  --disk 60 \
  --direct
```

**Expected cost:** $0.20-0.25/h × 6h = **$1.20-1.50 total**

## Setup

```bash
# SSH
vastai ssh-url INSTANCE_ID
ssh root@<IP> -p <PORT>

# Clone & install
apt-get update && apt-get install -y git wget
git clone https://github.com/YOUR_USERNAME/RON-ViT5.git
cd RON-ViT5
bash setup.sh

# Download dataset (Kaggle API)
pip install kaggle
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json  # Paste token
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d anlgyn/receiptvqa -p data/
unzip data/receiptvqa.zip -d data/
```

## Run Training (16GB-optimized)

```bash
# Use 16GB script (batch=4 for consistency/ron_naca)
bash run_all_16gb.sh
```

**Time:** ~6-7h (slower due to smaller batch)

## Monitor

```bash
# GPU usage (should stay under 15GB)
watch -n 5 nvidia-smi

# Training logs
tail -f outputs/logs/consistency_only.log
```

## Expected Memory Usage

```
Epoch 1/3: 100%|██████████| 1299/1299 [12:34<00:00, loss=2.3456]
GPU Memory: 6.8 GB / 16.0 GB  ✓ Safe
```

If OOM still happens:
```bash
# Emergency: reduce batch to 2
nano configs/consistency_16gb.yaml
# Change: batch_size: 2
```

## Stop Instance

```bash
# After training finishes (~7h)
exit  # Exit SSH

# Stop billing
vastai stop instance INSTANCE_ID
```

## Cost Breakdown

| GPU | $/hour | 7h Total | vs 4090 |
|-----|--------|----------|---------|
| RTX 5070 Ti | $0.22 | $1.54 | 14% cheaper |
| RTX 4070 Ti | $0.25 | $1.75 | Same price |
| RTX 4090 | $0.30 | $2.10 | Baseline |

**Result:** RTX 5070 Ti with batch=4 **saves $0.56** vs RTX 4090 batch=8, but **1h slower**.

## Trade-off

- **Pros:** Cheaper ($1.54 vs $2.10)
- **Cons:** Slower (7h vs 6h), requires config tweaks
- **Verdict:** Good choice if budget-conscious, OK với thêm 1h

## Quick Commands

```bash
# Search
vastai search offers 'gpu_name=RTX_5070_Ti reliability>0.95' --order 'dph_total+'

# Rent
vastai create instance <ID> --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# SSH
vastai ssh-url <ID>

# Run (after setup)
bash run_all_16gb.sh

# Monitor
nvidia-smi
tail -f outputs/logs/*.log

# Stop
vastai stop instance <ID>
```
