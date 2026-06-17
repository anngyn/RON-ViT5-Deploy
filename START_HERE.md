# RON-ViT5 Training - Start Here

Complete training pipeline cho Vietnamese ReceiptVQA với OCR noise robustness.

## Quick Links

- **First time?** → Read [QUICKSTART.md](QUICKSTART.md) (chạy 2 methods đầu - 2h)
- **Full pipeline?** → Read [README.md](README.md) (chạy 5 methods - 6h)
- **16GB GPU?** → Read [VAST_AI_16GB.md](VAST_AI_16GB.md)
- **Detailed setup?** → Read [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)

## File Structure

```
RON-ViT5-Deploy/
├── START_HERE.md              ← You are here
├── QUICKSTART.md              ← Baseline + Noisy Aug only (2h, $0.5)
├── README.md                  ← Full documentation
├── DEPLOY_GUIDE.md            ← Detailed GPU rental guide
├── VAST_AI_16GB.md            ← 16GB GPU specific config
│
├── requirements.txt           ← Dependencies
├── setup.sh                   ← Installation script
│
├── run_baseline_and_noisy.sh ← Run 2 methods (recommended first)
├── run_all.sh                 ← Run all 5 methods
├── run_all_16gb.sh            ← Run all 5 methods (16GB GPU)
│
├── configs/                   ← Training configs
│   ├── baseline.yaml
│   ├── noisy_aug.yaml
│   ├── adapter.yaml
│   ├── consistency.yaml
│   ├── consistency_16gb.yaml
│   ├── ron_naca.yaml
│   └── ron_naca_16gb.yaml
│
├── scripts/                   ← Training scripts
│   ├── train_baseline.py
│   └── train_noisy_aug.py
│
└── src/                       ← Core modules
    ├── dataset.py
    ├── models.py
    ├── noise.py
    ├── train.py
    └── evaluate.py
```

## 5 Methods

1. **Baseline** - ViT5 on clean data only
2. **Noisy Aug** - ViT5 + noisy augmentation
3. **Adapter** - ViT5 + bottleneck adapter (no consistency)
4. **Consistency** - ViT5 + consistency loss (no adapter)
5. **RON-NACA** - Adapter + Consistency (full method)

## Recommended Workflow

### Step 1: Test with 2 methods first (Baseline + Noisy Aug)

**Why:** Validate pipeline, check results quality, only 2h

```bash
# Follow QUICKSTART.md
bash run_baseline_and_noisy.sh
```

**Cost:** ~$0.50 on RTX 5070 Ti

### Step 2: If results good → Run full pipeline

```bash
# Follow README.md
bash run_all.sh         # 24GB+ GPU
# OR
bash run_all_16gb.sh    # 16GB GPU
```

**Cost:** ~$1.50 on RTX 5070 Ti

## Prerequisites

1. **GPU rental account** (Vast.ai / Runpod / Colab Pro)
2. **Dataset**: Google Drive link (auto-download with script)
   - https://drive.google.com/file/d/1-LJyQ27HNmfCCuE1AAnwF2WYm84VlWPS
3. **Git basics** (clone, commit)

## Quick Start (TL;DR)

```bash
# 1. Rent GPU (Vast.ai - cheapest)
vastai search offers 'gpu_ram>=16 reliability>0.95' --order 'dph_total+'
vastai create instance <ID> --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 2. SSH & setup
vastai ssh-url <ID>
git clone https://github.com/YOUR_REPO/RON-ViT5-Deploy.git
cd RON-ViT5-Deploy
bash setup.sh

# 3. Download dataset (auto from Drive)
bash download_dataset.sh

# 4. Run training
bash run_baseline_and_noisy.sh  # 2 methods (~2h)
# OR
bash run_all.sh                  # 5 methods (~6h)

# 5. Download results
# On local: scp -r root@<IP>:~/RON-ViT5-Deploy/outputs ./
```

## GPU Recommendations

| GPU | VRAM | Time | Cost | Config |
|-----|------|------|------|--------|
| RTX 5070 Ti | 16GB | 2.5h | $0.55 | run_baseline_and_noisy.sh |
| RTX 4090 | 24GB | 2h | $0.60 | run_baseline_and_noisy.sh |
| A100 | 40GB | 1.5h | $1.50 | run_baseline_and_noisy.sh |

Full pipeline (5 methods): multiply by 3x

## Expected Results

**Baseline (clean test):** ANLS ~0.72  
**Noisy Aug (mixed noise):** ANLS ~0.70 (+0.02 vs baseline)  
**RON-NACA (mixed noise):** ANLS ~0.72 (+0.04 vs baseline)

## Support

Issues? Check:
1. `outputs/logs/*.log` - training logs
2. `nvidia-smi` - GPU status
3. Dataset structure: `ls -lh data/ReceiptVQA-Dataset/`

## Next Steps

→ Open [QUICKSTART.md](QUICKSTART.md) to begin
