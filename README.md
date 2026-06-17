# RON-ViT5: Robust OCR Noise-Aware Consistency Adapter

Training pipeline cho Vietnamese ReceiptVQA với 5 methods:
1. **Baseline**: ViT5 on clean data
2. **Noisy Aug**: ViT5 + noisy augmentation
3. **Adapter Only**: ViT5 + bottleneck adapter (no consistency)
4. **Consistency Only**: ViT5 + consistency loss (no adapter)
5. **RON-NACA**: Adapter + Consistency (full method)

## Setup

### 1. Clone & Install
```bash
git clone <repo>
cd RON-ViT5
bash setup.sh
```

### 2. Upload Dataset
Download ReceiptVQA từ Kaggle: `anlgyn/receiptvqa`

Extract vào:
```
data/
└── ReceiptVQA-Dataset/
    ├── ReceiptVQA_annotations/
    │   └── ReceiptVQA_annotations/
    │       ├── ReceiptVQA_train.csv
    │       ├── ReceiptVQA_dev.csv
    │       └── ReceiptVQA_test.csv
    └── features/
        └── google_ocr/
            └── google_ocr/
                ├── 1.npy
                ├── 2.npy
                └── ...
```

## Training

### Run All Methods (Sequential)
```bash
bash run_all.sh
```

**Time:** ~5-6 giờ total on A100 (3 epochs × 5 methods)

**GPU Memory:** ~12-16GB

### Run Individual Methods
```bash
python scripts/train_baseline.py      # Method 1
python scripts/train_noisy_aug.py     # Method 2
python scripts/train_adapter.py       # Method 3
python scripts/train_consistency.py   # Method 4
python scripts/train_ron_naca.py      # Method 5
```

## GPU Rental Options

### Recommended: Vast.ai (cheapest)
```bash
# Search RTX 4090 instances
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95 disk_space>50'

# Rent instance (replace INSTANCE_ID)
vastai create instance INSTANCE_ID --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime --disk 60

# SSH into instance
vastai ssh-url INSTANCE_ID

# Clone & run
git clone <repo>
cd RON-ViT5
bash setup.sh
# Upload dataset to ./data/
bash run_all.sh
```

**Cost:** RTX 4090 ~$0.3/h × 6h = ~$1.8 total

### Alternative: Colab Pro
Upload notebooks to Google Drive, attach to Colab Pro with A100.

**Cost:** $10/month subscription

### Alternative: Runpod
Similar to Vast.ai, GUI-based.

**Cost:** A100 ~$1/h × 6h = ~$6 total

## Output Structure

```
outputs/
├── models/
│   ├── baseline/
│   ├── noisy_aug/
│   ├── adapter_only/
│   ├── consistency_only/
│   └── ron_naca_full/
├── results/
│   ├── baseline_results.csv
│   ├── noisy_aug_results.csv
│   ├── adapter_only_results.csv
│   ├── consistency_only_results.csv
│   └── ron_naca_full_results.csv
└── logs/
    ├── baseline.log
    ├── noisy_aug.log
    ├── adapter_only.log
    ├── consistency_only.log
    └── ron_naca_full.log
```

## Results Format

Each CSV contains:
```csv
noise_type,anls
clean,0.7233
mixed,0.6810
char,0.6950
money,0.7100
date,0.7050
accent,0.6500
```

## Configuration

Edit `configs/*.yaml` to change:
- `batch_size`: GPU memory tradeoff
- `learning_rate`: training speed
- `num_epochs`: training duration
- `subset_ratio`: dataset size (default 0.2 = 20%)

## Troubleshooting

**Out of Memory:**
```yaml
# Reduce batch size in config
batch_size: 4  # instead of 8
```

**Dataset not found:**
```bash
# Check paths
ls -lh data/ReceiptVQA-Dataset/
```

**Slow training:**
```bash
# Check GPU utilization
nvidia-smi
```

## Citation

Based on LiGT_VQA repository:
```
@Article{Le2025,
  author={Le, Thanh-Phong and Phan, Trung Le Chi and Nguyen, Nghia Hieu and Van Nguyen, Kiet},
  title={LiGT: layout-infused generative transformer for visual question answering on Vietnamese receipts},
  journal={International Journal on Document Analysis and Recognition (IJDAR)},
  year={2025},
  doi={10.1007/s10032-025-00515-z}
}
```
