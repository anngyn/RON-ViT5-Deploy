# RON-ViT5-Deploy Contents

Complete deployment package - 23 files total.

## Documentation (5 files)

| File | Purpose | Read When |
|------|---------|-----------|
| **START_HERE.md** | Entry point, navigation | First time |
| **QUICKSTART.md** | Run 2 methods (2h, $0.5) | Want quick test |
| **README.md** | Full docs, all 5 methods | Want complete pipeline |
| **DEPLOY_GUIDE.md** | Detailed GPU rental steps | Need setup help |
| **VAST_AI_16GB.md** | 16GB GPU specific guide | Using RTX 5070Ti/4070Ti |

## Scripts (3 files)

| File | Purpose | GPU Time | Cost |
|------|---------|----------|------|
| **run_baseline_and_noisy.sh** | Train 2 methods | 2h | $0.5 |
| **run_all.sh** | Train all 5 methods (24GB+) | 6h | $1.8 |
| **run_all_16gb.sh** | Train all 5 methods (16GB) | 7h | $1.5 |

## Configs (7 files)

| File | Method | Batch | GPU Mem |
|------|--------|-------|---------|
| baseline.yaml | ViT5 clean only | 8 | 4GB |
| noisy_aug.yaml | ViT5 + noisy aug | 8 | 5GB |
| adapter.yaml | Adapter only | 8 | 3GB |
| consistency.yaml | Consistency only | 8 | 12GB |
| consistency_16gb.yaml | Consistency (16GB) | 4 | 7GB |
| ron_naca.yaml | Full method | 8 | 10GB |
| ron_naca_16gb.yaml | Full method (16GB) | 4 | 6GB |

## Training Scripts (2 files)

| File | Trains | Uses Dataset |
|------|--------|--------------|
| scripts/train_baseline.py | Baseline | TextOnlyVQADataset |
| scripts/train_noisy_aug.py | Noisy Aug | NoisyVQADataset |

**TODO:** Add 3 more scripts for methods 3-5 (copy structure from baseline.py)

## Core Modules (5 files)

| File | Contains | LOC |
|------|----------|-----|
| src/noise.py | OCRNoiseGenerator | ~120 |
| src/dataset.py | 3 dataset classes | ~250 |
| src/models.py | Adapter, RON-NACA | ~180 |
| src/train.py | Training loops | ~150 |
| src/evaluate.py | ANLS metric | ~80 |

## Setup (1 file)

- **setup.sh** - Install deps, create dirs, download dataset instructions

## Dependencies

- **requirements.txt** - 9 packages (torch, transformers, pandas, etc)

## Output Structure (created during training)

```
outputs/
├── models/
│   ├── baseline/              # Best checkpoint
│   ├── noisy_aug/
│   ├── adapter_only/
│   ├── consistency_only/
│   └── ron_naca_full/
├── results/
│   ├── baseline_results.csv   # ANLS by noise type
│   ├── noisy_aug_results.csv
│   ├── adapter_only_results.csv
│   ├── consistency_only_results.csv
│   └── ron_naca_full_results.csv
└── logs/
    ├── baseline.log           # Full training logs
    ├── noisy_aug.log
    ├── adapter_only.log
    ├── consistency_only.log
    └── ron_naca_full.log
```

## Data Structure (user provides)

```
data/
└── ReceiptVQA-Dataset/
    ├── ReceiptVQA_annotations/
    │   └── ReceiptVQA_annotations/
    │       ├── ReceiptVQA_train.csv  (51,886 rows)
    │       ├── ReceiptVQA_dev.csv    (6,426 rows)
    │       └── ReceiptVQA_test.csv   (6,500 rows)
    └── features/
        └── google_ocr/
            └── google_ocr/
                ├── 1.npy
                ├── 2.npy
                └── ... (9,769 files)
```

## Usage Flow

```
1. Read START_HERE.md
   ↓
2. Choose path:
   - Quick test → QUICKSTART.md (2 methods)
   - Full pipeline → README.md (5 methods)
   ↓
3. Setup GPU (DEPLOY_GUIDE.md or VAST_AI_16GB.md)
   ↓
4. Run setup.sh
   ↓
5. Download dataset (Kaggle)
   ↓
6. Run training script:
   - bash run_baseline_and_noisy.sh
   - OR bash run_all.sh
   ↓
7. Monitor:
   - nvidia-smi
   - tail -f outputs/logs/*.log
   ↓
8. Download results:
   - scp -r outputs/ ./
   ↓
9. Compare ANLS scores
```

## File Dependencies

```
setup.sh
  → requirements.txt

run_baseline_and_noisy.sh
  → scripts/train_baseline.py
  → scripts/train_noisy_aug.py

train_baseline.py
  → configs/baseline.yaml
  → src/dataset.py (TextOnlyVQADataset)
  → src/train.py (train_epoch_standard)
  → src/evaluate.py (evaluate, compute_anls)

train_noisy_aug.py
  → configs/noisy_aug.yaml
  → src/dataset.py (NoisyVQADataset)
  → src/noise.py (OCRNoiseGenerator)
  → src/train.py (train_epoch_standard)
  → src/evaluate.py (evaluate, compute_anls)

src/dataset.py
  → src/noise.py (for NoisyVQADataset)

src/models.py
  → transformers.T5ForConditionalGeneration
```

## Completeness Status

✅ **Ready to use:**
- Documentation (5/5)
- Setup scripts (1/1)
- Configs (7/7)
- Core modules (5/5)
- Training scripts for 2 methods (2/5)

⚠️ **TODO:**
- Training scripts for methods 3-5 (copy from train_baseline.py structure)
  - scripts/train_adapter.py
  - scripts/train_consistency.py
  - scripts/train_ron_naca.py

## Total Size

- Source code: ~1,000 LOC
- Configs: ~150 lines
- Docs: ~2,000 lines
- **Total package:** ~3,200 lines

**Without dataset/outputs:** ~100 KB  
**With dataset:** ~2.5 GB  
**After training:** ~4 GB (models + outputs)

## License

(Add your license here)

## Citation

```bibtex
@inproceedings{ron-vit5-2026,
  title={RON-NACA: Robust OCR Noise-Aware Consistency Adapter for Vietnamese Receipt QA},
  author={Your Name},
  year={2026}
}
```
