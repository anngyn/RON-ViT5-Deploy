# Consistency v2: Collapse-Resistant Regularization for RON-ViT5

Design note for **Idea 2** — replacing the mean-pool cosine consistency loss with
two collapse-resistant objectives: **InfoNCE (contrastive)** and **symmetric-KL
(R-Drop)**.

## 1. Motivation

The original consistency method under-performs simple noisy augmentation. Measured
results (`outputs/results/`, single seed, ViT5-base, L2 noise):

| Model | Clean ANLS | mixed_noise ANLS |
|-------|:----------:|:----------------:|
| Baseline (clean only) | 0.8411 | 0.7422 |
| Noisy Aug | **0.8534** | **0.8062** |
| Consistency (cosine) | 0.8397 | 0.7911 |

Two facts stand out:

1. On **noisy** conditions, consistency recovers a large share of the damage —
   comparable to augmentation (e.g. money and character-confusion recover ~73% and
   ~83% of their drop). So the method is *not* weak at robustness.
2. On the **clean** test set it *loses* accuracy versus the baseline
   (0.8397 < 0.8411). Augmentation, in contrast, *gains* on clean (+0.0123).

The report attributed this to "over-regularization." This note gives the specific
mechanism and fixes it.

## 2. Diagnosis: cosine-to-1 has no repulsion term

The current loss (`src/models.py::consistency_loss`) mean-pools each encoded
sequence to one vector and maximizes cosine similarity between the clean and noisy
vectors:

```
L_cons = mean( 1 - cos( pool(h_clean), pool(h_noisy) ) )
```

This objective is minimized by **any** mapping that makes the two views identical —
including the degenerate solution where the encoder maps *every* input toward the
same region of space. There is no term that pushes different samples apart, so the
model is free to trade representational spread (which clean-set accuracy depends on)
for cheap clean/noisy alignment. This is **representation collapse**, and it is a
sufficient explanation for the clean-ANLS regression.

A second, independent weakness: pooling 256 encoder tokens into one 768-dim vector
discards token-level structure, so the signal only weakly constrains *how* the
model should read individual noisy tokens.

## 3. Fix

### 3.1 Variant A — InfoNCE (contrastive)

Add explicit negatives. Within a batch, pull the clean/noisy views of the *same*
sample together and push *different* samples apart:

```
z_c = normalize(pool(h_clean)),  z_n = normalize(pool(h_noisy))
S   = z_c @ z_nᵀ / τ                      # [B, B] similarity logits
L   = ½ ( CE(S, diag) + CE(Sᵀ, diag) )    # symmetric InfoNCE
```

The off-diagonal negatives make collapse a *high-loss* solution: if all
representations coincide, every pair is equally similar and the classification loss
is maximal. Alignment and discriminability are optimized jointly.
Implemented as `contrastive_consistency_loss` (temperature `τ = 0.1`).

### 3.2 Variant B — Symmetric KL / R-Drop

Skip the encoder pooling entirely and align the objects that actually matter: the
**decoder output distributions**. Because clean and noisy inputs share the same
answer (same teacher-forced labels), their per-position next-token distributions are
directly comparable:

```
L = mean_{valid t} ½ ( KL(p_clean ‖ p_noisy) + KL(p_noisy ‖ p_clean) )
```

masked to the answer positions (`labels != -100`). This is R-Drop / self-distillation
applied across the clean–noisy pair. It has no pooling bottleneck and cannot collapse
representations, because the target is the task distribution itself.
Implemented as `kl_consistency_loss`. Requires a model that exposes decoder logits
(plain T5 — the current `train_consistency.py` uses `T5ForConditionalGeneration`, so
this holds; the adapter wrappers do not expose logits).

### 3.3 Total objective (unchanged structure)

```
L_total = CE_clean + CE_noisy + β · L_cons
```

`consistency_type ∈ {cosine, contrastive, kl}` selects `L_cons`. `β` is raised from
0.5 to **1.0** because InfoNCE has a different scale (~log B) than `1 - cos`.

## 4. Why this should beat the cosine variant

- **Clean ANLS preserved.** Both variants remove the collapse incentive, so the
  encoder no longer pays for alignment with representational spread. Expectation:
  clean ANLS ≥ baseline (0.8411), versus 0.8397 for cosine.
- **Robustness kept.** Alignment between clean and noisy views is still enforced
  (positives in InfoNCE; matched distributions in KL), so noisy-condition recovery
  should stay at least as strong as the cosine variant.
- **KL is finer-grained.** By acting on the answer distribution, KL targets exactly
  the tokens ANLS scores, side-stepping the pooling information loss.

These are **hypotheses**, not results. They must be confirmed on this dataset (§6).

## 5. Implementation

Files changed:

- `src/models.py` — added `contrastive_consistency_loss`, `kl_consistency_loss`,
  and a `_masked_mean_pool` helper. The original `consistency_loss` is untouched.
- `src/train.py` — `train_epoch_consistency` now takes `consistency_type` and
  `temperature`, captures decoder logits, and dispatches to the selected loss.
- `scripts/train_consistency.py` — reads `consistency_type` / `temperature` from the
  config and logs the active variant.
- `configs/consistency.yaml`, `configs/consistency_16gb.yaml` — new keys
  `consistency_type: contrastive`, `temperature: 0.1`, and `beta: 1.0`.

Backward compatible: `consistency_type` defaults to `cosine`, so old configs behave
exactly as before.

## 6. Evaluation protocol

Train three consistency variants with everything else fixed (same seed, subset,
epochs), then evaluate each on the 14-type noise grid at L2:

```bash
# edit configs/consistency_16gb.yaml -> consistency_type: cosine | contrastive | kl
python scripts/train_consistency.py --config configs/consistency_16gb.yaml --skip-final-eval
python scripts/eval_noise_grid.py --config configs/consistency_16gb.yaml \
    --model-dir outputs/models/consistency_only --model-tag consistency_<variant> \
    --levels 2 --output-csv outputs/results/consistency_<variant>_noise_l2.csv
```

**Primary metric — the collapse test:** clean ANLS of each variant.

- Success if `contrastive` / `kl` clean ANLS ≥ 0.8411 (baseline) while cosine stays
  at ~0.8397. That isolates the collapse fix.

**Secondary metric:** mean drop across the 14 noise types (robustness must not
regress versus cosine).

**Ablations worth running:** `β ∈ {0.5, 1.0, 2.0}`; `τ ∈ {0.05, 0.1, 0.2}` for
InfoNCE; KL with the clean side detached (true self-distillation) vs. symmetric.

## 7. Limitations and risks

- **Single seed.** As in the base report, sub-0.005 ANLS gaps are within noise.
  Report multi-seed if a variant is to be claimed as better.
- **InfoNCE needs enough negatives.** With `batch_size 4` (16 GB config), each step
  has only 3 negatives, which is weak. The `_dp_scaled` path doubles it to 8; still
  small. If contrastive under-performs, the batch size is the first suspect — try a
  larger batch or a small memory bank of past representations.
- **KL memory.** Two `[B, 64, vocab]` logit tensors are materialized on one device;
  fine at these batch sizes but watch VRAM if the batch is scaled up.
- **KL assumes shared labels.** Valid here because `PairedVQADataset` tokenizes the
  same answer for both views. If that ever changes, the position-wise KL is no longer
  meaningful.
- These changes address the *form* of the consistency loss only. The augmentation
  distribution itself is still uniform over noise types (see the separate weighted-
  sampling / answer-anchored ideas).
