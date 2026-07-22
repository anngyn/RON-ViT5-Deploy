"""Do phan phoi nhieu tren du lieu that + noi voi ket qua ANLS da chay.

Sinh 3 bieu do:
  1. fig_noise_answer_hit.png   - ty le nhieu lam hong answer (exposure thuc te)
  2. fig_noise_perturbation.png - phan phoi muc bien dang context theo tung loai
  3. fig_exposure_vs_drop.png   - tuong quan answer-hit-rate vs drop da do

Chay:
    python scripts/analyze_noise_distribution.py --subset-ratio 0.05 --level 2
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_data
from src.noise import OCRNoiseGenerator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")

NOISE_TYPES = [
    "accent_removal", "tone_confusion", "vowel_diacritic_confusion", "dd_confusion",
    "character_confusion", "glyph_confusion", "character_deletion", "token_deletion",
    "line_shuffle", "token_split", "money_noise", "date_noise", "code_noise", "mixed_noise",
]
SHORT = {
    "accent_removal": "accent", "tone_confusion": "tone",
    "vowel_diacritic_confusion": "vowel", "dd_confusion": "dd",
    "character_confusion": "char_conf", "glyph_confusion": "glyph",
    "character_deletion": "char_del", "token_deletion": "token_del",
    "line_shuffle": "line_shuf", "token_split": "token_split",
    "money_noise": "money", "date_noise": "date",
    "code_noise": "code", "mixed_noise": "mixed",
}


def token_change_rate(clean, noisy):
    """Ty le token cua ban sach khong con nguyen ven trong ban nhieu."""
    c, n = clean.split(), noisy.split()
    if not c:
        return 0.0
    from collections import Counter
    cc, nc = Counter(c), Counter(n)
    kept = sum(min(v, nc[k]) for k, v in cc.items())
    return 1.0 - kept / len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "baseline.yaml"))
    ap.add_argument("--subset-ratio", type=float, default=0.05)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=1500)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_dir = config["data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)

    (_, _), (_, _), (test_qa, test_ocr) = load_data(data_dir, args.subset_ratio)
    df = pd.merge(test_qa, test_ocr[["image_id", "texts"]], on="image_id", how="inner")
    df = df.head(args.max_samples).reset_index(drop=True)
    print("So mau phan tich:", len(df))

    gen = OCRNoiseGenerator(seed=42)
    rows, perturb = [], {}

    for ntype in NOISE_TYPES:
        hits = 0          # answer bi hong
        eligible = 0      # answer von xuat hien trong context sach
        rates = []
        for _, r in df.iterrows():
            clean = " ".join(str(t) for t in r["texts"])
            answer = str(r["answer"]).strip()
            noisy = gen.apply_noise(clean, ntype, args.level)
            rates.append(token_change_rate(clean, noisy))
            if answer and answer.lower() in clean.lower():
                eligible += 1
                if answer.lower() not in noisy.lower():
                    hits += 1
        perturb[SHORT[ntype]] = rates
        rows.append({
            "noise_type": ntype,
            "short": SHORT[ntype],
            "answer_hit_rate": hits / eligible if eligible else float("nan"),
            "mean_token_change": float(np.mean(rates)),
            "eligible": eligible,
        })

    stats = pd.DataFrame(rows).sort_values("answer_hit_rate", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "noise_distribution_stats.csv")
    stats.to_csv(csv_path, index=False)
    print(stats.to_string(index=False))
    print("Da luu:", csv_path)

    # 1. Ty le lam hong answer
    plt.figure(figsize=(10, 5))
    plt.bar(stats["short"], stats["answer_hit_rate"] * 100, color="#eb6834")
    plt.ylabel("% mau bi hong answer")
    plt.title("Nhieu lam hong answer bao nhieu (exposure thuc te)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_noise_answer_hit.png"), dpi=150)
    plt.close()

    # 2. Phan phoi muc bien dang context
    order = sorted(perturb, key=lambda k: np.mean(perturb[k]), reverse=True)
    plt.figure(figsize=(10, 5))
    plt.boxplot([perturb[k] for k in order], labels=order, showfliers=False)
    plt.ylabel("Ty le token bi thay doi")
    plt.title("Phan phoi muc bien dang context theo loai nhieu (L%d)" % args.level)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_noise_perturbation.png"), dpi=150)
    plt.close()

    # 3. Doi chieu voi drop da do (neu co flow1 CSV)
    flow1 = os.path.join(RESULTS_DIR, "flow1_vit_clean_noise_l2.csv")
    if os.path.exists(flow1):
        meas = pd.read_csv(flow1)
        meas = meas[meas["condition_id"] != "clean"][["noise_type", "drop_from_clean"]]
        m = stats.merge(meas, on="noise_type", how="inner")
        m["drop_pts"] = m["drop_from_clean"] * 100
        plt.figure(figsize=(7, 6))
        plt.scatter(m["answer_hit_rate"] * 100, m["drop_pts"], s=60, color="#2a78d6")
        for _, r in m.iterrows():
            plt.annotate(r["short"], (r["answer_hit_rate"] * 100, r["drop_pts"]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
        if len(m) > 2:
            corr = m["answer_hit_rate"].corr(m["drop_pts"])
            plt.title("Exposure vs thiet hai (r = %.2f)" % corr)
            print("Tuong quan answer_hit_rate vs drop: r = %.3f" % corr)
        plt.xlabel("% mau bi hong answer")
        plt.ylabel("Giam ANLS (diem x100)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "fig_exposure_vs_drop.png"), dpi=150)
        plt.close()
    else:
        print("Chua co flow1 CSV -> bo qua bieu do tuong quan.")

    print("Xong. Xem cac file trong outputs/results/")


if __name__ == "__main__":
    main()
