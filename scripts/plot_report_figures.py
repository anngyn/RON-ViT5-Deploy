"""Generate report figures: answer-type distribution, noise-drop rankings, method comparison.

Outputs PNGs under outputs/results/ for the final report.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def find_test_csv():
    for cand in [
        'data/ReceiptVQA-Dataset/ReceiptVQA_annotations/ReceiptVQA_annotations/ReceiptVQA_test.csv',
        'data/ReceiptVQA-Dataset/ReceiptVQA_annotations/ReceiptVQA_test.csv',
    ]:
        p = resolve(cand)
        if os.path.exists(p):
            return p
    return None


MONEY_RE = re.compile(r'\d[\d.,]*\s*(đ|d|vnd|vnđ)\b', re.IGNORECASE)
DATE_RE = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
PHONE_RE = re.compile(r'\d[\d.\s]{6,}\d')
NUM_RE = re.compile(r'^\s*[\d.,\s]+\s*$')


def classify_answer(ans):
    a = str(ans).strip()
    if MONEY_RE.search(a):
        return 'money'
    if DATE_RE.search(a):
        return 'date'
    if NUM_RE.match(a) or PHONE_RE.search(a):
        return 'number/phone'
    return 'text'


def fig_answer_distribution(plt):
    csv = find_test_csv()
    if not csv:
        print('test csv not found; skip answer distribution')
        return
    df = pd.read_csv(csv)
    df['atype'] = df['answer'].apply(classify_answer)
    counts = df['atype'].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#d62728', '#9467bd', '#1f77b4', '#2ca02c']
    ax1.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors[:len(counts)], startangle=90)
    ax1.set_title('Answer Type Distribution (test set)')

    # answer length histogram
    df['ans_len'] = df['answer'].astype(str).str.len()
    ax2.hist(df['ans_len'].clip(upper=60), bins=30, color='#1f77b4', alpha=0.8)
    ax2.set_title('Answer Length Distribution')
    ax2.set_xlabel('Answer length (chars, clipped @60)')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = resolve('outputs/results/fig_answer_distribution.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')
    return counts


def load_flows():
    f1 = pd.read_csv(resolve('outputs/results/flow1_vit_clean_noise_l2.csv'))
    f2 = pd.read_csv(resolve('outputs/results/flow2_vit_aug_noise_l2.csv'))
    f3 = pd.read_csv(resolve('outputs/results/flow3_consistency_noise_l2.csv'))
    return f1, f2, f3


def fig_noise_drop_ranking(plt):
    f1, _, _ = load_flows()
    d = f1[f1.condition_id != 'clean'].copy()
    d = d.sort_values('drop_from_clean')
    colors = ['#d62728' if v > 0.03 else '#ff7f0e' if v > 0.015 else '#2ca02c'
              for v in d['drop_from_clean']]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(d['noise_type'], d['drop_from_clean'] * 100, color=colors)
    ax.set_title('Noise Impact Ranking — ANLS Drop from Clean (Baseline, L2)')
    ax.set_xlabel('Drop from clean (ANLS points)')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    for i, v in enumerate(d['drop_from_clean'] * 100):
        ax.text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=8)
    plt.tight_layout()
    out = resolve('outputs/results/fig_noise_drop_ranking.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')


def fig_recovery(plt):
    f1, f2, f3 = load_flows()
    m = f1[['condition_id', 'noise_type', 'anls']].rename(columns={'anls': 'base'})
    m = m.merge(f2[['condition_id', 'anls']].rename(columns={'anls': 'aug'}), on='condition_id')
    m = m.merge(f3[['condition_id', 'anls']].rename(columns={'anls': 'consist'}), on='condition_id')
    m = m[m.condition_id != 'clean'].copy()
    m['aug_gain'] = (m['aug'] - m['base']) * 100
    m['consist_gain'] = (m['consist'] - m['base']) * 100
    m = m.sort_values('aug_gain', ascending=True)

    y = np.arange(len(m))
    h = 0.38
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y + h/2, m['aug_gain'], height=h, color='#ff7f0e', label='Noisy Aug')
    ax.barh(y - h/2, m['consist_gain'], height=h, color='#1f77b4', label='Consistency')
    ax.set_yticks(y)
    ax.set_yticklabels(m['noise_type'])
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_title('Recovery per Noise Type (method − baseline, L2)')
    ax.set_xlabel('ANLS gain (points)')
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    out = resolve('outputs/results/fig_recovery.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')


def fig_method_summary(plt):
    f1, f2, f3 = load_flows()

    def stats(df, tag):
        clean = df.loc[df.condition_id == 'clean', 'anls'].values[0]
        noisy = df[df.condition_id != 'clean']['anls'].mean()
        mixed = df.loc[df.condition_id == 'N20', 'anls'].values[0]
        return {'tag': tag, 'clean': clean, 'avg_noisy': noisy, 'mixed': mixed}

    rows = [stats(f1, 'Baseline'), stats(f2, 'Noisy Aug'), stats(f3, 'Consistency')]
    s = pd.DataFrame(rows)

    x = np.arange(len(s))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, s['clean'], w, label='Clean', color='#2ca02c')
    ax.bar(x, s['avg_noisy'], w, label='Avg Noisy (14)', color='#ff7f0e')
    ax.bar(x + w, s['mixed'], w, label='Mixed (N20)', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(s['tag'])
    ax.set_ylim(0.70, 0.87)
    ax.set_ylabel('ANLS')
    ax.set_title('Method Summary: Clean vs Noisy Robustness')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for i, row in s.iterrows():
        ax.text(i - w, row['clean'] + 0.002, f"{row['clean']:.3f}", ha='center', fontsize=8)
        ax.text(i, row['avg_noisy'] + 0.002, f"{row['avg_noisy']:.3f}", ha='center', fontsize=8)
        ax.text(i + w, row['mixed'] + 0.002, f"{row['mixed']:.3f}", ha='center', fontsize=8)
    plt.tight_layout()
    out = resolve('outputs/results/fig_method_summary.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('matplotlib required') from exc

    fig_answer_distribution(plt)
    fig_noise_drop_ranking(plt)
    fig_recovery(plt)
    fig_method_summary(plt)


if __name__ == '__main__':
    main()
