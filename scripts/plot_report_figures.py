"""Generate every figure used by RESULTS_REPORT.md.

Two groups:

1. Descriptive figures
   fig_answer_distribution, fig_noise_drop_ranking, fig_recovery, fig_method_summary

2. Effect-decomposition figures (report sections 3 and 4.1-4.4), which separate a
   method's general generalization shift (L) from its noise-specific denoising (R)
   fig_noise_floor, fig_recovery_equivalence, fig_lift_decomposition,
   fig_reducibility, fig_retention
   plus relationship_metrics.csv and a printed summary table.

Requires the three noise-grid CSVs written by eval_noise_grid.py.

Usage:
    python scripts/plot_report_figures.py
    python scripts/plot_report_figures.py --floor 0.006
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


# ---------------------------------------------------------------------------
# Effect decomposition (report sections 3 and 4.1-4.4)
#
#   L_m        = A_m(clean) - A_B(clean)      clean-set generalization shift
#   delta_m(n) = A_m(n) - A_B(n)              absolute gain over baseline
#   R_m(n)     = delta_m(n) - L_m
#              = drop_B(n) - drop_m(n)        noise-specific recovery
#   rho_m(n)   = 1 - drop_m(n)/drop_B(n)      reducibility ratio
#   eta_m(n)   = A_m(n)/A_m(clean)            retention ratio
#
# Comparing raw drop_from_clean across models is confounded: each model's drop is
# measured against its own clean ceiling, so a shorter bar does not imply a more
# robust model. These metrics remove that confound.
# ---------------------------------------------------------------------------

SHORT_NAME = {
    'accent_removal': 'accent', 'tone_confusion': 'tone',
    'vowel_diacritic_confusion': 'vowel', 'dd_confusion': 'dd',
    'character_confusion': 'char_conf', 'glyph_confusion': 'glyph',
    'character_deletion': 'char_del', 'token_deletion': 'token_del',
    'line_shuffle': 'line_shuf', 'token_split': 'token_split',
    'money_noise': 'money', 'date_noise': 'date',
    'code_noise': 'code', 'mixed_noise': 'mixed',
}
DELETION = {'char_del', 'token_del'}
DEFAULT_FLOOR = 0.006  # ANLS units; 0.006 = 0.6 points

C_BASELINE, C_AUG, C_CONSISTENCY = '#7f7f7f', '#ff7f0e', '#1f77b4'


def build_metrics(floor=DEFAULT_FLOOR):
    """Merge the three flows and derive all decomposition metrics.

    Returns (table, clean_anls_per_model, lift_per_method).
    """
    def split(df):
        clean = float(df.loc[df.condition_id == 'clean', 'anls'].values[0])
        noisy = df[df.condition_id != 'clean'][['noise_type', 'anls', 'drop_from_clean']]
        return clean, noisy

    clean, frames = {}, {}
    for tag, df in zip(('baseline', 'aug', 'consistency'), load_flows()):
        clean[tag], frames[tag] = split(df)

    t = frames['baseline'].rename(
        columns={'anls': 'anls_baseline', 'drop_from_clean': 'drop_baseline'})
    for tag in ('aug', 'consistency'):
        t = t.merge(
            frames[tag].rename(
                columns={'anls': 'anls_' + tag, 'drop_from_clean': 'drop_' + tag}),
            on='noise_type', how='inner')

    t['short'] = t['noise_type'].map(lambda s: SHORT_NAME.get(s, s))
    lift = {tag: clean[tag] - clean['baseline'] for tag in ('aug', 'consistency')}

    for tag in ('aug', 'consistency'):
        t['delta_' + tag] = t['anls_' + tag] - t['anls_baseline']
        t['R_' + tag] = t['drop_baseline'] - t['drop_' + tag]  # identity: delta - L
    for tag in ('baseline', 'aug', 'consistency'):
        t['eta_' + tag] = t['anls_' + tag] / clean[tag]

    # A condition is unreliable at one seed when the baseline effect is tiny, or
    # when the drop changes sign across the three models.
    drops = t[['drop_baseline', 'drop_aug', 'drop_consistency']]
    sign_unstable = (drops > 0).sum(axis=1).between(1, 2)
    t['below_floor'] = (t['drop_baseline'].abs() < floor) | sign_unstable

    # rho is a ratio: unstable when the denominator sits near zero.
    for tag in ('aug', 'consistency'):
        t['rho_' + tag] = np.where(
            t['below_floor'], np.nan, 1.0 - t['drop_' + tag] / t['drop_baseline'])
    return t, clean, lift


def export_metrics_csv(table):
    cols = ['noise_type', 'short', 'anls_baseline', 'anls_aug', 'anls_consistency',
            'drop_baseline', 'drop_aug', 'drop_consistency',
            'delta_aug', 'delta_consistency', 'R_aug', 'R_consistency',
            'rho_aug', 'rho_consistency',
            'eta_baseline', 'eta_aug', 'eta_consistency', 'below_floor']
    out = resolve('outputs/results/relationship_metrics.csv')
    table[cols].to_csv(out, index=False)
    print(f'Saved: {out}')


def fig_noise_floor(plt, table, floor):
    """Baseline effect sizes against the single-seed reliability band."""
    d = table.sort_values('drop_baseline', ascending=False)
    idx = np.arange(len(d))
    colors = ['#c7c7c7' if f else ('#d62728' if s in DELETION else '#1f77b4')
              for f, s in zip(d['below_floor'], d['short'])]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(idx, d['drop_baseline'] * 100, color=colors)
    ax.axhspan(-floor * 100, floor * 100, color='#e5e5e5', zorder=0,
               label=f'reliability floor (±{floor * 100:.1f} pts)')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(d['short'], rotation=45, ha='right')
    ax.set_ylabel('Baseline drop from clean (ANLS points)')
    n = int(d['below_floor'].sum())
    ax.set_title(f'Effect Size vs Reliability Floor — {n} of {len(d)} conditions '
                 f'indistinguishable from zero')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    out = resolve('outputs/results/fig_noise_floor.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')


def fig_recovery_equivalence(plt, table):
    """R_aug vs R_consistency. Points on the identity line mean the two methods
    repair the same amount of noise-specific damage."""
    d = table[~table['below_floor']]
    x, y = d['R_aug'] * 100, d['R_consistency'] * 100

    fig, ax = plt.subplots(figsize=(7.5, 7))
    lo, hi = float(min(x.min(), y.min())) - 0.5, float(max(x.max(), y.max())) + 0.5
    ax.plot([lo, hi], [lo, hi], '--', color='#999999', linewidth=1,
            label='identity (equal recovery)')
    ax.scatter(x, y, s=70, color='#9467bd', zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r['short'], (r['R_aug'] * 100, r['R_consistency'] * 100),
                    fontsize=9, xytext=(6, 4), textcoords='offset points')
    title = 'Noise-Specific Recovery: Augmentation vs Consistency'
    if len(d) > 2:
        title += f' (r = {np.corrcoef(x, y)[0, 1]:.3f})'
    ax.set_title(title)
    ax.set_xlabel('R_aug (ANLS points)')
    ax.set_ylabel('R_consistency (ANLS points)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    out = resolve('outputs/results/fig_recovery_equivalence.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')


def fig_lift_decomposition(plt, table, lift):
    """Stacked bars showing delta = L (uniform lift) + R (noise-specific recovery)."""
    d = table[~table['below_floor']].sort_values('R_aug', ascending=False)
    idx = np.arange(len(d))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tag, color in zip(axes, ('aug', 'consistency'), (C_AUG, C_CONSISTENCY)):
        L = lift[tag] * 100
        ax.bar(idx, np.full(len(idx), L), color='#c7c7c7',
               label=f'L (uniform clean-set lift) = {L:+.2f}')
        ax.bar(idx, d['R_' + tag] * 100, bottom=np.full(len(idx), L), color=color,
               label='R (noise-specific recovery)')
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_xticks(idx)
        ax.set_xticklabels(d['short'], rotation=45, ha='right')
        ax.set_title(tag)
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Gain over baseline (ANLS points)')
    fig.suptitle('Decomposition of Gain over Baseline:  delta = L + R')
    plt.tight_layout()
    out = resolve('outputs/results/fig_lift_decomposition.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')


def fig_reducibility(plt, table):
    """Drop before vs after each method. Equal bars mean the noise is irreducible."""
    d = table[~table['below_floor']].sort_values('drop_baseline', ascending=False)
    idx = np.arange(len(d))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(idx - w, d['drop_baseline'] * 100, w, color=C_BASELINE, label='baseline')
    ax.bar(idx, d['drop_aug'] * 100, w, color=C_AUG, label='after noisy aug')
    ax.bar(idx + w, d['drop_consistency'] * 100, w, color=C_CONSISTENCY,
           label='after consistency')
    ax.set_xticks(idx)
    ax.set_xticklabels(d['short'], rotation=45, ha='right')
    ax.set_ylabel('Drop from clean (ANLS points)')
    ax.set_title('Residual Damage after Robustification (equal bars = irreducible noise)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    out = resolve('outputs/results/fig_reducibility.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')


def fig_retention(plt, table):
    """Retention eta = A(n)/A(clean): robustness normalised by each model's own
    clean ceiling, so it is comparable across models."""
    d = table[~table['below_floor']].sort_values('eta_baseline')
    idx = np.arange(len(d))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(idx - w, d['eta_baseline'] * 100, w, color=C_BASELINE, label='baseline')
    ax.bar(idx, d['eta_aug'] * 100, w, color=C_AUG, label='noisy aug')
    ax.bar(idx + w, d['eta_consistency'] * 100, w, color=C_CONSISTENCY, label='consistency')
    ax.set_xticks(idx)
    ax.set_xticklabels(d['short'], rotation=45, ha='right')
    ax.set_ylim(90, 101)
    ax.set_ylabel('Retention eta = ANLS(noise) / ANLS(clean)  (%)')
    ax.set_title("Relative Robustness, Normalised by Each Model's Own Clean Ceiling")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    out = resolve('outputs/results/fig_retention.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved: {out}')


def print_decomposition_summary(table, clean, lift, floor):
    p = lambda v: v * 100
    print('\n' + '=' * 74)
    print('CLEAN-SET GENERALIZATION SHIFT (L)')
    print(f"  baseline    clean ANLS = {clean['baseline']:.4f}")
    print(f"  aug         clean ANLS = {clean['aug']:.4f}   L_aug         = {p(lift['aug']):+.2f} pts")
    print(f"  consistency clean ANLS = {clean['consistency']:.4f}   L_consistency = {p(lift['consistency']):+.2f} pts")
    print(f"  advantage of aug explained by lift alone: {p(lift['aug'] - lift['consistency']):+.2f} pts")

    sig = table[~table['below_floor']]
    print('\n' + '=' * 74)
    print('NOISE-SPECIFIC RECOVERY (R) - lift removed')
    print('%-12s %9s %9s %9s %9s' % ('condition', 'R_aug', 'R_cons', 'rho_aug', 'rho_cons'))
    for _, r in sig.sort_values('R_aug', ascending=False).iterrows():
        print('%-12s %9.2f %9.2f %9.2f %9.2f'
              % (r['short'], p(r['R_aug']), p(r['R_consistency']),
                 r['rho_aug'], r['rho_consistency']))
    print('%-12s %9.2f %9.2f' % ('MEAN', p(sig['R_aug'].mean()), p(sig['R_consistency'].mean())))
    if len(sig) > 2:
        print('  correlation(R_aug, R_consistency) = %.3f'
              % np.corrcoef(sig['R_aug'], sig['R_consistency'])[0, 1])
        print('  mean |R_aug - R_consistency|      = %.2f pts'
              % p((sig['R_aug'] - sig['R_consistency']).abs().mean()))

    flagged = table[table['below_floor']]['short'].tolist()
    print('\n' + '=' * 74)
    print(f'RELIABILITY FLOOR (|drop| < {floor:.3f} or sign unstable)')
    print(f'  {len(flagged)} of {len(table)} flagged: {", ".join(flagged)}')
    print(f'  effective dimensionality of the perturbation grid = {len(table) - len(flagged)}')
    print('=' * 74)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor', type=float, default=DEFAULT_FLOOR,
                        help='reliability floor in ANLS units (default 0.006 = 0.6 points)')
    args = parser.parse_args()
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

    table, clean, lift = build_metrics(args.floor)
    export_metrics_csv(table)
    fig_noise_floor(plt, table, args.floor)
    fig_recovery_equivalence(plt, table)
    fig_lift_decomposition(plt, table, lift)
    fig_reducibility(plt, table)
    fig_retention(plt, table)
    print_decomposition_summary(table, clean, lift, args.floor)


if __name__ == '__main__':
    main()
