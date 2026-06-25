"""Plot grouped ANLS/drop charts from noise-grid CSV files."""
import argparse
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONDITION_ORDER = ['clean', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N10', 'N13', 'N14', 'N16', 'N17', 'N18', 'N20']


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def build_condition_label(row):
    if row['condition_id'] == 'clean':
        return 'clean'
    return f"{row['condition_id']}-L{int(row['level'])}"


def order_key(label):
    if label == 'clean':
        return (0, 0)
    condition_id, level = label.split('-L')
    return (CONDITION_ORDER.index(condition_id), int(level))


def main():
    parser = argparse.ArgumentParser(description="Plot ANLS and drop charts from noise-grid CSV files.")
    parser.add_argument('--csv', nargs='+', required=True, help='One or more CSV files from eval_noise_grid.py')
    parser.add_argument('--output-prefix', required=True, help='Output prefix for PNG files')
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it first.") from exc

    frames = []
    for csv_path in args.csv:
        resolved = resolve_project_path(csv_path)
        df = pd.read_csv(resolved)
        df['condition_label'] = df.apply(build_condition_label, axis=1)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    labels = sorted(combined['condition_label'].unique(), key=order_key)

    anls_pivot = combined.pivot_table(index='condition_label', columns='model_tag', values='anls', aggfunc='first')
    drop_pivot = combined.pivot_table(index='condition_label', columns='model_tag', values='drop_from_clean', aggfunc='first')

    anls_pivot = anls_pivot.reindex(labels)
    drop_pivot = drop_pivot.reindex(labels)

    output_prefix = resolve_project_path(args.output_prefix)
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    for metric_name, pivot_df, ylabel, filename_suffix in [
        ('ANLS', anls_pivot, 'ANLS', 'anls'),
        ('Drop from clean', drop_pivot, 'Drop from clean', 'drop'),
    ]:
        ax = pivot_df.plot(kind='bar', figsize=(16, 6), width=0.85)
        ax.set_title(f"ReceiptVQA OCR noise comparison: {metric_name}")
        ax.set_xlabel('Condition')
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(title='Model', loc='best')
        plt.tight_layout()
        output_path = f"{output_prefix}_{filename_suffix}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved plot: {output_path}")


if __name__ == '__main__':
    main()
