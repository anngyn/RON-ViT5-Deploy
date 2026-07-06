"""Plot per-epoch training curves (dev ANLS, train loss) for the 3 flows.

Data source: outputs/logs/*.log. Where a log is missing per-epoch dev ANLS,
values are taken from the training run records embedded below (from run logs).
"""
import argparse
import os
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Per-epoch records captured from training runs (REAL data only).
# - baseline: from run console output (dev ANLS + avg loss per epoch)
# - consistency: parsed from outputs/logs/consistency_only.log
# noisy_aug per-epoch dev ANLS was NOT persisted -> omitted (no fabrication).
KNOWN = {
    'baseline':    {'dev_anls': [0.7842, 0.8140, 0.8291], 'train_loss': [0.6435, 0.4084, 0.3286]},
    'consistency': {'dev_anls': [0.7868, 0.8110, 0.8310], 'train_loss': None},
}


def parse_log_dev_anls(log_path):
    if not os.path.exists(log_path):
        return None
    vals = []
    pat = re.compile(r'(?<!Best )Dev ANLS:\s*([0-9.]+)')
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m:
                vals.append(float(m.group(1)))
    return vals or None


def main():
    parser = argparse.ArgumentParser(description="Plot training curves for the 3 flows.")
    parser.add_argument('--output-prefix', default='outputs/results/training_curves')
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib required.") from exc

    # Prefer real log for consistency (has per-epoch dev ANLS)
    cons_log = parse_log_dev_anls(os.path.join(PROJECT_ROOT, 'outputs/logs/consistency_only.log'))
    if cons_log:
        KNOWN['consistency']['dev_anls'] = cons_log

    epochs = [1, 2, 3]
    colors = {'baseline': '#2ca02c', 'noisy_aug': '#ff7f0e', 'consistency': '#1f77b4'}

    output_prefix = os.path.join(PROJECT_ROOT, args.output_prefix)
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # --- Dev ANLS curves ---
    plt.figure(figsize=(9, 5.5))
    for tag, rec in KNOWN.items():
        plt.plot(epochs, rec['dev_anls'], marker='o', linewidth=2,
                 color=colors[tag], label=tag)
        for x, y in zip(epochs, rec['dev_anls']):
            plt.annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=8)
    plt.title('Dev ANLS per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dev ANLS')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title='Flow')
    plt.tight_layout()
    p1 = f"{output_prefix}_dev_anls.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Saved: {p1}")

    # --- Train loss curves (flows with loss data) ---
    plt.figure(figsize=(9, 5.5))
    for tag, rec in KNOWN.items():
        if rec['train_loss'] is None:
            continue
        plt.plot(epochs, rec['train_loss'], marker='s', linewidth=2,
                 color=colors[tag], label=tag)
        for x, y in zip(epochs, rec['train_loss']):
            plt.annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=8)
    plt.title('Train Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Avg CE Loss')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title='Flow')
    plt.tight_layout()
    p2 = f"{output_prefix}_train_loss.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Saved: {p2}")


if __name__ == '__main__':
    main()
