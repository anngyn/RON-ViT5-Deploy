"""DEPRECATED - kept only so existing commands keep working.

The effect-decomposition metrics and figures now live in ``plot_report_figures.py``
so that one command regenerates every figure in RESULTS_REPORT.md and there is a
single source of truth for the numbers.

Prefer:
    python scripts/plot_report_figures.py

This wrapper runs only the decomposition part (it skips the descriptive figures).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_report_figures import (  # noqa: E402
    DEFAULT_FLOOR,
    build_metrics,
    export_metrics_csv,
    fig_lift_decomposition,
    fig_noise_floor,
    fig_recovery_equivalence,
    fig_reducibility,
    fig_retention,
    print_decomposition_summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--floor', type=float, default=DEFAULT_FLOOR,
                        help='reliability floor in ANLS units (default 0.006 = 0.6 points)')
    args = parser.parse_args()

    print('NOTE: this script is deprecated; use plot_report_figures.py instead.\n')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('matplotlib required') from exc

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
