#!/usr/bin/env python3
"""Generate merged DiskANN vs IVFPQ internal QPS comparison - side by side bars."""

from __future__ import annotations

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None


DISKANN_COLOR = "#fb8072"  # Coral
IVFPQ_COLOR = "#80b1d3"    # Blue
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "plots", "internal_qps_diskann_vs_ivfpq.png")

# DiskANN internal QPS (from search_disk_index, Beamwidth=4, L=2000)
DISKANN_DATA = [
    {"label": "L=2000", "QPS": 989.19},
]

# IVFPQ internal QPS (calculated from search_ms: QPS = 1000 / search_ms)
IVFPQ_DATA = [
    {"label": "nprobe=32", "QPS": 1000 / 4.39},    # 227.8
    {"label": "nprobe=64", "QPS": 1000 / 5.02},    # 199.2
    {"label": "nprobe=128", "QPS": 1000 / 6.27},   # 159.5
    {"label": "nprobe=256", "QPS": 1000 / 8.70},   # 114.9
    {"label": "nprobe=512", "QPS": 1000 / 18.76},  # 53.3
]


def main() -> None:
    if sns:
        sns.set_theme(style="whitegrid")

    # Create figure with side-by-side grouped bars (smaller size to match top row)
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Prepare data
    n_diskann = len(DISKANN_DATA)
    n_ivfpq = len(IVFPQ_DATA)
    
    # X positions
    x_diskann = np.arange(n_diskann)
    x_ivfpq = np.arange(n_diskann, n_diskann + n_ivfpq)
    
    # Values
    diskann_vals = [d["QPS"] for d in DISKANN_DATA]
    ivfpq_vals = [d["QPS"] for d in IVFPQ_DATA]
    
    # Labels
    diskann_labels = [d["label"] for d in DISKANN_DATA]
    ivfpq_labels = [d["label"] for d in IVFPQ_DATA]

    # Plot bars
    bars1 = ax.bar(x_diskann, diskann_vals, color=DISKANN_COLOR, label="DiskANN", width=0.7)
    bars2 = ax.bar(x_ivfpq, ivfpq_vals, color=IVFPQ_COLOR, label="IVFPQ", width=0.7)

    # Add separator line
    ax.axvline(x=n_diskann - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Labels
    ax.set_xlabel("Configuration", fontsize=10)
    ax.set_ylabel("QPS (↑ higher is better)", fontsize=10)
    ax.set_title("Internal Index QPS", fontsize=11, fontweight='bold')
    
    # X-axis ticks
    ax.set_xticks(list(x_diskann) + list(x_ivfpq))
    ax.set_xticklabels(diskann_labels + ivfpq_labels, rotation=30, ha='right', fontsize=9)
    
    # Y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Annotations
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    xytext=(0, 2), textcoords='offset points')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    xytext=(0, 2), textcoords='offset points')

    # Legend
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
