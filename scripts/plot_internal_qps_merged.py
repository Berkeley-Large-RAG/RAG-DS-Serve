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

# Paired data: DiskANN L values paired with IVFPQ nprobe values
# Each pair: (DiskANN config, IVFPQ config)
# IVFPQ internal QPS = 1000 / faiss_ms (pure index search, from batched test 2024-12-11)
PAIRED_DATA = [
    {"diskann_label": "L=150", "diskann_qps": 10715.00, "ivfpq_label": "nprobe=32", "ivfpq_qps": 1000 / 1.00},   # 1000
    {"diskann_label": "L=500", "diskann_qps": 4037.98, "ivfpq_label": "nprobe=64", "ivfpq_qps": 1000 / 1.64},   # 610
    {"diskann_label": "L=1000", "diskann_qps": 1885.35, "ivfpq_label": "nprobe=128", "ivfpq_qps": 1000 / 2.97}, # 337
    {"diskann_label": "L=1500", "diskann_qps": 1324.28, "ivfpq_label": "nprobe=256", "ivfpq_qps": 1000 / 5.41}, # 185
    {"diskann_label": "L=2000", "diskann_qps": 989.19, "ivfpq_label": "nprobe=512", "ivfpq_qps": 1000 / 10.01}, # 100
]


def main() -> None:
    if sns:
        sns.set_theme(style="whitegrid")

    # Create figure with paired grouped bars (smaller size to match top row)
    fig, ax = plt.subplots(figsize=(10, 4))

    n_pairs = len(PAIRED_DATA)
    bar_width = 0.35
    gap_between_pairs = 0.3  # Gap between each pair
    
    # Calculate x positions for paired bars
    x_positions = []
    current_x = 0
    for i in range(n_pairs):
        x_positions.append(current_x)
        current_x += 1 + gap_between_pairs  # Move to next pair with gap
    
    x_positions = np.array(x_positions)
    
    # Extract data
    diskann_vals = [d["diskann_qps"] for d in PAIRED_DATA]
    ivfpq_vals = [d["ivfpq_qps"] for d in PAIRED_DATA]
    diskann_labels = [d["diskann_label"] for d in PAIRED_DATA]
    ivfpq_labels = [d["ivfpq_label"] for d in PAIRED_DATA]

    # Plot paired bars
    bars1 = ax.bar(x_positions - bar_width/2, diskann_vals, bar_width, color=DISKANN_COLOR, label="DiskANN")
    bars2 = ax.bar(x_positions + bar_width/2, ivfpq_vals, bar_width, color=IVFPQ_COLOR, label="IVFPQ")

    # Labels
    ax.set_ylabel("QPS (↑ higher is better)", fontsize=10)
    ax.set_title("Internal Index QPS", fontsize=11, fontweight='bold')
    
    # X-axis: show both labels for each pair
    pair_labels = [f"{d}\n{i}" for d, i in zip(diskann_labels, ivfpq_labels)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pair_labels, fontsize=9)
    
    # Y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Annotations for DiskANN bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=7, fontweight='bold',
                    xytext=(0, 2), textcoords='offset points')
    
    # Annotations for IVFPQ bars
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=7, fontweight='bold',
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
