#!/usr/bin/env python3
"""Generate DiskANN Search Complexity (L) Ablation figure with 3 panels.

All three subplots are aligned in height with consistent coral colorscheme.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(REPO_ROOT, "docs", "plots")

# Coral color for DiskANN
BAR_COLOR = "#fb8072"


def main() -> None:
    # === Data ===
    
    # Internal QPS (index-only, from search_disk_index, Beamwidth=4)
    internal_qps = [
        {"L": 150, "QPS": 10715.00},
        {"L": 500, "QPS": 4037.98},
        {"L": 1000, "QPS": 1885.35},
        {"L": 1500, "QPS": 1324.28},
        {"L": 2000, "QPS": 989.19},
    ]
    
    # Batched e2e QPS (from plot_qps.py)
    batched_qps = [
        {"L": 1000, "QPS": 238.10},
        {"L": 2000, "QPS": 232.56},
        {"L": 3000, "QPS": 200.00},
        {"L": 4000, "QPS": 200.00},
        {"L": 5000, "QPS": 147.06},
    ]
    
    # Single-request latency (total_ms)
    single_latency = [
        {"L": 100, "latency_ms": 41.49},
        {"L": 500, "latency_ms": 56.77},
        {"L": 1000, "latency_ms": 76.96},
        {"L": 1500, "latency_ms": 96.32},
        {"L": 2000, "latency_ms": 124.04},
    ]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    if sns:
        sns.set_theme(style="whitegrid")

    # Create figure with 3 subplots, all same height
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax1, ax2, ax3 = axes

    # === Subplot 1: Internal QPS vs L ===
    internal_sorted = sorted(internal_qps, key=lambda r: r["L"])
    labels1 = [str(r["L"]) for r in internal_sorted]
    vals1 = [r["QPS"] for r in internal_sorted]
    
    ax1.bar(labels1, vals1, color=BAR_COLOR, edgecolor='none')
    ax1.set_xlabel("L", fontsize=11)
    ax1.set_ylabel("QPS (↑ higher is better)", fontsize=10)
    ax1.set_title("Internal Index QPS", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    for i, v in enumerate(vals1):
        ax1.annotate(f"{v:,.0f}", (i, v), ha="center", va="bottom",
                     fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    ax1.set_ylim(0, max(vals1) * 1.15)

    # === Subplot 2: Batched e2e QPS vs L ===
    batched_sorted = sorted(batched_qps, key=lambda r: r["L"])
    labels2 = [str(r["L"]) for r in batched_sorted]
    vals2 = [r["QPS"] for r in batched_sorted]
    
    ax2.bar(labels2, vals2, color=BAR_COLOR, edgecolor='none')
    ax2.set_xlabel("L", fontsize=11)
    ax2.set_ylabel("QPS (↑ higher is better)", fontsize=10)
    ax2.set_title("Batched E2E QPS", fontsize=12)
    
    for i, v in enumerate(vals2):
        ax2.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom",
                     fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    ax2.set_ylim(0, max(vals2) * 1.15)

    # === Subplot 3: Single-request latency vs L ===
    latency_sorted = sorted(single_latency, key=lambda r: r["L"])
    labels3 = [str(r["L"]) for r in latency_sorted]
    vals3 = [r["latency_ms"] for r in latency_sorted]
    
    ax3.bar(labels3, vals3, color=BAR_COLOR, edgecolor='none')
    ax3.set_xlabel("L", fontsize=11)
    ax3.set_ylabel("Latency (ms) (↓ lower is better)", fontsize=10)
    ax3.set_title("Single-request Latency", fontsize=12)
    
    for i, v in enumerate(vals3):
        ax3.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom",
                     fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    ax3.set_ylim(0, max(vals3) * 1.15)

    # Ensure all subplots have same height
    fig.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "diskann_ablation_three_panel.png")
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

