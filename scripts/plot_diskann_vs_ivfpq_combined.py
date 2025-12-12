#!/usr/bin/env python3
"""Generate combined DiskANN vs IVFPQ comparison: throughput, latency, accuracy."""

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


def main() -> None:
    # Colors - consistent with other figures
    colors = {
        "DiskANN": "#fb8072",  # coral
        "IVFPQ": "#80b1d3",    # blue
    }
    
    # Data for recommended configs: DiskANN L=2000, IVFPQ nprobe=256
    # Throughput (batched QPS) - from 2024-12-11 tests
    throughput = {"DiskANN (L=2000)": 232.56, "IVFPQ (nprobe=256)": 66.67}
    
    # Latency (single-request, ms) - from 2024-12-11 tests
    latency = {"DiskANN (L=2000)": 124.04, "IVFPQ (nprobe=256)": 307.18}

    FIGSIZE = (7, 4)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    if sns:
        sns.set_theme(style="whitegrid")

    # Throughput subplot
    vals_qps = list(throughput.values())
    bar_colors_qps = [colors["DiskANN"], colors["IVFPQ"]]
    ax1.bar(["DiskANN", "IVFPQ"], vals_qps, color=bar_colors_qps, edgecolor='none')
    ax1.set_title("Batched throughput", fontsize=12)
    ax1.set_ylabel("QPS (↑ higher is better)")
    for i, v in enumerate(vals_qps):
        ax1.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    ax1.set_ylim(0, max(vals_qps) * 1.15)

    # Latency subplot
    vals_lat = list(latency.values())
    bar_colors_lat = [colors["DiskANN"], colors["IVFPQ"]]
    ax2.bar(["DiskANN", "IVFPQ"], vals_lat, color=bar_colors_lat, edgecolor='none')
    ax2.set_title("Single-request latency", fontsize=12)
    ax2.set_ylabel("Latency (ms) (↓ lower is better)")
    for i, v in enumerate(vals_lat):
        ax2.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    ax2.set_ylim(0, max(vals_lat) * 1.15)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "diskann_vs_ivfpq_combined.png"), dpi=200)
    plt.close(fig)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'diskann_vs_ivfpq_combined.png')}")


if __name__ == "__main__":
    main()

