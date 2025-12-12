#!/usr/bin/env python3
"""Generate combined DiskANN vs IVFPQ figure: throughput, latency, TriviaQA, and NaturalQS accuracy.

All four subplots are aligned in height (same top and bottom edges).
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


def plot_accuracy_panel(ax, metrics_data, metric_order, title, colors, bar_width=0.36):
    """Helper to plot an accuracy panel with grouped bars."""
    x_positions = list(range(len(metric_order)))
    
    ivfpq_vals = [metrics_data[m]["ivfpq"] for m in metric_order]
    diskann_vals = [metrics_data[m]["diskann"] for m in metric_order]
    
    ax.bar([x - bar_width / 2 for x in x_positions], ivfpq_vals, width=bar_width,
           label="IVFPQ", color=colors["IVFPQ"])
    ax.bar([x + bar_width / 2 for x in x_positions], diskann_vals, width=bar_width,
           label="DiskANN", color=colors["DiskANN"])
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_order)
    ax.set_ylabel("Score (↑ higher is better)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    
    # Set y-axis limits with padding
    all_vals = ivfpq_vals + diskann_vals
    vmin, vmax = min(all_vals), max(all_vals)
    padding = max(0.02, (vmax - vmin) * 0.25)
    ax.set_ylim(max(0.0, vmin - padding), min(1.0, vmax + padding))
    
    # Annotate bars with percentages
    for i, v in enumerate(ivfpq_vals):
        ax.annotate(f"{v*100:.1f}%", (x_positions[i] - bar_width / 2, v),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    xytext=(0, 2), textcoords="offset points")
    for i, v in enumerate(diskann_vals):
        ax.annotate(f"{v*100:.1f}%", (x_positions[i] + bar_width / 2, v),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    xytext=(0, 2), textcoords="offset points")


def main() -> None:
    # Colors - consistent with other figures
    colors = {
        "DiskANN": "#fb8072",  # coral
        "IVFPQ": "#80b1d3",    # blue
    }
    
    # Data for recommended configs: DiskANN L=2000, IVFPQ nprobe=256
    # Throughput (batched QPS) - from 2024-12-11 tests
    throughput = {"DiskANN": 232.56, "IVFPQ": 66.67}
    
    # Latency (single-request, ms) - from 2024-12-11 tests
    latency = {"DiskANN": 124.04, "IVFPQ": 307.18}
    
    # TriviaQA accuracy data (from plot_ivfpq_vs_diskann.py)
    triviaqa_metrics = {
        "Recall": {"ivfpq": 0.882, "diskann": 0.896},
        "F1": {"ivfpq": 0.885, "diskann": 0.898},
        "EM": {"ivfpq": 0.840, "diskann": 0.855},
    }
    
    # NaturalQS accuracy data (from plot_ivfpq_vs_diskann.py)
    naturalqs_metrics = {
        "Recall": {"ivfpq": 0.449, "diskann": 0.484},
        "F1": {"ivfpq": 0.492, "diskann": 0.507},
        "EM": {"ivfpq": 0.385, "diskann": 0.392},
    }
    metric_order = ["Recall", "F1", "EM"]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    if sns:
        sns.set_theme(style="whitegrid")

    # Create figure with 4 subplots, all same height
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    ax1, ax2, ax3, ax4 = axes

    # === Subplot 1: Batched throughput ===
    labels1 = list(throughput.keys())
    vals1 = list(throughput.values())
    bar_colors1 = [colors["DiskANN"], colors["IVFPQ"]]
    ax1.bar(labels1, vals1, color=bar_colors1, edgecolor='none')
    ax1.set_title("Batched throughput", fontsize=11)
    ax1.set_ylabel("QPS (↑ higher is better)")
    for i, v in enumerate(vals1):
        ax1.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    ax1.set_ylim(0, max(vals1) * 1.18)

    # === Subplot 2: Single-request latency ===
    labels2 = list(latency.keys())
    vals2 = list(latency.values())
    bar_colors2 = [colors["DiskANN"], colors["IVFPQ"]]
    ax2.bar(labels2, vals2, color=bar_colors2, edgecolor='none')
    ax2.set_title("Single-request latency", fontsize=11)
    ax2.set_ylabel("Latency (ms) (↓ lower is better)")
    for i, v in enumerate(vals2):
        ax2.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    ax2.set_ylim(0, max(vals2) * 1.18)

    # === Subplot 3: TriviaQA accuracy ===
    plot_accuracy_panel(ax3, triviaqa_metrics, metric_order, 
                        "TriviaQA Accuracy", colors)

    # === Subplot 4: NaturalQS accuracy ===
    plot_accuracy_panel(ax4, naturalqs_metrics, metric_order,
                        "NaturalQS Accuracy", colors)

    # Ensure all subplots have same height by using tight_layout
    fig.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "diskann_vs_ivfpq_four_panel.png")
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

