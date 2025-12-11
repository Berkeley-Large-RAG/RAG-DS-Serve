#!/usr/bin/env python3
"""Generate LLM interpolation accuracy comparison figure."""

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
    # Data from Table 1: LLaMa 3.1 8B Instruct results
    tasks = ["MMLU", "MMLU Pro", "AGI Eval", "MATH", "GPQA"]
    no_ds_serve = [68.9, 39.8, 56.2, 46.9, 29.9]
    ds_serve = [73.5, 47.5, 56.2, 50.0, 31.7]
    ds_serve_exact = [73.7, 49.4, 58.3, 53.1, 36.6]

    # Colors matching other figures (darker Set3 tones, consistent with other plots)
    colors = {
        "No DS Serve": "#b3b3b3",       # gray (baseline)
        "DS Serve": "#80b1d3",          # blue
        "DS Serve + Exact": "#fb8072",  # coral (best option, matches DiskANN color)
    }

    x = list(range(len(tasks)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if sns:
        sns.set_theme(style="whitegrid")

    # Create grouped bars
    bars1 = ax.bar([i - width for i in x], no_ds_serve, width, 
                   label="No DS Serve", color=colors["No DS Serve"], edgecolor='none')
    bars2 = ax.bar(x, ds_serve, width, 
                   label="DS Serve", color=colors["DS Serve"], edgecolor='none')
    bars3 = ax.bar([i + width for i in x], ds_serve_exact, width, 
                   label="DS Serve + Exact", color=colors["DS Serve + Exact"], edgecolor='none')

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Interpolating DS Serve with LLM (LLaMa 3.1 8B Instruct)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.legend(fontsize=10, loc="upper right")

    # Add value annotations
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold",
                        xytext=(0, 2), textcoords="offset points")

    # Set y-axis limits with headroom
    all_vals = no_ds_serve + ds_serve + ds_serve_exact
    ax.set_ylim(0, max(all_vals) * 1.12)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "llm_interpolation_accuracy.png"), dpi=200)
    plt.close()
    print(f"Saved: {os.path.join(PLOTS_DIR, 'llm_interpolation_accuracy.png')}")


if __name__ == "__main__":
    main()

