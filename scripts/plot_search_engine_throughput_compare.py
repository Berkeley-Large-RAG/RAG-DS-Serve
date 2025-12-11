#!/usr/bin/env python3
"""Generate latency/throughput comparison and accuracy plots for Google API vs DS-Serve Database."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # noqa: SIM105
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - seaborn is optional
    sns = None


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(REPO_ROOT, "docs", "plots")


def main() -> None:
    # Darker subset of ColorBrewer Set3 - no dash in "DS Serve"
    colors = {
        "Google API": "#fb8072",            # coral
        "DS Serve": "#80b1d3",              # blue
    }

    # Single-request data (QPS -> latency)
    single_qps = {
        "Google API": 6.6055,
        "DS Serve": 12.4713,
    }
    single_latency_ms = {k: (1000.0 / v if v > 0 else 0) for k, v in single_qps.items()}

    # Batched throughput (QPS)
    batched_qps = {
        "Google API": 8.2488,
        "DS Serve": 238.10,
    }

    # Use identical figsize for both figures so they align perfectly
    FIGSIZE = (8, 4)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Combined figure: latency (single) + throughput (batched)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    if sns:
        sns.set_theme(style="whitegrid")

    # Latency subplot (single-request)
    labels_lat = list(single_latency_ms.keys())
    vals_lat = [single_latency_ms[k] for k in labels_lat]
    ax1.bar(labels_lat, vals_lat, color=[colors[l] for l in labels_lat], edgecolor='none')
    ax1.set_title("Single-request latency (ms)")
    ax1.set_ylabel("Latency (ms)")
    for i, v in enumerate(vals_lat):
        ax1.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    try:
        ymax = max(vals_lat)
        if ymax > 0:
            ax1.set_ylim(0, ymax * 1.15)
    except Exception:
        pass

    # Throughput subplot (batched)
    labels_qps = list(batched_qps.keys())
    vals_qps = [batched_qps[k] for k in labels_qps]
    ax2.bar(labels_qps, vals_qps, color=[colors[l] for l in labels_qps], edgecolor='none')
    ax2.set_title("Batched throughput (QPS)")
    ax2.set_ylabel("QPS")
    for i, v in enumerate(vals_qps):
        ax2.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    try:
        ymax = max(vals_qps)
        if ymax > 0:
            ax2.set_ylim(0, ymax * 1.15)
    except Exception:
        pass

    fig1.tight_layout()
    fig1.savefig(os.path.join(PLOTS_DIR, "search_engine_latency_throughput.png"), dpi=200)
    plt.close(fig1)

    # Accuracy figure (AVG row from provided table) - same figsize for alignment
    # Use shorter labels to keep x-axis flat (no rotation)
    accuracy_rows = [
        ("No Retrieval", 48.3),
        ("CSE", 51.3),
        ("CSE+LM", 51.5),
        ("DS Serve", 55.1),
        ("DS Serve+LM", 56.0),
    ]
    fig2, ax = plt.subplots(figsize=FIGSIZE)
    if sns:
        sns.set_theme(style="whitegrid")
    labels_acc = [r[0] for r in accuracy_rows]
    vals_acc = [r[1] for r in accuracy_rows]
    color_acc = []
    for lab in labels_acc:
        if "DS Serve" in lab:
            color_acc.append(colors["DS Serve"])
        elif "CSE" in lab:
            color_acc.append(colors["Google API"])
        else:
            color_acc.append("#b3b3b3")
    ax.bar(labels_acc, vals_acc, color=color_acc)
    ax.set_title("AVG accuracy (Llama 3.1 8B Instruct)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(len(labels_acc)))
    ax.set_xticklabels(labels_acc, rotation=0, ha="center", fontsize=9)  # flat labels
    for i, v in enumerate(vals_acc):
        ax.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    try:
        ymax = max(vals_acc)
        if ymax > 0:
            ax.set_ylim(0, ymax * 1.12)
    except Exception:
        pass
    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOTS_DIR, "search_engine_accuracy_avg.png"), dpi=200)
    plt.close(fig2)


if __name__ == "__main__":
    main()

