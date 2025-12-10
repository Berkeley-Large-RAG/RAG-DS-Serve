#!/usr/bin/env python3
"""Plot DiskANN latency breakdown for single-request measurements."""

from __future__ import annotations

import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # noqa: SIM105
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOT_PATH = os.path.join(REPO_ROOT, "docs", "plots", "diskann_single_request_latency_vs_L.png")

SINGLE_REQUEST_LATENCY: List[dict] = [
    {"L": 100, "embed_ms": 30.05, "search_ms": 11.43, "map_ms": 0.51, "total_ms": 41.49},
    {"L": 500, "embed_ms": 28.90, "search_ms": 27.88, "map_ms": 0.74, "total_ms": 56.77},
    {"L": 1000, "embed_ms": 29.06, "search_ms": 47.90, "map_ms": 0.68, "total_ms": 76.96},
    {"L": 1500, "embed_ms": 29.07, "search_ms": 67.26, "map_ms": 0.94, "total_ms": 96.32},
    {"L": 2000, "embed_ms": 30.60, "search_ms": 93.44, "map_ms": 0.94, "total_ms": 124.04},
]


def annotate(ax: plt.Axes, width: float, metrics: List[str]) -> None:
    for idx, metric in enumerate(metrics):
        for group_idx, row in enumerate(SINGLE_REQUEST_LATENCY):
            val = row[metric]
            if val <= 0:
                continue
            x = group_idx + idx * width - (width * (len(metrics) - 1) / 2)
            ax.annotate(
                f"{val:.2f}",
                (x, val),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )


def main() -> None:
    if sns:
        sns.set_theme(style="whitegrid")

    metrics = [
        ("embed_ms", "Embed"),
        ("search_ms", "Index search"),
        ("map_ms", "Passage map"),
        ("total_ms", "Total"),
    ]

    labels = [str(row["L"]) for row in SINGLE_REQUEST_LATENCY]
    num_groups = len(labels)
    num_metrics = len(metrics)
    width = 0.85 / num_metrics

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    # ColorBrewer Set3 palette
    palette = ["#8dd3c7", "#fb8072", "#b3de69", "#fdb462"]

    for idx, (metric_key, display_name) in enumerate(metrics):
        values = [row[metric_key] for row in SINGLE_REQUEST_LATENCY]
        positions = [i + idx * width - (width * (num_metrics - 1) / 2) for i in range(num_groups)]
        ax.bar(positions, values, width=width, label=display_name, color=palette[idx % len(palette)])

    ax.set_xticks(range(num_groups))
    ax.set_xticklabels(labels)
    ax.set_xlabel("L", fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=13)  # match breakdown style; not bold
    ax.set_title("DiskANN Single-request Latency Breakdown", fontsize=14)  # smaller, unbold
    ax.legend(ncol=4, fontsize=11)

    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontsize(11)
        tick.set_fontweight("bold")

    annotate(ax, width, [key for key, _ in metrics])
    plt.tight_layout()

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()


