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
    {"L": 1000, "embed_ms": 23.05, "search_ms": 350.32, "map_ms": 185.37, "total_ms": 373.38},
    {"L": 1500, "embed_ms": 22.50, "search_ms": 238.81, "map_ms": 75.26, "total_ms": 261.32},
    {"L": 2000, "embed_ms": 39.74, "search_ms": 230.19, "map_ms": 56.74, "total_ms": 269.93},
    {"L": 2500, "embed_ms": 22.21, "search_ms": 274.06, "map_ms": 43.02, "total_ms": 296.27},
    {"L": 3000, "embed_ms": 22.15, "search_ms": 294.56, "map_ms": 39.07, "total_ms": 316.71},
]


def annotate(ax: plt.Axes, width: float, metrics: List[str]) -> None:
    for idx, metric in enumerate(metrics):
        for group_idx, row in enumerate(SINGLE_REQUEST_LATENCY):
            val = row[metric]
            if val <= 0:
                continue
            x = group_idx + idx * width - (width * (len(metrics) - 1) / 2)
            ax.annotate(
                f"{val:.1f}",
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
        ("search_ms", "DiskANN search"),
        ("map_ms", "Mapping"),
        ("total_ms", "Total"),
    ]

    labels = [str(row["L"]) for row in SINGLE_REQUEST_LATENCY]
    num_groups = len(labels)
    num_metrics = len(metrics)
    width = 0.85 / num_metrics

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for idx, (metric_key, display_name) in enumerate(metrics):
        values = [row[metric_key] for row in SINGLE_REQUEST_LATENCY]
        positions = [i + idx * width - (width * (num_metrics - 1) / 2) for i in range(num_groups)]
        ax.bar(positions, values, width=width, label=display_name, color=palette[idx % len(palette)])

    ax.set_xticks(range(num_groups))
    ax.set_xticklabels(labels)
    ax.set_xlabel("L (DiskANN list size)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latency per request (ms)", fontsize=13, fontweight="bold")
    ax.set_title("DiskANN Single-request Latency Breakdown", fontsize=16, fontweight="bold")
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


