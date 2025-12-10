#!/usr/bin/env python3
"""Generate IVFPQ benchmarking plots (batched & single) per nprobe sweep."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOT_DIR = os.path.join(REPO_ROOT, "docs", "plots")
# ColorBrewer Set3 palette for consistent professional colors
PALETTE = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69"]
COLOR_QPS = PALETTE[4]      # soft blue
COLOR_EMBED = PALETTE[0]    # teal
COLOR_SEARCH = PALETTE[3]   # coral
COLOR_TOTAL = PALETTE[5]    # orange

# Measurements collected with COUNT=100 shared queries.
BATCHED_RESULTS: List[Dict[str, float]] = [
    {"nprobe": 64, "qps": 9.09, "embed": 2.54, "search": 10.71, "total": 13.25},
    {"nprobe": 128, "qps": 10.00, "embed": 2.41, "search": 10.79, "total": 13.19},
    {"nprobe": 256, "qps": 9.09, "embed": 2.52, "search": 12.60, "total": 15.12},
    {"nprobe": 512, "qps": 9.09, "embed": 2.54, "search": 18.76, "total": 21.30},
]

SINGLE_RESULTS: List[Dict[str, float]] = [
    {"nprobe": 64, "qps": 3.12, "embed": 27.23, "search": 113.87, "total": 141.11},
    {"nprobe": 128, "qps": 2.56, "embed": 27.76, "search": 179.66, "total": 207.42},
    {"nprobe": 256, "qps": 2.04, "embed": 28.73, "search": 281.10, "total": 309.83},
    {"nprobe": 512, "qps": 1.47, "embed": 32.15, "search": 486.72, "total": 518.88},
]


def _plot_qps(rows: List[Dict[str, float]], mode_label: str, filename: str) -> str:
    labels = [str(r["nprobe"]) for r in rows]
    values = [r["qps"] for r in rows]

    plt.figure(figsize=(7, 4))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()
    ax.bar(labels, values, color=COLOR_QPS)
    ax.set_xlabel("nprobe", fontsize=12)
    ax.set_ylabel("QPS", fontsize=12)
    ax.set_title(f"IVFPQ {mode_label} QPS vs nprobe", fontsize=14)

    for label, val in zip(labels, values):
        ax.annotate(
            f"{val:.2f}",
            (label, val),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )

    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontsize(11)

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _plot_latency(rows: List[Dict[str, float]], mode_label: str, filename: str) -> str:
    labels = [str(r["nprobe"]) for r in rows]
    metrics = [
        ("embed", "Embed", COLOR_EMBED),
        ("search", "Search", COLOR_SEARCH),
        ("total", "Total", COLOR_TOTAL),
    ]

    plt.figure(figsize=(8.5, 4.2))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    num_metrics = len(metrics)
    width = 0.8 / num_metrics
    base_positions = list(range(len(labels)))

    for idx, (key, disp, color) in enumerate(metrics):
        values = [r[key] for r in rows]
        offsets = [pos + (idx - (num_metrics - 1) / 2) * width for pos in base_positions]
        ax.bar(offsets, values, width=width, label=disp, color=color)
        for x, val in zip(offsets, values):
            ax.annotate(
                f"{val:.1f}",
                (x, val),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax.set_xticks(base_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("nprobe", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(f"IVFPQ {mode_label} latency breakdown", fontsize=14)
    ax.legend(fontsize=10)

    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontsize(11)

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    outputs = [
        _plot_qps(BATCHED_RESULTS, "Batched", "ivfpq_qps_batched.png"),
        _plot_latency(BATCHED_RESULTS, "Batched", "ivfpq_latency_batched.png"),
        _plot_qps(SINGLE_RESULTS, "Single-request", "ivfpq_qps_single.png"),
        _plot_latency(SINGLE_RESULTS, "Single-request", "ivfpq_latency_single.png"),
    ]
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
