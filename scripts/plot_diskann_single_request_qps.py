#!/usr/bin/env python3
"""Generate DiskANN single-request QPS vs L plot.

This reproduces the styling of existing performance charts so it can be embedded
directly in the docs alongside the batched measurements.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # noqa: SIM105
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - seaborn is optional
    sns = None


BAR_COLOR = "#4C78A8"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "plots", "diskann_single_request_qps_vs_L.png")

# Measurements captured via `REQUEST_MODE=single COUNT=20` sweep.
SINGLE_REQUEST_RESULTS: List[dict] = [
    {"L": 1000, "QPS": 2.99},
    {"L": 1500, "QPS": 3.39},
    {"L": 2000, "QPS": 3.12},
    {"L": 2500, "QPS": 3.12},
    {"L": 3000, "QPS": 2.70},
]


def annotate(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.annotate(
            f"{height:.2f}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            xytext=(0, 3),
            textcoords="offset points",
        )


def main() -> None:
    rows = sorted(SINGLE_REQUEST_RESULTS, key=lambda r: r["L"])
    labels = [str(r["L"]) for r in rows]
    values = [r["QPS"] for r in rows]

    if sns:
        sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8.5, 4.2))
    ax = plt.gca()
    ax.bar(labels, values, color=BAR_COLOR)
    ax.set_xlabel("L (DiskANN list size)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Single-request QPS", fontsize=13, fontweight="bold")
    ax.set_title("DiskANN Single-request QPS vs L", fontsize=16, fontweight="bold")

    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontsize(11)
        tick.set_fontweight("bold")

    annotate(ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()



