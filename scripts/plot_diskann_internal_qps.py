#!/usr/bin/env python3
"""Generate DiskANN internal (index-only) QPS vs L plot.

This shows the raw index search throughput without embedding or network overhead.
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


BAR_COLOR = "#fb8072"  # Darker Set3 coral (DiskANN color)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "plots", "diskann_internal_qps_vs_L.png")

# Internal QPS measurements from search_disk_index (Beamwidth=4)
INTERNAL_QPS_RESULTS: List[dict] = [
    {"L": 150, "QPS": 10715.00},
    {"L": 500, "QPS": 4037.98},
    {"L": 1000, "QPS": 1885.35},
    {"L": 1500, "QPS": 1324.28},
    {"L": 2000, "QPS": 989.19},
]


def annotate(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        # Format with commas for thousands
        label = f"{height:,.0f}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            xytext=(0, 3),
            textcoords="offset points",
        )


def main() -> None:
    rows = sorted(INTERNAL_QPS_RESULTS, key=lambda r: r["L"])
    labels = [str(r["L"]) for r in rows]
    values = [r["QPS"] for r in rows]

    if sns:
        sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 4.5))
    ax = plt.gca()
    ax.bar(labels, values, color=BAR_COLOR)
    ax.set_xlabel("L", fontsize=13)
    ax.set_ylabel("QPS (↑ higher is better)", fontsize=13)
    ax.set_title("DiskANN Internal Index QPS vs L", fontsize=14)

    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontsize(11)

    # Add thousands separator to y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    annotate(ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
