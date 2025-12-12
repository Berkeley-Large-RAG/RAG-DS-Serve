#!/usr/bin/env python3
"""Generate DiskANN vs IVFPQ internal (index-only) QPS comparison plot."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None


DISKANN_COLOR = "#fb8072"  # Coral
IVFPQ_COLOR = "#80b1d3"    # Blue
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "plots", "internal_qps_diskann_vs_ivfpq.png")

# DiskANN internal QPS (from search_disk_index, Beamwidth=4)
DISKANN_INTERNAL = [
    {"L": 150, "QPS": 10715.00},
    {"L": 500, "QPS": 4037.98},
    {"L": 1000, "QPS": 1885.35},
    {"L": 1500, "QPS": 1324.28},
    {"L": 2000, "QPS": 989.19},
]

# IVFPQ internal QPS (calculated from search_ms: QPS = 1000 / search_ms)
# search_ms values from plot_qps.py faiss_rows
IVFPQ_INTERNAL = [
    {"nprobe": 32, "search_ms": 4.39, "QPS": 1000 / 4.39},   # 227.8
    {"nprobe": 64, "search_ms": 5.02, "QPS": 1000 / 5.02},   # 199.2
    {"nprobe": 128, "search_ms": 6.27, "QPS": 1000 / 6.27},  # 159.5
    {"nprobe": 256, "search_ms": 8.70, "QPS": 1000 / 8.70},  # 114.9
]


def main() -> None:
    if sns:
        sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # === DiskANN subplot ===
    diskann_sorted = sorted(DISKANN_INTERNAL, key=lambda r: r["L"])
    labels1 = [str(r["L"]) for r in diskann_sorted]
    vals1 = [r["QPS"] for r in diskann_sorted]

    ax1.bar(labels1, vals1, color=DISKANN_COLOR)
    ax1.set_xlabel("L", fontsize=12)
    ax1.set_ylabel("QPS (↑ higher is better)", fontsize=12)
    ax1.set_title("DiskANN Internal Index QPS", fontsize=13)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    for i, v in enumerate(vals1):
        ax1.annotate(f"{v:,.0f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 3), textcoords="offset points")

    # === IVFPQ subplot ===
    ivfpq_sorted = sorted(IVFPQ_INTERNAL, key=lambda r: r["nprobe"])
    labels2 = [str(r["nprobe"]) for r in ivfpq_sorted]
    vals2 = [r["QPS"] for r in ivfpq_sorted]

    ax2.bar(labels2, vals2, color=IVFPQ_COLOR)
    ax2.set_xlabel("nprobe", fontsize=12)
    ax2.set_ylabel("QPS (↑ higher is better)", fontsize=12)
    ax2.set_title("IVFPQ Internal Index QPS", fontsize=13)

    for i, v in enumerate(vals2):
        ax2.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 3), textcoords="offset points")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
