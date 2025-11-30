#!/usr/bin/env python3
"""Plot FAISS batched vs single-request QPS/latency for nprobe sweep."""

from __future__ import annotations

import os
from typing import List, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

# Hard-coded measurements (COUNT=400, shared queries, nprobe=32/64/128/256)
BATCHED_RESULTS: List[Dict[str, float]] = [
    {"nprobe": 32, "qps": 3.33, "embed": 2.47, "search": 297.75, "total": 300.22},
    {"nprobe": 64, "qps": 3.42, "embed": 2.29, "search": 292.17, "total": 294.47},
    {"nprobe": 128, "qps": 3.39, "embed": 2.57, "search": 291.86, "total": 294.43},
    {"nprobe": 256, "qps": 3.36, "embed": 2.40, "search": 293.60, "total": 296.01},
]

SINGLE_RESULTS: List[Dict[str, float]] = [
    {"nprobe": 32, "qps": 4.12, "embed": 29.27, "search": 213.39, "total": 242.67},
    {"nprobe": 64, "qps": 3.57, "embed": 30.16, "search": 246.77, "total": 276.93},
    {"nprobe": 128, "qps": 2.92, "embed": 30.61, "search": 310.37, "total": 340.99},
    {"nprobe": 256, "qps": 2.20, "embed": 30.20, "search": 423.83, "total": 454.03},
]


def plot_qps(out_dir: str) -> str:
    np_values = [str(row["nprobe"]) for row in BATCHED_RESULTS]
    batched_qps = [row["qps"] for row in BATCHED_RESULTS]
    single_qps = [row["qps"] for row in SINGLE_RESULTS]

    plt.figure(figsize=(8, 4))
    if sns:
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(
            x=np_values + np_values,
            y=batched_qps + single_qps,
            hue=(["Batched"] * len(np_values)) + (["Single-request"] * len(np_values)),
            palette=["#4C78A8", "#F58518"],
        )
    else:
        width = 0.4
        x = range(len(np_values))
        ax = plt.gca()
        ax.bar([i - width / 2 for i in x], batched_qps, width=width, label="Batched", color="#4C78A8")
        ax.bar([i + width / 2 for i in x], single_qps, width=width, label="Single-request", color="#F58518")
        ax.set_xticks(list(x))
        ax.set_xticklabels(np_values)

    plt.xlabel("nprobe")
    plt.ylabel("QPS")
    plt.title("FAISS QPS: Batched vs Single-request (COUNT=400)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "faiss_qps_batch_vs_single.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_latency(out_dir: str) -> str:
    np_values = [str(row["nprobe"]) for row in BATCHED_RESULTS]
    metrics = [("embed", "Embed"), ("search", "Search"), ("total", "Total")]

    plt.figure(figsize=(10, 5))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    width = 0.35
    x = range(len(np_values))

    for idx, (key, label) in enumerate(metrics):
        batched = [row[key] for row in BATCHED_RESULTS]
        single = [row[key] for row in SINGLE_RESULTS]
        offsets = [-width, width]
        ax.bar(
            [i + offsets[0] + idx * width / len(metrics) for i in x],
            batched,
            width=width / len(metrics),
            label=f"Batched {label}" if idx == 0 else "",
            color="#4C78A8",
            alpha=0.8 - idx * 0.1,
        )
        ax.bar(
            [i + offsets[1] + idx * width / len(metrics) for i in x],
            single,
            width=width / len(metrics),
            label=f"Single {label}" if idx == 0 else "",
            color="#F58518",
            alpha=0.8 - idx * 0.1,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(np_values)
    ax.set_xlabel("nprobe")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("FAISS Latency Components: Batched vs Single-request")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "faiss_latency_batch_vs_single.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main() -> None:
    out_dir = os.path.join(DOCS_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)
    qps_path = plot_qps(out_dir)
    lat_path = plot_latency(out_dir)
    print(f"Wrote {qps_path}")
    print(f"Wrote {lat_path}")


if __name__ == "__main__":
    main()

