#!/usr/bin/env python3
"""Generate per-dataset IVFPQ vs DiskANN accuracy comparison plots.

The figures mirror the styling of the FAISS vs DiskANN accuracy charts by
reusing the same palette, typography, and bar layout while producing one image
per dataset (TriviaQA and NaturalQS) in Recall → F1 → EM order.
"""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (matplotlib backend must be set first)

try:  # noqa: SIM105
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - seaborn is optional
    sns = None


IVFPQ_COLOR = "#4C78A8"
DISKANN_COLOR = "#F58518"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
METRIC_ORDER = ["Recall", "F1", "EM"]
DATASETS = {
    "TriviaQA": {
        "EM": {"diskann": 0.857, "ivfpq": 0.840},
        "F1": {"diskann": 0.900, "ivfpq": 0.885},
        "Recall": {"diskann": 0.899, "ivfpq": 0.882},
    },
    "NaturalQS": {
        "EM": {"diskann": 0.392, "ivfpq": 0.385},
        "F1": {"diskann": 0.507, "ivfpq": 0.492},
        "Recall": {"diskann": 0.484, "ivfpq": 0.449},
    },
}
OUTPUT_FILES = {
    "TriviaQA": os.path.join(DOCS_DIR, "accuracy_ivfpq_vs_diskann_triviaqa.png"),
    "NaturalQS": os.path.join(DOCS_DIR, "accuracy_ivfpq_vs_diskann_naturalqs.png"),
}


def percent_annotation(ax: plt.Axes, fmt: str = "{:.1f}%", y_offset: int = 4) -> None:
    """Annotate each bar in *ax* with a percentage label."""

    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.annotate(
            fmt.format(height * 100.0),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            xytext=(0, y_offset),
            textcoords="offset points",
        )


def build_dataset(dataset: str) -> Tuple[list[str], list[float], list[float]]:
    """Return ordered metric labels and values for a given dataset."""

    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'")

    metrics = DATASETS[dataset]
    labels = METRIC_ORDER
    ivfpq_vals = [metrics[metric]["ivfpq"] for metric in labels]
    diskann_vals = [metrics[metric]["diskann"] for metric in labels]

    return labels, ivfpq_vals, diskann_vals


def plot_dataset(dataset: str) -> str:
    """Create a side-by-side bar chart for a single dataset."""

    if sns:
        sns.set_theme(style="whitegrid")

    labels, ivfpq_vals, diskann_vals = build_dataset(dataset)
    x_positions = list(range(len(labels)))
    bar_width = 0.36

    plt.figure(figsize=(7.5, 4.5))
    ax = plt.gca()

    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        ivfpq_vals,
        width=bar_width,
        label="IVFPQ",
        color=IVFPQ_COLOR,
    )
    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        diskann_vals,
        width=bar_width,
        label="DiskANN",
        color=DISKANN_COLOR,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title(f"{dataset} Accuracy: IVFPQ vs DiskANN", fontsize=16)
    ax.legend(fontsize=12)

    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontsize(12)

    all_vals = ivfpq_vals + diskann_vals
    vmin = min(all_vals)
    vmax = max(all_vals)
    padding = max(0.02, (vmax - vmin) * 0.2)
    ax.set_ylim(max(0.0, vmin - padding), min(1.0, vmax + padding))

    percent_annotation(ax)
    plt.tight_layout()

    out_path = OUTPUT_FILES[dataset]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def main() -> None:
    outputs = [plot_dataset(dataset) for dataset in DATASETS.keys()]
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()


