#!/usr/bin/env python3
"""Generate throughput comparison plots for Google API vs DS Serve (single and batched)."""

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


def make_plot(
    title: str,
    data: List[Tuple[str, float]],
    output_path: str,
    bar_colors: Dict[str, str],
) -> None:
    if sns:
        sns.set_theme(style="whitegrid")

    labels = [label for label, _ in data]
    values = [val for _, val in data]
    colors = [bar_colors.get(label, "#4C78A8") for label in labels]

    plt.figure(figsize=(5.2, 4.2))
    ax = plt.gca()
    ax.bar(labels, values, color=colors)
    ax.set_xlabel("System", fontsize=13)
    ax.set_ylabel("QPS", fontsize=13)
    ax.set_title(title, fontsize=14)

    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontsize(11)

    annotate(ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def main() -> None:
    colors = {
        "Google API": "#E45756",  # red
        "DS Serve": "#4C78A8",    # blue
    }

    single = [
        ("Google API", 6.6055),
        ("DS Serve", 12.4713),
    ]
    batched = [
        ("Google API", 8.2488),
        ("DS Serve", 238.10),
    ]

    make_plot(
        title="Throughput (single)",
        data=single,
        output_path=os.path.join(PLOTS_DIR, "search_engine_qps_single.png"),
        bar_colors=colors,
    )

    make_plot(
        title="Throughput (batched)",
        data=batched,
        output_path=os.path.join(PLOTS_DIR, "search_engine_qps_batched.png"),
        bar_colors=colors,
    )


if __name__ == "__main__":
    main()

