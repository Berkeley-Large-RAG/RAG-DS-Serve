import os
import math
import argparse

# Use a non-interactive backend for servers without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # Optional, falls back if unavailable
except Exception:
    sns = None


def ensure_output_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def annotate_bars(ax) -> None:
    for p in ax.patches:
        height = p.get_height()
        if not math.isfinite(height):
            continue
        if height <= 0:
            continue
        ax.annotate(f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom",
                    fontsize=12, fontweight='bold', rotation=0, xytext=(0, 3), textcoords="offset points")


def plot_diskann_qps(data, out_dir: str) -> None:
    # Sort by L ascending
    rows = sorted(data, key=lambda r: r["L"])
    x = [str(r["L"]) for r in rows]
    y = [r["QPS"] for r in rows]

    plt.figure(figsize=(9, 4))
    bar_color = "#80b1d3"  # darker Set3 blue
    if sns:
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=x, y=y, color=bar_color)
    else:
        ax = plt.bar(x, y, color=bar_color)
        ax = plt.gca()
    plt.title("DiskANN QPS vs L")
    plt.xlabel("L")
    plt.ylabel("QPS")
    # Add headroom so value labels don't clip at the top
    try:
        ymax = max(y)
        if math.isfinite(ymax) and ymax > 0:
            plt.ylim(0, ymax * 1.15)
    except Exception:
        pass
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diskann_qps_vs_L.png"), dpi=160)
    plt.close()


def plot_diskann_vs_ivfpq_summary(diskann_rows, faiss_rows, out_dir: str) -> None:
    """Throughput-only comparison with clear L/nprobe labels, more bars."""
    colors = {"DiskANN": "#fb8072", "IVFPQ": "#80b1d3"}  # darker Set3 (swapped for consistency)

    labels = []
    vals = []
    bar_colors = []
    for r in sorted(diskann_rows, key=lambda r: r["L"]):
        labels.append(f"DiskANN L={r['L']}")
        vals.append(r["QPS"])
        bar_colors.append(colors["DiskANN"])
    for r in sorted(faiss_rows, key=lambda r: r["nprobe"]):
        labels.append(f"IVFPQ nprobe={r['nprobe']}")
        vals.append(r["QPS"])
        bar_colors.append(colors["IVFPQ"])

    plt.figure(figsize=(9, 4))  # consistent size with latency figure
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()
    ax.bar(range(len(labels)), vals, color=bar_colors, edgecolor='none')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("QPS (↑ higher is better)")
    # No title - section title serves as figure title
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=colors["DiskANN"], label="DiskANN"),
        plt.Rectangle((0, 0), 1, 1, color=colors["IVFPQ"], label="IVFPQ"),
    ], fontsize=11)

    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")

    try:
        ymax = max(vals)
        if math.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.15)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diskann_vs_ivfpq_qps_multi.png"), dpi=160)
    plt.close()


def plot_diskann_vs_ivfpq_latency(diskann_rows, faiss_rows, out_dir: str) -> None:
    """Single-request latency comparison: DiskANN vs IVFPQ with same color scheme."""
    colors = {"DiskANN": "#fb8072", "IVFPQ": "#80b1d3"}  # same as throughput

    # Use recommended configs: DiskANN L=1000, IVFPQ nprobe=256
    diskann_1000 = next((r for r in diskann_rows if r["L"] == 1000), diskann_rows[0])
    ivfpq_256 = next((r for r in faiss_rows if r["nprobe"] == 256), faiss_rows[0])

    labels = ["DiskANN (L=1000)", "IVFPQ (nprobe=256)"]
    # Total latency for single request
    vals = [diskann_1000["total_ms"], ivfpq_256["total_ms"]]
    bar_colors = [colors["DiskANN"], colors["IVFPQ"]]

    plt.figure(figsize=(9, 4))  # same size as throughput figure
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()
    ax.bar(labels, vals, color=bar_colors, edgecolor='none')

    ax.set_ylabel("Latency (ms) (↓ lower is better)")
    # No title - section title serves as figure title
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=colors["DiskANN"], label="DiskANN"),
        plt.Rectangle((0, 0), 1, 1, color=colors["IVFPQ"], label="IVFPQ"),
    ], fontsize=11)

    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")

    try:
        ymax = max(vals)
        if math.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.15)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diskann_vs_ivfpq_latency.png"), dpi=160)
    plt.close()


def plot_diskann_vs_ivfpq_perf_combined(diskann_rows, faiss_rows, out_dir: str) -> None:
    """Combined throughput + latency figure for recommended configs only (L=1000, nprobe=256)."""
    colors = {"DiskANN": "#fb8072", "IVFPQ": "#80b1d3"}

    # Get recommended configs
    diskann_1000 = next((r for r in diskann_rows if r["L"] == 1000), diskann_rows[0])
    ivfpq_256 = next((r for r in faiss_rows if r["nprobe"] == 256), faiss_rows[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
    if sns:
        sns.set_theme(style="whitegrid")

    labels = ["DiskANN", "IVFPQ"]
    bar_colors = [colors["DiskANN"], colors["IVFPQ"]]

    # Throughput subplot
    qps_vals = [diskann_1000["QPS"], ivfpq_256["QPS"]]
    ax1.bar(labels, qps_vals, color=bar_colors, edgecolor='none')
    ax1.set_ylabel("QPS (↑ higher is better)")
    ax1.set_title("Batched throughput", fontsize=12)
    for i, v in enumerate(qps_vals):
        ax1.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    ax1.set_ylim(0, max(qps_vals) * 1.15)

    # Latency subplot
    lat_vals = [diskann_1000["total_ms"], ivfpq_256["total_ms"]]
    ax2.bar(labels, lat_vals, color=bar_colors, edgecolor='none')
    ax2.set_ylabel("Latency (ms) (↓ lower is better)")
    ax2.set_title("Single-request latency", fontsize=12)
    for i, v in enumerate(lat_vals):
        ax2.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    ax2.set_ylim(0, max(lat_vals) * 1.15)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diskann_vs_ivfpq_perf.png"), dpi=160)
    plt.close(fig)


def annotate_bars_custom(ax, fontsize: int = 10, y_offset: int = 4) -> None:
    for p in ax.patches:
        height = p.get_height()
        if not math.isfinite(height):
            continue
        if height <= 0:
            continue
        ax.annotate(f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom",
                    fontsize=fontsize, fontweight='bold', rotation=0, xytext=(0, y_offset), textcoords="offset points")

def plot_diskann_latency_breakdown(data, out_dir: str) -> None:
    # metrics to plot as grouped bars
    metrics = [
        ("embed_ms", "Embed"),
        ("DA_batch_ms", "Index search"),
        ("map_ms", "Passage map"),
        ("total_ms", "Total"),
    ]

    rows = sorted(data, key=lambda r: r["L"])
    labels = [str(r["L"]) for r in rows]

    # Build a matrix of values
    values_by_metric = []
    for key, _disp in metrics:
        values_by_metric.append([r[key] for r in rows])

    num_groups = len(labels)
    num_bars = len(metrics)
    x = range(num_groups)
    width = 0.85 / num_bars

    plt.figure(figsize=(12, 5))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    palette = ["#80b1d3", "#fb8072", "#b3de69", "#fdb462", "#bc80bd"]  # darker Set3 tones
    for idx, (series, (key, disp)) in enumerate(zip(values_by_metric, metrics)):
        bar_positions = [i + idx * width - (width * (num_bars - 1) / 2) for i in x]
        ax.bar(bar_positions, series, width=width, label=disp, color=palette[idx % len(palette)])

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("L")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("DiskANN Latency Breakdown")
    ax.legend(ncol=4, fontsize=11, loc="upper left")

    # Add headroom so annotations don't clip
    try:
        ymax = max(max(series) for series in values_by_metric if series)
        if math.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.20)
    except Exception:
        pass

    # Annotate every bar with larger, clearer labels
    for idx, (series, _m) in enumerate(zip(values_by_metric, metrics)):
        for i, val in enumerate(series):
            if math.isfinite(val) and val > 0:
                xpos = i + idx * width - (width * (num_bars - 1) / 2)
                ax.annotate(f"{val:.2f}", (xpos, val), ha="center", va="bottom",
                            fontsize=12, fontweight='bold', xytext=(0, 4), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diskann_latency_breakdown_vs_L.png"), dpi=160)
    plt.close()


def plot_faiss_qps_and_latency(data, out_dir: str, which: str = "both") -> None:
    rows = sorted(data, key=lambda r: r["nprobe"])
    x = [str(r["nprobe"]) for r in rows]

    # QPS
    if which in ("both", "qps"):
        y_qps = [r["QPS"] for r in rows]
        plt.figure(figsize=(9, 4))
        if sns:
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(x=x, y=y_qps, color="#4C78A8")
        else:
            ax = plt.bar(x, y_qps, color="#4C78A8")
            ax = plt.gca()
        plt.title("IVF_PQ QPS")
        plt.xlabel("nprobe")
        plt.ylabel("QPS")
        # Add headroom so value labels don't clip at the top
        try:
            ymax = max(y_qps)
            if math.isfinite(ymax) and ymax > 0:
                plt.ylim(0, ymax * 1.15)
        except Exception:
            pass
        annotate_bars(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "faiss_qps_vs_nprobe.png"), dpi=160)
        plt.close()

    # Latency grouped bars
    if which in ("both", "latency"):
        metrics = [
            ("embed_ms", "Embed"),
            ("search_ms", "Search (server)"),
            ("total_ms", "Total"),
        ]
        values_by_metric = []
        for key, _disp in metrics:
            values_by_metric.append([r[key] for r in rows])

        num_groups = len(rows)
        num_bars = len(metrics)
        x_idx = range(num_groups)
        width = 0.85 / num_bars

        plt.figure(figsize=(12, 5))
        if sns:
            sns.set_theme(style="whitegrid")
        ax = plt.gca()
        palette = ["#4C78A8", "#F58518", "#54A24B"]
        for idx, (series, (key, disp)) in enumerate(zip(values_by_metric, metrics)):
            bar_positions = [i + idx * width - (width * (num_bars - 1) / 2) for i in x_idx]
            ax.bar(bar_positions, series, width=width, label=disp, color=palette[idx % len(palette)])

        ax.set_xticks(list(x_idx))
        ax.set_xticklabels([str(r["nprobe"]) for r in rows])
        ax.set_xlabel("nprobe")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("IVF_PQ ANN Latency")
        ax.legend(ncol=3, fontsize=9, loc="upper left")

        # Add headroom so annotations don't clip
        try:
            ymax = max(max(series) for series in values_by_metric if series)
            if math.isfinite(ymax) and ymax > 0:
                ax.set_ylim(0, ymax * 1.20)
        except Exception:
            pass

        # Annotate every bar with larger, clearer labels
        for idx, series in enumerate(values_by_metric):
            for i, val in enumerate(series):
                if math.isfinite(val) and val > 0:
                    xpos = i + idx * width - (width * (num_bars - 1) / 2)
                    ax.annotate(f"{val:.2f}", (xpos, val), ha="center", va="bottom",
                                fontsize=12, fontweight='bold', xytext=(0, 4), textcoords="offset points")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "faiss_latency_vs_nprobe.png"), dpi=160)
        plt.close()


def plot_diskann_index_only_qps(data, out_dir: str) -> None:
    rows = sorted(data, key=lambda r: r["L"]) 
    x = [str(r["L"]) for r in rows]
    y = [r["QPS"] for r in rows]

    plt.figure(figsize=(9, 4))
    if sns:
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=x, y=y, color="#72B7B2")
    else:
        ax = plt.bar(x, y, color="#72B7B2")
        ax = plt.gca()
    plt.title("DiskANN Index-only QPS")
    plt.xlabel("L")
    plt.ylabel("QPS")
    # Add headroom so value labels don't clip at the top
    try:
        ymax = max(y)
        if math.isfinite(ymax) and ymax > 0:
            plt.ylim(0, ymax * 1.15)
    except Exception:
        pass
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diskann_index_only_qps_vs_L.png"), dpi=160)
    plt.close()


def plot_accuracy_triviaqa_faiss_vs_diskann(out_dir: str) -> None:
    labels = ["Exact match", "F1", "Recall"]
    faiss_vals = [0.584511447516577, 0.6708467199132743, 0.8836481921681472]
    diskann_vals = [0.635431002126861, 0.72239609694681, 0.8977855623670712]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 4))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    faiss_color = "#4C78A8"
    diskann_color = "#F58518"

    ax.bar([i - width / 2 for i in x], faiss_vals, width=width, label="FAISS", color=faiss_color)
    ax.bar([i + width / 2 for i in x], diskann_vals, width=width, label="DiskANN", color=diskann_color)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("TriviaQA (validation) Accuracy: FAISS vs DiskANN")
    ax.legend()

    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_triviaqa_faiss_vs_diskann.png"), dpi=160)
    plt.close()


def plot_accuracy_nqopen_faiss_vs_diskann(out_dir: str) -> None:
    labels = ["Exact match", "F1"]
    faiss_vals = [0.2155124653739612, 0.33179501385041454]
    diskann_vals = [0.22770083102493074, 0.3470803324099715]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(7.5, 4))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    faiss_color = "#4C78A8"
    diskann_color = "#F58518"

    ax.bar([i - width / 2 for i in x], faiss_vals, width=width, label="FAISS", color=faiss_color)
    ax.bar([i + width / 2 for i in x], diskann_vals, width=width, label="DiskANN", color=diskann_color)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("NQ-Open (validation) Accuracy: FAISS vs DiskANN")
    ax.legend()

    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_nqopen_faiss_vs_diskann.png"), dpi=160)
    plt.close()


def plot_accuracy_combined_faiss_vs_diskann(out_dir: str) -> None:
    # Order: TriviaQA (EM, F1, Recall), NQ-Open (EM, F1)
    labels = [
        ("TriviaQA", "Exact match"),
        ("TriviaQA", "F1"),
        ("TriviaQA", "Recall"),
        ("NQ-Open", "Exact match"),
        ("NQ-Open", "F1"),
    ]

    faiss_vals = [
        0.584511447516577,  # TriviaQA EM
        0.6708467199132743, # TriviaQA F1
        0.8836481921681472, # TriviaQA Recall
        0.2155124653739612, # NQ-Open EM
        0.33179501385041454 # NQ-Open F1
    ]

    diskann_vals = [
        0.635431002126861,  # TriviaQA EM
        0.72239609694681,   # TriviaQA F1
        0.8977855623670712, # TriviaQA Recall
        0.22770083102493074,# NQ-Open EM
        0.3470803324099715  # NQ-Open F1
    ]

    x = range(len(labels))
    width = 0.36

    plt.figure(figsize=(12, 5))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    faiss_color = "#4C78A8"
    diskann_color = "#F58518"

    ax.bar([i - width / 2 for i in x], faiss_vals, width=width, label="FAISS", color=faiss_color)
    ax.bar([i + width / 2 for i in x], diskann_vals, width=width, label="DiskANN", color=diskann_color)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{ds}\n{m}" for ds, m in labels])
    ax.set_ylabel("Score", fontsize=14, fontweight='bold')
    ax.set_title("ANN vs DiskANN", fontsize=18, fontweight='bold')
    ax.legend(fontsize=12)

    # Bold tick labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(12)
        lbl.set_fontweight('bold')

    # Zoom Y scale around the data to make differences clearer
    _vals = faiss_vals + diskann_vals
    _vmin, _vmax = min(_vals), max(_vals)
    _pad = max(0.02, (_vmax - _vmin) * 0.15)
    ax.set_ylim(max(0.0, _vmin - _pad), min(1.0, _vmax + _pad))

    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_faiss_vs_diskann_triviaqa_nq.png"), dpi=160)
    plt.close()


def plot_compact_ds_ann_exact_diskann(out_dir: str) -> None:
    # Datasets on x-axis
    datasets = ["MMLU", "MMLU Pro", "MATH"]

    # Values from the user's table (This datastore; last two rows) and DiskANN sheet
    no_retr = [68.9, 39.8, 46.9]
    ann_only = [73.6, 46.8, 51.6]
    ann_exact = [74.0, 48.6, 54.0]
    diskann   = [74.2, 48.3, 52.0]

    x = list(range(len(datasets)))
    width = 0.20

    plt.figure(figsize=(10, 4.5))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    color_no_retr = "#B279A2"    # purple baseline
    color_ann_only = "#4C78A8"   # blue
    color_ann_exact = "#54A24B"  # green
    color_diskann = "#F58518"    # orange

    ax.bar([i - 1.5*width for i in x], no_retr,  width=width, label="No Retrieval", color=color_no_retr)
    ax.bar([i - 0.5*width for i in x], ann_only, width=width, label="ANN Only", color=color_ann_only)
    ax.bar([i + 0.5*width for i in x], ann_exact, width=width, label="ANN + Exact", color=color_ann_exact)
    ax.bar([i + 1.5*width for i in x], diskann,  width=width, label="DiskANN", color=color_diskann)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("DiskANN and IVF_PQ ANN Accuracy Comparison", fontsize=18)
    ax.legend(fontsize=12)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(12)

    # Zoom Y scale for clarity (scores are 0-100 range here)
    vals = no_retr + ann_only + ann_exact + diskann
    vmin, vmax = min(vals), max(vals)
    pad = max(1.0, (vmax - vmin) * 0.15)
    ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

    annotate_bars_custom(ax, fontsize=9, y_offset=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_compact_ds_ann_exact_diskann.png"), dpi=160)
    plt.close()


def plot_ann_diskann_full_table_bars(out_dir: str) -> None:
    # Datasets and scores (trimmed to 3 tasks; add No Retrieval)
    datasets = ["MMLU", "MMLU Pro", "MATH"]
    no_retr = [68.9, 39.8, 46.9]
    ann_only = [73.6, 46.8, 51.6]
    ann_exact = [74.0, 48.6, 54.0]
    diskann  = [74.2, 48.3, 52.0]

    x = list(range(len(datasets)))
    width = 0.20

    plt.figure(figsize=(10.5, 4.5))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    color_no_retr = "#B279A2"
    color_ann_only = "#4C78A8"
    color_ann_exact = "#54A24B"
    color_diskann = "#F58518"

    ax.bar([i - 1.5*width for i in x], no_retr,  width=width, label="No Retrieval", color=color_no_retr)
    ax.bar([i - 0.5*width for i in x], ann_only, width=width, label="ANN Only", color=color_ann_only)
    ax.bar([i + 0.5*width for i in x], ann_exact, width=width, label="ANN + Exact", color=color_ann_exact)
    ax.bar([i + 1.5*width for i in x], diskann,  width=width, label="DiskANN", color=color_diskann)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("DiskANN and IVF_PQ ANN Accuracy Comparison", fontsize=18)
    ax.legend(fontsize=12)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(12)

    vals = no_retr + ann_only + ann_exact + diskann
    vmin, vmax = min(vals), max(vals)
    pad = max(1.0, (vmax - vmin) * 0.15)
    ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

    annotate_bars_custom(ax, fontsize=9, y_offset=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_ann_diskann_full_table_bars.png"), dpi=160)
    plt.close()


def plot_ann_diskann_full_table_lines(out_dir: str) -> None:
    datasets = ["MMLU", "MMLU Pro", "AGI Eval", "MATH", "GPQA"]
    ann_only = [73.6, 46.8, 57.5, 51.6, 30.8]
    diskann  = [74.2, 48.3, 57.1, 52.0, 31.2]
    ann_exact = [74.0, 48.6, 57.4, 54.0, 35.7]

    x = list(range(len(datasets)))

    plt.figure(figsize=(11, 4.5))
    if sns:
        sns.set_theme(style="whitegrid")
    ax = plt.gca()

    color_ann_only = "#4C78A8"
    color_ann_exact = "#54A24B"
    color_diskann = "#F58518"

    ax.plot(x, ann_only, marker='o', color=color_ann_only, label="ANN Only", linewidth=2)
    ax.plot(x, ann_exact, marker='^', color=color_ann_exact, label="ANN + Exact", linewidth=2)
    ax.plot(x, diskann, marker='s', color=color_diskann, label="DiskANN", linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Score", fontsize=14, fontweight='bold')
    ax.set_title("DiskANN vs ANN", fontsize=18, fontweight='bold')
    ax.legend(fontsize=12)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(12)
        lbl.set_fontweight('bold')

    vals = ann_only + ann_exact + diskann
    vmin, vmax = min(vals), max(vals)
    pad = max(1.0, (vmax - vmin) * 0.15)
    ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

    # Annotate points with two decimals
    for xi, yi in zip(x, ann_only):
        if math.isfinite(yi):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=12, fontweight='bold')
    for xi, yi in zip(x, ann_exact):
        if math.isfinite(yi):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=12, fontweight='bold')
    for xi, yi in zip(x, diskann):
        if math.isfinite(yi):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_ann_diskann_full_table_lines.png"), dpi=160)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DS-Serve plots")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[
            "diskann_qps",
            "diskann_latency",
            "faiss_qps",
            "faiss_latency",
            "diskann_index_only",
            "accuracy_triviaqa",
            "accuracy_nqopen",
            "accuracy_combined",
            "compact_ds_ann_exact_diskann",
            "accuracy_full_table_bars",
            "accuracy_full_table_lines",
        ],
        help="Generate only the specified figure(s)",
    )
    args = parser.parse_args()

    # Resolve output directory relative to this script to avoid hardcoded paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "docs", "plots")
    ensure_output_dir(out_dir)

    # === DiskANN E2E data (from user) ===
    diskann_rows = [
        {"L": 5000, "Reqs": 10000, "Duration_s": 68, "QPS": 147.06, "embed_ms": 2.59, "search_ms": 3.77, "total_ms": 6.36, "DA_batch_ms": 2.42, "map_ms": 1.34},
        {"L": 4000, "Reqs": 10000, "Duration_s": 50, "QPS": 200.00, "embed_ms": 2.36, "search_ms": 2.27, "total_ms": 4.63, "DA_batch_ms": 1.97, "map_ms": 0.30},
        {"L": 3000, "Reqs": 10000, "Duration_s": 50, "QPS": 200.00, "embed_ms": 2.69, "search_ms": 1.82, "total_ms": 4.52, "DA_batch_ms": 1.48, "map_ms": 0.34},
        {"L": 2000, "Reqs": 10000, "Duration_s": 43, "QPS": 232.56, "embed_ms": 2.52, "search_ms": 1.40, "total_ms": 3.92, "DA_batch_ms": 0.99, "map_ms": 0.42},
        {"L": 1000, "Reqs": 10000, "Duration_s": 42, "QPS": 238.10, "embed_ms": 2.61, "search_ms": 1.17, "total_ms": 3.78, "DA_batch_ms": 0.50, "map_ms": 0.67},
    ]

    # === FAISS E2E data (from user) ===
    faiss_rows = [
        {"nprobe": 256, "Reqs": 4096, "Duration_s": 45, "QPS": 91.02,  "embed_ms": 1.87, "search_ms": 8.70, "total_ms": 10.57},
        {"nprobe": 128, "Reqs": 4096, "Duration_s": 35, "QPS": 117.03, "embed_ms": 1.80, "search_ms": 6.27, "total_ms": 8.07},
        {"nprobe":  64, "Reqs": 4096, "Duration_s": 30, "QPS": 136.53, "embed_ms": 1.96, "search_ms": 5.02, "total_ms": 6.98},
        {"nprobe":  32, "Reqs": 4096, "Duration_s": 28, "QPS": 146.29, "embed_ms": 1.86, "search_ms": 4.39, "total_ms": 6.25},
    ]

    # === DiskANN index-only data (from user) ===
    idx_only_rows = [
        {"L": 1000, "Beamwidth": 4, "QPS": 2019.40, "Mean_Latency": 62729.66, "P999_Latency": 173463.00, "Mean_IOs": 1051.40, "Mean_IO_us": 45159.21, "CPU_s": 15218.08},
        {"L": 2000, "Beamwidth": 4, "QPS":  952.53, "Mean_Latency": 133149.38, "P999_Latency": 237803.00, "Mean_IOs": 2049.87, "Mean_IO_us":  97681.02, "CPU_s": 30773.26},
        {"L": 3000, "Beamwidth": 4, "QPS":  656.03, "Mean_Latency": 193506.00, "P999_Latency": 326917.00, "Mean_IOs": 3047.38, "Mean_IO_us": 139010.82, "CPU_s": 47450.94},
        {"L": 4000, "Beamwidth": 4, "QPS":  496.58, "Mean_Latency": 255526.36, "P999_Latency": 395824.00, "Mean_IOs": 4045.19, "Mean_IO_us": 180862.49, "CPU_s": 65208.78},
        {"L": 5000, "Beamwidth": 4, "QPS":  403.84, "Mean_Latency": 314023.55, "P999_Latency": 478652.00, "Mean_IOs": 5043.97, "Mean_IO_us": 211945.95, "CPU_s": 89573.35},
    ]

    # Generate plots
    if args.only:
        actions = {
            "diskann_qps": lambda: plot_diskann_qps(diskann_rows, out_dir),
            "diskann_latency": lambda: plot_diskann_latency_breakdown(diskann_rows, out_dir),
            "diskann_vs_ivfpq": lambda: plot_diskann_vs_ivfpq_summary(diskann_rows, faiss_rows, out_dir),
            "faiss_qps": lambda: plot_faiss_qps_and_latency(faiss_rows, out_dir, which="qps"),
            "faiss_latency": lambda: plot_faiss_qps_and_latency(faiss_rows, out_dir, which="latency"),
            "diskann_index_only": lambda: plot_diskann_index_only_qps(idx_only_rows, out_dir),
            "accuracy_triviaqa": lambda: plot_accuracy_triviaqa_faiss_vs_diskann(out_dir),
            "accuracy_nqopen": lambda: plot_accuracy_nqopen_faiss_vs_diskann(out_dir),
            "accuracy_combined": lambda: plot_accuracy_combined_faiss_vs_diskann(out_dir),
            "compact_ds_ann_exact_diskann": lambda: plot_compact_ds_ann_exact_diskann(out_dir),
            "accuracy_full_table_bars": lambda: plot_ann_diskann_full_table_bars(out_dir),
            "accuracy_full_table_lines": lambda: plot_ann_diskann_full_table_lines(out_dir),
        }
        for key in args.only:
            actions[key]()
    else:
        plot_diskann_qps(diskann_rows, out_dir)
        plot_diskann_latency_breakdown(diskann_rows, out_dir)
        plot_diskann_vs_ivfpq_summary(diskann_rows, faiss_rows, out_dir)
        plot_faiss_qps_and_latency(faiss_rows, out_dir)
        plot_diskann_index_only_qps(idx_only_rows, out_dir)
        plot_accuracy_triviaqa_faiss_vs_diskann(out_dir)
        plot_accuracy_nqopen_faiss_vs_diskann(out_dir)
        plot_accuracy_combined_faiss_vs_diskann(out_dir)
        plot_compact_ds_ann_exact_diskann(out_dir)
        plot_ann_diskann_full_table_bars(out_dir)
        plot_ann_diskann_full_table_lines(out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()


