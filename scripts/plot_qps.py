import os
import math

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
        ax.annotate(f"{height:.1f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom",
                    fontsize=8, rotation=0, xytext=(0, 3), textcoords="offset points")


def plot_diskann_qps(data, out_dir: str) -> None:
    # Sort by L ascending
    rows = sorted(data, key=lambda r: r["L"])
    x = [str(r["L"]) for r in rows]
    y = [r["QPS"] for r in rows]

    plt.figure(figsize=(9, 4))
    if sns:
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=x, y=y, color="#4C78A8")
    else:
        ax = plt.bar(x, y, color="#4C78A8")
        ax = plt.gca()
    plt.title("DiskANN QPS vs L")
    plt.xlabel("L")
    plt.ylabel("QPS")
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diskann_qps_vs_L.png"), dpi=160)
    plt.close()


def plot_diskann_latency_breakdown(data, out_dir: str) -> None:
    # metrics to plot as grouped bars
    metrics = [
        ("embed_ms", "Embed"),
        ("search_ms", "Search (server)"),
        ("DA_batch_ms", "DiskANN batch"),
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

    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    for idx, (series, (key, disp)) in enumerate(zip(values_by_metric, metrics)):
        bar_positions = [i + idx * width - (width * (num_bars - 1) / 2) for i in x]
        ax.bar(bar_positions, series, width=width, label=disp, color=palette[idx % len(palette)])

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("L")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("DiskANN Latency Breakdown vs L")
    ax.legend(ncol=3, fontsize=9)

    # Annotate only the Total bars to reduce clutter
    total_index = [i for i, (k, _d) in enumerate(metrics) if k == "total_ms"][0]
    for i, r in enumerate(rows):
        xpos = i + total_index * width - (width * (num_bars - 1) / 2)
        h = r["total_ms"]
        if math.isfinite(h) and h > 0:
            ax.annotate(f"{h:.2f}", (xpos, h), ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diskann_latency_breakdown_vs_L.png"), dpi=160)
    plt.close()


def plot_faiss_qps_and_latency(data, out_dir: str) -> None:
    rows = sorted(data, key=lambda r: r["nprobe"])
    x = [str(r["nprobe"]) for r in rows]

    # QPS
    y_qps = [r["QPS"] for r in rows]
    plt.figure(figsize=(9, 4))
    if sns:
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=x, y=y_qps, color="#4C78A8")
    else:
        ax = plt.bar(x, y_qps, color="#4C78A8")
        ax = plt.gca()
    plt.title("FAISS QPS vs nprobe")
    plt.xlabel("nprobe")
    plt.ylabel("QPS")
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "faiss_qps_vs_nprobe.png"), dpi=160)
    plt.close()

    # Latency grouped bars
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
    ax.set_title("FAISS Latency vs nprobe")
    ax.legend(ncol=3, fontsize=9)

    # Annotate Total bars
    total_index = [i for i, (k, _d) in enumerate(metrics) if k == "total_ms"][0]
    for i, r in enumerate(rows):
        xpos = i + total_index * width - (width * (num_bars - 1) / 2)
        h = r["total_ms"]
        if math.isfinite(h) and h > 0:
            ax.annotate(f"{h:.2f}", (xpos, h), ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")

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
    plt.title("DiskANN Index-only QPS vs L (Beamwidth=4)")
    plt.xlabel("L")
    plt.ylabel("QPS")
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diskann_index_only_qps_vs_L.png"), dpi=160)
    plt.close()


def main() -> None:
    out_dir = "/mnt/md-256k/jinjian/DS/runtime/plots"
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
    plot_diskann_qps(diskann_rows, out_dir)
    plot_diskann_latency_breakdown(diskann_rows, out_dir)
    plot_faiss_qps_and_latency(faiss_rows, out_dir)
    plot_diskann_index_only_qps(idx_only_rows, out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()


