# Quick tests

Helper scripts in `scripts/` reproduce the QPS/latency sweeps used in the docs. Run them from the repo root and override parameters via environment variables.

## DiskANN batched benchmark
Continuous POST with shared payload (Figures 4a–4b).
```bash
L_LIST="100 500 1000 1500 2000" \
COUNT=2000 \
HOST=http://api.ds-serve.org:30888 \
scripts/diskann_qps_batch.sh
```
Knobs: `HOST`, `QUERIES`, `COUNT`, `CONCURRENCY`, `K`, `W`, `THREADS`, `L_LIST`.

## DiskANN single-request benchmark
One POST per query (Figure 6).
```bash
COUNT=1000 \
WARMUP_SKIP=100 \
L_LIST="100 500 1000 1500 2000" \
scripts/diskann_qps_single.sh
```
Supports the same overrides as the batched script plus `WARMUP_SKIP` to drop warmup queries.

## FAISS / IVFPQ batched benchmark
```bash
COUNT=100 \
NPROBE_LIST="64 128 256 512" \
scripts/ivfpq_qps_batch.sh
```
Set `K`, `NPROBE`, `EXACT`, `DIVERSE`, `LAMBDA`, `HOST`, and `QUERIES` as needed. When `NPROBE_LIST` is provided the script sweeps through each value using the same shuffled sample.

## FAISS / IVFPQ single-request benchmark
```bash
COUNT=100 \
CONCURRENCY=64 \
NPROBE_LIST="64 128 256 512" \
scripts/ivfpq_qps_single.sh
```
Each query is sent independently; `CONCURRENCY` controls the `xargs -P` fan-out.

## Regenerating plots
After collecting new measurements, run:
```bash
python scripts/plot_diskann_single_request_qps.py
python scripts/plot_diskann_single_request_latency.py
python scripts/plot_faiss_batch_vs_single.py
```
These write to `docs/plots/` and keep the figures used in `docs/index.md` in sync.

