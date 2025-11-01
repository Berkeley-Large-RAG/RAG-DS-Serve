## Introduction

This repository contains the Compact‑DS Dive Public API and server code. It exposes a production‑ready Flask service for retrieval‑augmented generation (RAG) backed by a billion‑scale FAISS IVFPQ index. The server provides adjustable settings and search modes at low-latency. A small CLI helps download/prepare indices and start the server, and an example domain (`index_dev/`) shows the expected data layout. Please refer to the Quickstart below to set up.

### Expected data layout (under DATASTORE_PATH)

```
<DATASTORE_PATH>/
  <domain_name>/
    config.json              # loader config (encoder, nprobe, index filename, etc.)
    index/                   # a single merged FAISS file (*.faiss)
    passages/                # *.jsonl shards for text lookup
```

## Quickstart 

### Datasets
- [CompactDS-102GB](https://huggingface.co/datasets/alrope/CompactDS-102GB)
  - Core index and passages. Please refer to the dataset card for details.
- [Full embeddings](https://huggingface.co/datasets/alrope/cpds_embeddings/tree/main)
  - PubMed embeddings are sharded, so combine locally if needed:
```bash
cat massiveds-pubmed--passages7_00.pkl_{aa,ab,ac,ad,ae,af,ag,ah,ai} \
  > massiveds-pubmed--passages7_00.pkl
```
### 0) Prepare the repo

```
git clone https://github.com/Berkeley-Large-RAG/RAG-DS-Serve.git 
cd RAG-DS-Serve
git submodule update --init --recursive
```

### 1) Download the dataset/index from Hugging Face

- Choose a local data root (DATASTORE_PATH). Example:
```bash
export DATASTORE_PATH=/home/ubuntu/massive-serve-dev
```

- Download the dataset into `$DATASTORE_PATH/<domain_name>` (example uses `index_dev`):
```bash
huggingface-cli download <ORG_OR_USER>/<DATASET_REPO> \
  --repo-type dataset \
  --local-dir $DATASTORE_PATH/index_dev
```

Notes:
- The directory should include an `index/` directory with a FAISS index and a `passages/` directory with `.jsonl` files.
- If your index is uploaded in split/chunked form, see Step 3 to combine shards.

### 2) Build the position mapping arrays 

The server looks up passage text by FAISS index id using position mapping arrays. Generate them once from your `passages/` directory:

- Open `build_arr.py` and set `INPUT_DIR` to your passages folder, e.g.:
```python
INPUT_DIR = "/home/ubuntu/massive-serve-dev/index_dev/passages"
```

- Then run from the repo root:
```bash
python build_arr.py
```

This writes three files next to the script (and the server expects them under `index_dev/` as configured by the code):
- `index_dev/position_array.npy`
- `index_dev/filename_index_array.npy`
- `index_dev/filename_list.npy`

### 3) Combine FAISS index shards

Your `index/` folder contains split parts, combine them into a single `.faiss` file before serving.

- Simple shard set (concatenate all `...faiss_**` parts in order; do NOT include the `.meta` file):
```bash
cd $DATASTORE_PATH/index_dev/index
# Example names: index_IVFPQ.100000000.768.65536.64.faiss_aa, ..._ab, ..._ac, ...
cat $(ls index_IVFPQ.100000000.768.65536.64.faiss_* | sort) > index_full.faiss
```

After combining, ensure there is exactly one `.faiss` file in `index/` (e.g., `index/index_full.faiss`).

### 4) Launch the API server

Use `index_dev` as the domain name (provided by `index_dev/config.json`):
```bash
DATASTORE_PATH=/home/ubuntu/massive-serve-dev \
python -m massive_serve.cli serve --domain_name index_dev
```

By default the server starts at port `30888` and exposes `/search` and `/vote` endpoints.

### 5) Test the API
For the full reference and examples, see `API_DOCUMENTATION.md`. You can use curl commands to run quick tests. 

Single query:
```bash
curl -X POST http://compactds.duckdns.org:30888/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me more about Albert Einstein", "n_docs": 5, "nprobe": 32}'
```

Batched queries:
```bash
curl -X POST http://compactds.duckdns.org:30888/search \
  -H "Content-Type: application/json" \
  -d '{"queries": ["quantum computing", "Who is Nikola Tesla", "AI ethics"], "n_docs": 2}'
```


##  DiskANN build
## DiskANN serving (setup and launch)

### 1) Create environment

```bash
conda create -n ds-serve python=3.10 -y
conda activate ds-serve
python -m pip install -U pip setuptools wheel
# Install dependencies
pip install -r requirements.txt
# Install DiskANN
pip install --no-deps diskannpy==0.7.0
```

### 2) Launch the server (DiskANN)

From the repo root (absolute paths):
```bash
mkdir -p /mnt/md-256k/jinjian/DS/runtime/votes /mnt/md-256k/jinjian/DS/runtime/query_logs

PYTHONPATH="/mnt/md-256k/jinjian/DS:/mnt/md-256k/jinjian/DS/rerank/contriever/src" \
VOTES_DIR=/mnt/md-256k/jinjian/DS/runtime/votes \
QUERY_LOG_DIR=/mnt/md-256k/jinjian/DS/runtime/query_logs \
MASSIVE_SERVE_PORT=30888 \
MS_BACKEND=diskann \
DATASTORE_PATH=/mnt/md-256k/jinjian/DS \
DISKANN_INDEX_DIR=/mnt/md-256k/jinjian/DS/DiskANN-build/DiskANN_index \
DISKANN_INDEX_PREFIX=diskann_mips_f32_R60_L80_B200_M500 \
DISKANN_DISTANCE=mips \
DISKANN_NUM_THREADS=128 \
DISKANN_NODES_TO_CACHE=50000 \
DISKANN_L=150 \
DISKANN_W=4 \
DISKANN_WARMUP=1 \
DISKANN_WARMUP_QUERIES=5000 \
DISKANN_WARMUP_BATCH=256 \
DISKANN_WARMUP_QUERY_FILE=/mnt/md-256k/jinjian/DS/DiskANN-build/DiskANN_index/diskann_mips_f32_R60_L80_B200_M500_sample_data.bin \
DISKANN_WARMUP_KEEPALIVE=1 \
python -m massive_serve.cli serve --domain_name data
```

Required files (absolute paths pasted below for reference):
- /mnt/md-256k/jinjian/DS/position_array.npy
- /mnt/md-256k/jinjian/DS/filename_index_array.npy
- /mnt/md-256k/jinjian/DS/filename_list.npy
- /mnt/md-256k/jinjian/DS/data/passages/
- /mnt/md-256k/jinjian/DS/DiskANN-build/DiskANN_index/

Tips:
- Use a different `MASSIVE_SERVE_PORT` if firewall issues occur. 
- `DISKANN_NUM_THREADS` sets CPU threads for DiskANN search; 0 uses all logical CPUs.
- `DISKANN_NODES_TO_CACHE` pins popular nodes in RAM; warmup further primes OS page cache.

 
