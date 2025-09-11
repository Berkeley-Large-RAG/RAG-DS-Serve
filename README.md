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

### Download full precision
Look at [donwload file](DiskANN-build/download.py)
### Convert the embedding
Look at [convert file](DiskANN-build/convert.py)
## Citation
```
@article{lyu2025compactds,
  title={Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks},
  author={Xinxi Lyu and Michael Duan and Rulin Shao and Pang Wei Koh and Sewon Min}
  journal={arXiv preprint arXiv:2507.01297},
  year={2025}
}
```