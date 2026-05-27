# DS-Serve — Reproducible Setup Tutorial (Path A: FAISS IVFPQ)

This document captures **exactly** how the CompactDS-102GB retrieval server (`massive_serve`)
was brought up on one node, so it can be reproduced on another. It deliberately differs from
the repo README in several places where the README does not match the actual released dataset.

> **What you get:** a Flask HTTP service exposing `/search` over a ~1.5B-passage FAISS IVFPQ
> index, subsecond latency, ~116 GB RAM resident.
> **What this is NOT:** an index *builder*. This serves a pre-built index; it does not embed or
> train one.

---

## 0. Hardware / OS prerequisites

| Resource | Minimum | Notes |
|---|---|---|
| Disk (free) | **~2.5 TB** | Dataset is ~2.5 TB total (124 GB index + ~2.37 TB passages). The "102 GB" name refers only to the ANN index. |
| RAM | **~150 GB** | `faiss.read_index` loads the 108 GB index fully into RAM (not mmap), plus a direct-map and the encoder. |
| CPU | any multi-core | Query encoding runs on CPU fine. |
| GPU | **not required** | Only needed for the optional "Exact Search" rerank (GritLM). |
| Network | HF access | To download the dataset and the `facebook/contriever-msmarco` encoder. |

Software: Python 3.12, `git`, and either `uv` (used here) or `pip`/`venv`.

Pick a data root with space and a fixed home for the repo. This doc uses placeholders:
```bash
export REPO=/path/to/RAG-DS-Serve            # the cloned repo
export DATASTORE_PATH=/path/with/space/massive-serve   # data root
export DOMAIN=index_dev                        # domain/folder name (arbitrary but must be consistent)
export PORT=30888
```

---

## 1. Clone repo + create environment

```bash
git clone https://github.com/Berkeley-Large-RAG/RAG-DS-Serve.git "$REPO"
cd "$REPO"
git submodule update --init --recursive

uv venv .venv && . .venv/bin/activate
uv pip install -U pip setuptools wheel
uv pip install -r requirements.txt
uv pip install -e .
# sanity
python -c "import faiss, torch, sentence_transformers; print('ok', faiss.__version__)"
```

> **Patch #1 (REQUIRED — transformers 5.x compatibility).**
> `requirements.txt` is unpinned, so you'll likely get `transformers>=5.x`, which **removed**
> `tokenizer.batch_encode_plus`. The server will crash at startup with
> `AttributeError: BertTokenizer has no attribute batch_encode_plus`.
> Fix in `massive_serve/src/search.py` (~line 49): change
> `encoded_batch = tokenizer.batch_encode_plus(` → `encoded_batch = tokenizer(`
> (`__call__` is the modern, argument-compatible replacement.)
>
> Alternatively pin an older transformers (e.g. `transformers<5`) instead of patching.

---

## 2. Download the dataset

```bash
mkdir -p "$DATASTORE_PATH"
huggingface-cli download alrope/CompactDS-102GB \
  --repo-type dataset \
  --local-dir "$DATASTORE_PATH/$DOMAIN"
```

This is large (~2.5 TB) and slow. After it finishes, the layout is:
```
$DATASTORE_PATH/$DOMAIN/
  README.md
  embeddings/index_IVFPQ/
    index_IVFPQ.100000000.768.65536.64.faiss_aa ... _dz   # 104 byte-split shards
    index_IVFPQ.100000000.768.65536.64.faiss.meta          # 12 GB, NOT needed for serving
    passage_filenames.npy        # 959 entries  -> becomes filename_list.npy
    passage_pos_id_array.npy     # ~1.5B int64  -> becomes position_array.npy
  passages/
    *.jsonl                      # 959 passage shards (~2.37 TB)
```

> **Note:** there is **no `index/` directory and no `config.json`** in the download. The repo's
> code expects both. We create them in steps 3–4.

---

## 3. Assemble the index (`index/` with exactly one `.faiss`)

`massive_serve/src/indicies/base.py` requires **exactly one** `*.faiss` file under
`$DATASTORE_PATH/$DOMAIN/index/`. Concatenate the byte-split shards (the `_aa.._dz` suffixes are
a plain `split`; `cat` in sorted order reconstructs the file). **Exclude** the `.meta` file — the
glob below already does.

```bash
SHARDS="$DATASTORE_PATH/$DOMAIN/embeddings/index_IVFPQ"
mkdir -p "$DATASTORE_PATH/$DOMAIN/index"

# integrity: expected size = sum of shard sizes
EXPECT=$(du -cb "$SHARDS"/index_IVFPQ.*.faiss_* | tail -1 | cut -f1)
cat $(ls "$SHARDS"/index_IVFPQ.*.faiss_* | sort) \
  > "$DATASTORE_PATH/$DOMAIN/index/index_IVFPQ.100000000.768.65536.64.faiss"
GOT=$(stat -c%s "$DATASTORE_PATH/$DOMAIN/index/index_IVFPQ.100000000.768.65536.64.faiss")
[ "$EXPECT" = "$GOT" ] && echo "COMBINE OK ($GOT bytes)" || echo "SIZE MISMATCH"
```

The filename encodes the params: dim **768**, ncentroids **65536**, subquantizers **64**.

(Optional, after the server boots OK, reclaim ~124 GB: `rm -rf "$SHARDS"`. Keep it as a backup
until you've confirmed the combined index loads.)

---

## 4. Write `config.json`

`serve.py` asserts `config["domain_name"] == --domain_name == folder name`, and reads these 7 keys.
The encoder is effectively hardcoded to Contriever; the value must contain `"contriever"` so the
query path uses mean-pooling (not CLS).

```bash
cat > "$DATASTORE_PATH/$DOMAIN/config.json" <<EOF
{
  "domain_name": "$DOMAIN",
  "query_encoder": "facebook/contriever-msmarco",
  "query_tokenizer": "facebook/contriever-msmarco",
  "index_type": "IVFPQ",
  "per_gpu_batch_size": 64,
  "question_maxlength": 512,
  "nprobe": 128
}
EOF
```

---

## 5. Build the passage-mapping arrays (the fast way)

The loader (`ivf_pq.py`) reads **three** arrays from the **`$DATASTORE_PATH` root** (NOT the
domain folder — `repo_root = Path(DATASTORE_PATH)`):
`position_array.npy`, `filename_index_array.npy`, `filename_list.npy`.

The repo's `utils/build_arr.py` would build these by scanning **all 2.37 TB** of passages (slow,
hours). **Skip it.** The dataset already ships:
- `passage_pos_id_array.npy` → this IS `position_array` (per-file byte offsets, ~1.5B int64).
- `passage_filenames.npy` → this IS `filename_list` (959 names).

Only `filename_index_array` must be derived. File boundaries are marked by `0` offsets (each file's
first line is at byte 0), so a cumulative file-id is trivial to compute:

```bash
. "$REPO/.venv/bin/activate"
DATASTORE_PATH="$DATASTORE_PATH" DOMAIN="$DOMAIN" python - <<'PY'
import os, numpy as np, shutil
root = os.environ["DATASTORE_PATH"]; dom = os.environ["DOMAIN"]
src  = os.path.join(root, dom, "embeddings", "index_IVFPQ")
# 1) position_array  &  2) filename_list  (copy precomputed, rename)
shutil.copyfile(os.path.join(src, "passage_pos_id_array.npy"), os.path.join(root, "position_array.npy"))
shutil.copyfile(os.path.join(src, "passage_filenames.npy"),    os.path.join(root, "filename_list.npy"))
# 3) filename_index_array  (derive from per-file zero offsets)
pos   = np.load(os.path.join(root, "position_array.npy"), mmap_mode="r")
zeros = np.where(pos[:] == 0)[0]            # one per file start
bounds = np.append(zeros, len(pos))
fia = np.empty(len(pos), dtype=np.int32)
for i in range(len(zeros)):
    fia[bounds[i]:bounds[i+1]] = i
np.save(os.path.join(root, "filename_index_array.npy"), fia)
# verify
nm = list(np.load(os.path.join(root, "filename_list.npy"), allow_pickle=True).tolist())
assert len(zeros) == len(nm), (len(zeros), len(nm))
assert len(pos) == len(fia) and int(fia.max()) < len(nm)
print(f"OK: {len(pos)} passages, {len(nm)} files; arrays written to {root}")
PY
```

> **Correctness check (worth doing on a new dataset version):** confirm `build_arr.sort_jsonl_files`
> ordering equals `passage_filenames.npy` ordering, and that a few sampled `(file, offset)` pairs
> seek to valid JSON passages. If the precomputed `passage_pos_id_array.npy` is ever absent, fall
> back to `utils/build_arr.py` (set its `INPUT_DIR` to the passages dir; outputs land in CWD — run
> it from `$DATASTORE_PATH`).

---

## 6. Launch

```bash
mkdir -p "$DATASTORE_PATH/logging"
cd "$REPO" && . .venv/bin/activate

PYTHONPATH="$REPO:$REPO/rerank/contriever/src" \
DATASTORE_PATH="$DATASTORE_PATH" \
DS_SERVE_LOG_DIR="$DATASTORE_PATH/logging" \
MASSIVE_SERVE_PORT="$PORT" \
PASSAGE_DIR="$DATASTORE_PATH/$DOMAIN/passages" \
nohup python -m massive_serve.cli serve --domain_name "$DOMAIN" \
  > "$DATASTORE_PATH/logging/server.log" 2>&1 &
echo "server PID $!"
```

Why each env var matters:
- **`PYTHONPATH` includes `rerank/contriever/src`** — *Patch #2 (REQUIRED).* `rerank/exact_rerank.py`
  and `rerank/diverse_rerank.py` do `from normalize_text import normalize` (a bare top-level import).
  Without this path the server crashes with `ModuleNotFoundError: No module named 'normalize_text'`.
- **`PASSAGE_DIR`** — `DatastoreAPI` constructs `DiskANNBackend()` *unconditionally* even for the
  FAISS path, and its `__init__` raises `FileNotFoundError` if it can't find a passage dir (it only
  looks at `$DATASTORE_PATH/data/passages` or `$DATASTORE_PATH/passages`). Pointing it at the real
  passages dir avoids the crash. (It does **not** load any DiskANN index, so no DiskANN data needed.)
- **`DS_SERVE_LOG_DIR`** — overrides a hardcoded `/mnt/data/jinjian/...` log path in `serve.py`.
- **`DATASTORE_PATH`** as env — otherwise the CLI *interactively prompts* for it and hangs in nohup.

Startup takes a few minutes (load 108 GB index + build direct map + a self-test search). Watch:
```bash
tail -f "$DATASTORE_PATH/logging/server.log"   # ready when the "MASSIVE SERVE SERVER" banner prints
```

---

## 7. Test

The `/search` endpoint **defaults `backend` to `diskann`** (not installed here), so you MUST pass
`"backend":"faiss"`.

```bash
curl -s -X POST "http://localhost:$PORT/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"What causes the seasons on Earth?","n_docs":5,"backend":"faiss","nprobe":128}' \
| python3 -c 'import sys,json;d=json.load(sys.stdin)["results"];print(d["timings_ms"]);[print("-",p["text"][:120]) for p in d["passages"][0]]'
```
Web UI: `http://<host>:$PORT/ui`.

Request knobs: `n_docs` (count), `nprobe` (recall vs latency; 64–256 typical for 65536 centroids),
`min_words` (drop short passages).

---

## 8. Managing / troubleshooting

- **Stop:** `pkill -f massive_serve.cli`
- **Logs:** `$DATASTORE_PATH/logging/server.log`; per-query/vote logs under `$DATASTORE_PATH/logging/`.
- **Shell hangs with no output (bonus gotcha on the original node):** if your `~/.bashrc` launches an
  interactive shell unconditionally (e.g. a bare `zsh`), every non-interactive shell (tooling, ssh
  commands) hangs. Guard it:
  ```bash
  if [[ $- == *i* ]] && [ -z "$ZSH_VERSION" ] && command -v zsh >/dev/null; then exec zsh; fi
  ```

## Appendix — the two required source/path deviations from upstream
1. `massive_serve/src/search.py`: `tokenizer.batch_encode_plus(...)` → `tokenizer(...)` (transformers 5.x).
2. Launch with `PYTHONPATH=$REPO/rerank/contriever/src` so `normalize_text` resolves.

Everything else is configuration (config.json, env vars, array placement), not code changes.
