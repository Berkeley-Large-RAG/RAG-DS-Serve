# API Documentationhttps://berkeley-large-rag.github.io/RAG-DS-Serve/

## 🚀 Overview

**DS Serve** is a single-node RAG server backed by CompactDS (2B passages). It exposes both a web UI and a JSON API. Two ANN engines are available:

**🌐 Live Demo**: [http://api.ds-serve.org:30888/ui](http://api.ds-serve.org:30888/ui)

### Key Features
- Dual ANN backends (DiskANN + IVFPQ), can be chosen by user.
- Configurable search knobs (n_docs, L/W, nprobe, diversity).
- Query history + vote logging for relevance feedback.
- Fully scriptable REST API.

---

## 📡 Endpoint

```
POST http://api.ds-serve.org:30888/search
Content-Type: application/json
```

---

## 🔧 Parameters

### Required
| Parameter | Type | Description |
|-----------|------|-------------|
| `query`   | string | Single query string |
| `queries` | array  | Batch of queries (use instead of `query`) |

### Optional
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | string | `"diskann"` | `"diskann"` or `"ivfpq"` |
| `n_docs`  | integer | 1 | Top‑k passages to return (1‑1000) |
| `nprobe`  | integer | 32 | IVFPQ clusters to scan (ignored for DiskANN) |
| `exact_search` | boolean | false | Brute-force rerank after ANN |
| `diverse_search` | boolean | false | Penalize near-duplicate passages |
| `lambda` | float | 0.5 | Diversity tradeoff used with `diverse_search` |
| `diskann_L` | integer | 300 | DiskANN candidate list size (≥ `n_docs`) |
| `diskann_W` | integer | 4 | DiskANN beam width / I/O fan-out |
| `diskann_threads` | integer | server default | Override worker thread count |
| `min_words` | integer | 0 | Minimum passage length filter |

### DiskANN Knobs
- **`diskann_L`**: Increase for higher recall; larger values add latency.
- **`diskann_W`**: Beam width; higher W reduces iterations but increases per-step work.
- **`diskann_threads`**: Explicit CPU threads used by DiskANN (`0`/omit = server default).
- **`min_words`**: Drop oversampled DiskANN hits shorter than the specified length.

---

## 📝 Request Examples (curl)

A few easy single request examples that you can test with curl. 

### 1. Default (DiskANN)
```bash
curl -X POST http://api.ds-serve.org:30888/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me more about Nikola Tesla",
    "n_docs": 10
  }'
```

### 2. DiskANN (can also be set explicitly)
```bash
curl -X POST http://api.ds-serve.org:30888/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me more about Nikola Tesla",
    "backend": "diskann",
    "n_docs": 10,
    "diskann_W": 8,
    "diskann_L": 500
  }'
```

### 3. IVFPQ
```bash
curl -X POST http://api.ds-serve.org:30888/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me more about Nikola Tesla",
    "backend": "ivfpq",
    "n_docs": 10,
    "nprobe": 256
  }'
```


