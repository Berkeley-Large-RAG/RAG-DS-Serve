---
layout: page
title: API documentation
permalink: /API_DOCUMENTATION.html
---

# API Documentation

## 🚀 Overview

**DS Serve** is a single-node RAG server backed by CompactDS (2B passages). It exposes both a web UI and a JSON API. Two ANN engines are available:

**🌐 Live Demo**: [http://api.ds-serve.org:30888/ui](http://api.ds-serve.org:30888/ui)

### Key Features
- Dual ANN backends (DiskANN + IVFPQ), can be chosen by user.
- Configurable search knobs (n_docs, L/W, nprobe, diversity).
- Query history + vote logging for relevance feedback.
- Fully scriptable REST API.

### Example Queries
These are some good example queries that yield quality results on moderate setting:
- "Tell me more about Albert Einstein"
- "Tell me more about Nikola Tesla"
- "Explain the basics of quantum physics"
- "Who is Matei Zaharia at UC Berkeley and founder of Apache Stark" 

---

## 📡 Endpoint

```
POST http://api.ds-serve.org:30888/search
Content-Type: application/json
```

---

## 🔧 Parameters

### Required

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>query</code></td>
      <td>string</td>
      <td>Single query string</td>
    </tr>
    <tr>
      <td><code>queries</code></td>
      <td>array</td>
      <td>Batch of queries (use instead of <code>query</code>)</td>
    </tr>
  </tbody>
</table>

### Optional

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>backend</code></td>
      <td>string</td>
      <td>"diskann"</td>
      <td>"diskann" or "ivfpq"</td>
    </tr>
    <tr>
      <td><code>n_docs</code></td>
      <td>integer</td>
      <td>1</td>
      <td>Top‑k passages to return (1‑1000)</td>
    </tr>
    <tr>
      <td><code>nprobe</code></td>
      <td>integer</td>
      <td>32</td>
      <td>IVFPQ clusters to scan (ignored for DiskANN)</td>
    </tr>
    <tr>
      <td><code>exact_search</code></td>
      <td>boolean</td>
      <td>false</td>
      <td>Brute-force rerank after ANN</td>
    </tr>
    <tr>
      <td><code>diverse_search</code></td>
      <td>boolean</td>
      <td>false</td>
      <td>Penalize near-duplicate passages</td>
    </tr>
    <tr>
      <td><code>lambda</code></td>
      <td>float</td>
      <td>0.5</td>
      <td>Diversity tradeoff used with <code>diverse_search</code></td>
    </tr>
    <tr>
      <td><code>diskann_L</code></td>
      <td>integer</td>
      <td>300</td>
      <td>DiskANN candidate list size (≥ <code>n_docs</code>)</td>
    </tr>
    <tr>
      <td><code>diskann_W</code></td>
      <td>integer</td>
      <td>4</td>
      <td>DiskANN beam width / I/O fan-out</td>
    </tr>
    <tr>
      <td><code>diskann_threads</code></td>
      <td>integer</td>
      <td>server default</td>
      <td>Override worker thread count</td>
    </tr>
    <tr>
      <td><code>min_words</code></td>
      <td>integer</td>
      <td>0</td>
      <td>Minimum passage length filter</td>
    </tr>
  </tbody>
</table>

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


