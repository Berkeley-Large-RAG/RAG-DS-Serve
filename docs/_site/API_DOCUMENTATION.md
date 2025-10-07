# Compact-DS Dive Documentation

## 🚀 Overview

**Compact-DS Dive** is a easy-to-use RAG UI backed by a billion-scale high-quality datastore. Compact-DS Dive combines approximate nearest neighbor (ANN) search with Exact and Diverse Search capabilities. This demo showcases flexible search techniques: configurable search parameters, different search options, history caching, and context expansion.

**🌐 Live Demo**: [https://tinyurl.com/compact-ds-dive](https://tinyurl.com/compact-ds-dive)

### Key Features
- **Fast ANN Search**: Efficient approximate search using IVF-PQ indexing
- **Exact Reranking**: Optional exact similarity computation for higher accuracy. Note: we integrated retrieval results caching, so similar queries will have lower delay only on the second Exact Search and afterwards.
- **Diverse Results**: Configurable diversity to avoid similar passages.
- **Batched Queries**: Support for processing multiple queries simultaneously
- **Real-time Performance**: Optimized for low-latency search operations

---

## 📡 API Endpoint

```
POST http://compactds.duckdns.org:30888/search
Content-Type: application/json
```

---

## 🔧 Parameters

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `query` | string | Your search query (for single queries) | `"machine learning algorithms"` |
| `queries` | array | Array of search queries (for batched queries) | `["query1", "query2", "query3"]` |

**Note**: Use either `query` (single) or `queries` (batched), not both.

### Optional Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_docs` | integer | 1 | 1-1000 | Number of top passages to retrieve |
| `nprobe` | integer | 32 | 1, 8, 16, 32, 64, 128, 256 | Number of clusters to search (higher = more accurate but slower) |
| `exact_search` | boolean | false | true/false | Use exact search instead of ANN (more accurate but slower) |
| `diverse_search` | boolean | false | true/false | Use diverse search to avoid similar results |
| `lambda` | float | 0.5 | 0.0-1.0 | Diversity tradeoff parameter (higher = more diverse, lower = more relevant) |


---

## 🎯 Parameter Details

### `nprobe` - Search Clusters
Controls the number of clusters searched during ANN retrieval:

### `exact_search` - Exact Reranking
When enabled, performs exact similarity computation instead of approximate search:
- **Benefits**: Higher accuracy
- **Trade-offs**: Slower response time by about 1.5s, higher computational cost. Delay will be reduced on similar search thanks to caching.

### `diverse_search` - Diverse Results
Enables diversity-aware result selection to avoid similar passages:
- **Benefits**: More varied results, better coverage
- **Trade-offs**: May reduce accuracy for some queries and also higher end-to-end delay.
- **Best with**: `lambda` parameter to control diversity vs accuracy trade-off

### `lambda` - Diversity Trade-off
Controls the balance between accuracy and diversity when `diverse_search` is enabled:
- **0.0**: Maximum accuracy, no diversity
- **0.2-0.3**: Balanced approach -- because of the harshness of the algorithm, we recommend setting lambda to a low value when diverse search is on.
- Anything higher can be risky, and please be aware to increase only for experimental purposes
---

## 📝 Request Examples

### 1. Basic Single Query

```python
import requests
import json

url = "http://compactds.duckdns.org:30888/search"
headers = {"Content-Type": "application/json"}

payload = {
    "query": "Tell me more about Albert Einstein",
    "n_docs": 5,
    "nprobe": 32
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()
```

### 2. High-Accuracy Search

```python
payload = {
    "query": "neural network architecture",
    "n_docs": 3,
    "nprobe": 256,
    "exact_search": True
}
```

### 3. Diverse Search

```python
payload = {
    "query": "artificial intelligence applications",
    "n_docs": 5,
    "diverse_search": True,
    "lambda": 0.2
}
```

### 4. Batched Queries

```python
payload = {
    "queries": [
        "quantum computing",
        "blockchain technology",
        "computer vision"
    ],
    "n_docs": 3,
    "exact_search": True,
    "diverse_search": True,
    "lambda": 0.2
}
```


---

## 📊 Response Format

### Single Query Response

```json
{
  "message": "Search completed for 'machine learning' from demo",
  "query": "machine learning",
  "n_docs": 5,
  "nprobe": 32,
  "results": {
    "scores": [[0.85, 0.82, 0.79, 0.76, 0.73]],
    "passages": [[
      {
        "text": "Machine learning is a subset of artificial intelligence...",
        "source": "c4_dclm_mixed",
        "index_id": 123456789,
        "passage_id": "passage_123"
      },
      "... more passages"
    ]]
  }
}
```

**Note**: Similarity scores are displayed in the UI for each passage, showing the relevance score where higher values indicate better matches.

**Score Types by Search Mode:**
- **ANN Search (default)**: FAISS index similarity scores
- **Exact Search**: Cosine similarity scores computed during exact reranking
- **Diverse Search**: Uses exact similarity scores with diversity penalty applied during selection

### Batched Query Response

```json
{
  "message": "Search completed for batched queries from demo",
  "query": ["query1", "query2", "query3"],
  "n_docs": 3,
  "nprobe": 32,
  "results": {
    "scores": [
      [0.85, 0.82, 0.79],  
      [0.88, 0.85, 0.81],  
      [0.83, 0.80, 0.77]   
    ],
    "passages": [
      [...],
      [...],
      [...]
    ]
  }
}
```

---

## 🎛️ Parameter Optimization Guide

### For Speed
```python
payload = {
    "query": "your query",
    "nprobe": ...,           # nprobe has minimal impact on delay
    "exact_search": False,  # Disable exact search
    "diverse_search": False # Disable diverse search
}
```

### For Accuracy
```python
payload = {
    "query": "your query",
    "nprobe": 256,          # Higher nprobe
    "exact_search": True,   # Enable exact search
    "diverse_search": False # Keep diverse search off for pure accuracy
}
```

### For Diversity
```python
payload = {
    "query": "your query",
    "nprobe": 64,           # Moderate nprobe
    "exact_search": False,  # Optional: can be enabled
    "diverse_search": True, # Enable diverse search
    "lambda": 0.5        # High lambda for more diversity
}
```

### For Balanced Performance
```python
payload = {
    "query": "your query",
    "nprobe": 32,           # Default nprobe
    "exact_search": False,  # Disable for speed
    "diverse_search": True, # Enable for variety
    "lambda": 0.25          # Balanced lambda
}
```

---

## 🧪 Testing Script

Here's a complete testing script you can use:

```python
#!/usr/bin/env python3
import requests
import json
import time

def test_api():
    url = "http://compactds.duckdns.org:30888/search"
    headers = {"Content-Type": "application/json"}
    
    # Test cases
    test_cases = [
        {
            "name": "Basic Search",
            "payload": {
                "query": "machine learning",
                "n_docs": 3
            }
        },
        {
            "name": "Exact Search",
            "payload": {
                "query": "neural networks",
                "n_docs": 3,
                "exact_search": True
            }
        },
        {
            "name": "Diverse Search",
            "payload": {
                "query": "artificial intelligence",
                "n_docs": 5,
                "diverse_search": True,
                "lambda": 0.3
            }
        },
        {
            "name": "Batched Queries",
            "payload": {
                "queries": ["quantum computing", "Who is Nikola Tesla", "AI ethics"],
                "n_docs": 2
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n🧪 Testing: {test['name']}")
        print(f"Payload: {json.dumps(test['payload'], indent=2)}")
        
        start = time.time()
        response = requests.post(url, headers=headers, json=test['payload'])
        latency = time.time() - start
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            passages = data.get('results', {}).get('passages', [])
            if isinstance(passages[0], list):  # Batched
                total_results = sum(len(p) for p in passages)
                print(f"✅ Success! {len(passages)} queries, {total_results} total results in {latency:.2f}s")
            else:  # Single
                print(f"✅ Success! {len(passages[0])} results in {latency:.2f}s")
        else:
            print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    test_api()
```
---

## 💻 Compute Resources & Performance

### System Specifications
- **CPU**: High-performance multi-core processors optimized for vector operations
- **RAM**: Sufficient memory for billion-scale index operations
- **Storage**: Fast SSD storage for quick data access
- **Network**: Low-latency connections for real-time search

### Performance Considerations
- **n_docs Impact**: Higher values (50-1000) increase response time linearly but provide more comprehensive results
- **Exact Search**: Adds ~1.5s latency but improves accuracy significantly
- **Diverse Search**: Moderate performance impact with configurable diversity trade-offs
- **Caching**: Similar queries benefit from result caching for faster subsequent responses

### Recommended Usage
- **Small Queries**: n_docs 1-20 for quick results
- **Comprehensive Search**: n_docs 50-500 for thorough exploration
- **Large-scale Analysis**: n_docs 1000 for maximum coverage (expect longer response times)

---

*This documentation covers the Compact-DS Search API v1.0. For updates and additional features, please refer to the latest version.* 