---
layout: page
title: API Documentation
---

<style>
/* keep pages visually consistent with index.md */
p { font-size: 18px; margin: 6px 0; }
code, pre { background: #f6f8fa; border: 1px solid #eaecef; border-radius: 6px; }
pre { padding: 12px 14px; overflow: auto; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid #eaecef; padding: 8px 10px; text-align: left; }
th { background: #fafbfc; }
</style>

## CompactDS API

Use this endpoint to run searches from your code. It mirrors the controls in the UI.

Endpoint:
```
POST http://compactds.duckdns.org:30888/search
Content-Type: application/json
```

### Request body
- Provide either `query` (single) or `queries` (list). Include any of the optional knobs.

| field | type | default | notes |
|---|---|---|---|
| query | string | — | single query string |
| queries | string[] | — | batch mode |
| n_docs | int | 1 | number of passages to return |
| nprobe | int | 32 | 1–256; higher improves recall |
| exact_search | bool | false | rerank with exact similarities |
| diverse_search | bool | false | enable diversity (MMR) |
| lambda | float | 0.5 | only used when `diverse_search=true` |

### Examples

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

### 2) High-accuracy

```python
payload = {
    "query": "neural network architecture",
    "n_docs": 3,
    "nprobe": 256,
    "exact_search": True
}
```

### 3) Diverse

```python
payload = {
    "query": "artificial intelligence applications",
    "n_docs": 5,
    "diverse_search": True,
    "lambda": 0.2
}
```

### 4) Batched

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

## Response format
Single query response:

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

Batched response:

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

## Tips
For speed:
```python
payload = {
    "query": "your query",
    "nprobe": ...,           # nprobe has minimal impact on delay
    "exact_search": False,  # Disable exact search
    "diverse_search": False # Disable diverse search
}
```

For accuracy:
```python
payload = {
    "query": "your query",
    "nprobe": 256,          # Higher nprobe
    "exact_search": True,   # Enable exact search
    "diverse_search": False # Keep diverse search off for pure accuracy
}
```

For diversity:
```python
payload = {
    "query": "your query",
    "nprobe": 64,           # Moderate nprobe
    "exact_search": False,  # Optional: can be enabled
    "diverse_search": True, # Enable diverse search
    "lambda": 0.5        # High lambda for more diversity
}
```

Balanced:
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

## Quick test script

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