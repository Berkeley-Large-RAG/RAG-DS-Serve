---
layout: page
title: Voting System
---

<style>
p { font-size: 18px; margin: 6px 0; }
code, pre { background: #f6f8fa; border: 1px solid #eaecef; border-radius: 6px; }
pre { padding: 12px 14px; overflow: auto; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid #eaecef; padding: 8px 10px; text-align: left; }
th { background: #fafbfc; }
</style>

## Votes: what we store and where

We capture every Yes/No vote in two places:
1. `votes.jsonl` — the human-readable log for votes.
2. `votes.sqlite3` — a tiny index so `/vote/peek` can answer quickly without scanning the log.

By default we write into `./logging/votes/` beneath the current working directory; set `DS_SERVE_LOG_DIR` before launching the server to place the folder elsewhere.

### JSONL record (append-only)
Each line is one vote, e.g.:
```json
{
  "time stamp": "2025-12-05 21:04:33 PDT",
  "query": "Explain how to make coffee",
  "passage_id": "749481",
  "vote": "yes",
  "relevant": true,
  "backend": "diskann",
  "parameters": {
    "nprobe": 32,
    "exact_search": false,
    "diverse_search": false,
    "lambda": 0,
    "k": 10,
    "min_words": 10,
    "diskann_L": 500,
    "diskann_W": 8,
    "diskann_threads": 64
  }
}
```
- `time stamp` is always in Californian Time/PDT for easy reading.
- `parameters` contains whichever knobs were active (IVFPQ or DiskANN). Fields are omitted when not applicable.

### SQLite sidecar (fast lookup)
We mirror the latest vote for `(query, parameter set, passage_id)` into `votes.sqlite3` so the API can answer “what is the last vote?” instantly. Schema:
```sql
CREATE TABLE IF NOT EXISTS queries (
  query_hash TEXT PRIMARY KEY,
  query_norm TEXT
);

CREATE TABLE IF NOT EXISTS contexts (
  ctx_hash TEXT PRIMARY KEY,
  nprobe INTEGER,
  exact_search INTEGER,
  diverse_search INTEGER,
  lambda REAL,
  k INTEGER,
  min_words INTEGER,
  diskann_L INTEGER,
  diskann_W INTEGER,
  diskann_threads INTEGER
);

CREATE TABLE IF NOT EXISTS votes (
  query_hash TEXT NOT NULL,
  ctx_hash TEXT NOT NULL,
  passage_id TEXT NOT NULL,
  relevant INTEGER NOT NULL,
  ts INTEGER NOT NULL,
  PRIMARY KEY (query_hash, ctx_hash, passage_id)
);
```
- `query_hash = sha1(normalized_query)` (lowercase + collapsed whitespace).
- `ctx_hash = sha1(canonicalized parameter JSON)` so identical knob settings map to the same row.
- `INSERT OR REPLACE` keeps only the latest vote for a passage/config pair.

### How voting works in the UI
1. User clicks **[YES]** or **[NO]** under a passage.
2. The frontend sends `/vote` a JSON body with the raw query, `passage_id`, boolean `relevant`, plus the active backend parameters (nprobe, min words, DiskANN L/W/threads, etc.).
3. The server appends the JSONL line, updates SQLite, and returns `{ "status": "ok" }`.

Example request:
```bash
curl -X POST http://localhost:30888/vote \
  -H 'Content-Type: application/json' \
  -d '{
        "query": "Explain how to make coffee",
        "passage_id": "749481",
        "relevant": true,
        "backend": "diskann",
        "config": {"k": 10, "min_words": 10, "diskann_L": 500, "diskann_W": 8, "diskann_threads": 64}
      }'
```

### Looking up votes
To see the latest relevance for a passage:
```sql
-- Assume you already know the normalized query hash and ctx hash.
SELECT relevant, ts
FROM votes
WHERE query_hash = :query_hash AND ctx_hash = :ctx_hash AND passage_id = :passage_id;
```

To rebuild reports, parse `votes.jsonl` sequentially (newest last) – it’s the source of truth.

### Maintenance tips
- Rotate `votes.jsonl` whenever needed (e.g., monthly) by moving it to `votes-YYYYMM.jsonl.zst` and letting the server recreate a fresh file.
- Occasionally run `sqlite3 logging/votes/votes.sqlite3 "PRAGMA wal_checkpoint(FULL); VACUUM;"` to compact the SQLite DB.
- Backups: `sqlite3 logging/votes/votes.sqlite3 ".backup 'votes-$(date +%F).sqlite3'"`.

### Posting a vote
```bash
curl -X POST http://localhost:30888/vote \
  -H 'Content-Type: application/json' \
  -d '{
        "query": "Your query here",
        "passage_id": "abc123",
        "relevant": true,
        "config": {"nprobe": 32, "exact_search": false, "diverse_search": true, "lambda": 0.5}
      }'
```

### Looking up votes (SQLite)
- All votes for a specific normalized query and config:
```sql
-- First compute sha1(normalized_query) and ctx_hash (sha1 of canonical config JSON)
SELECT v.passage_id, v.relevant, v.ts
FROM votes v
WHERE v.query_hash = :query_hash AND v.ctx_hash = :ctx_hash;
```

- Inspect configs and queries for readability:
```sql
SELECT q.query_norm, c.nprobe, c.exact_search, c.diverse_search, c.lambda
FROM votes v
JOIN queries q ON q.query_hash = v.query_hash
JOIN contexts c ON c.ctx_hash = v.ctx_hash
WHERE v.query_hash = :query_hash
LIMIT 50;
```

### Normalization and hashing
- Normalization: lowercase, trim, collapse internal whitespace.
- `query_hash = sha1(normalized_query)` deduplicates trivial query variants.
- `ctx_hash = sha1(canonical_config_json)` where the JSON includes only `nprobe`, `exact_search`, `diverse_search`, and optionally `lambda` when `diverse_search` is true, with sorted keys.

### Maintenance
- Checkpoint and compact the SQLite index:
```bash
sqlite3 /home/ubuntu/votes/votes.sqlite3 "PRAGMA wal_checkpoint(FULL); VACUUM;"
```

- Backup the SQLite index safely:
```bash
sqlite3 /home/ubuntu/votes/votes.sqlite3 \
  ".backup '/home/ubuntu/votes/votes-backup-$(date +%F).sqlite3'"
```

- Archiving JSONL (optional rotation):
  - Periodically move old `votes.jsonl` to `votes-YYYYMM.jsonl.zst` and start a new file.
  - The SQLite index remains current (latest votes only per key). If rebuilding the index from JSONL, replay newest-first per `(query_hash, ctx_hash, passage_id)`.




