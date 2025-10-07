## Votes Storage and Usage Guide

This document explains where votes are stored, what each file is for, the on-disk schemas, and how to query or maintain them.

### Location
- Directory: by default `/home/ubuntu/votes` (override with env `VOTES_DIR`)
- Files:
  - `votes.jsonl`: append-only audit log (one JSON per line)
  - `votes.sqlite3`: fast-lookup index (with `votes.sqlite3-wal` and `votes.sqlite3-shm` managed by SQLite WAL mode)

### What gets stored per vote
- We record the user’s binary relevance vote for a passage, tied to a specific query and query config.
- Query text is normalized (lowercased, trimmed, whitespace-collapsed) and hashed to avoid key fragmentation and save space.

#### JSONL record schema (one per line)
```json
{
  "ts": 1732132123,                // integer seconds since epoch
  "query_hash": "<sha1(norm_query)>",
  "query_norm": "normalized query text",
  "passage_id": "...",           // original passage id
  "relevant": true,               // true/false
  "config": {                     // compact config (optional)
    "nprobe": 32,                 // integer if provided
    "exact_search": false,        // boolean
    "diverse_search": true,       // boolean
    "lambda": 0.5                 // present only if diverse_search==true; rounded to 2 decimals
  }
}
```

Notes:
- `query_hash = sha1(query_norm)`. We store `query_norm` in JSONL for readability and for reconstructing the original normalized text.
- `config` is omitted if empty; `lambda` is omitted when `diverse_search` is false.

#### SQLite index schema (for efficient lookup)
- Database file: `votes.sqlite3`
- Tables:
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
  lambda REAL
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

Key idea: (query + config) → passage_id → latest relevance.
- `ctx_hash` is `sha1(canonical_config_json)`; canonicalization includes only the compact fields above.
- Latest vote wins via `INSERT OR REPLACE` on the `(query_hash, ctx_hash, passage_id)` primary key.

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

### Looking up votes (SQLite examples)
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

### Changing storage location
- Set `VOTES_DIR=/path/to/dir` before starting the server to change where votes are written.

### Privacy/PII note
- Only normalized query text and the hash are stored (no raw query in JSONL). Ensure this matches your privacy requirements.


