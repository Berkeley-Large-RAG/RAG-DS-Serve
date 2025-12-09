## Votes Storage and Usage Guide

This document explains where votes are stored, what each file is for, the on-disk schemas, and how to query or maintain them.

### Location
- Directory: `./logging/votes/` relative to the working directory (set `DS_SERVE_LOG_DIR` before launching the server to override).
- Files:
  - `votes.jsonl`: append-only audit log (one JSON per line, human-readable).
  - `votes.sqlite3`: fast-lookup index (plus the WAL sidecar files SQLite creates automatically).

### What gets stored per vote
- We persist the user’s Yes/No vote together with the query text, backend, and the knobs that were active.

#### JSONL record schema (one per line)
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

#### SQLite index schema (for efficient lookup)
```sql
CREATE TABLE IF NOT EXISTS votes (
  time_stamp TEXT NOT NULL,
  query TEXT NOT NULL,
  passage_id TEXT NOT NULL,
  vote TEXT NOT NULL,
  relevant INTEGER NOT NULL,
  backend TEXT,
  parameters TEXT NOT NULL,
  PRIMARY KEY (query, parameters, passage_id)
);
```
- `parameters` stores the canonical JSON string (sorted keys, no whitespace). Using that deterministic string keeps the latest vote per `(query, parameters, passage_id)`.

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
```sql
SELECT vote, relevant, time_stamp
FROM votes
WHERE query = :query
  AND parameters = :canonical_parameters_json
  AND passage_id = :passage_id;
```

`canonical_parameters_json` must be the sorted, no-whitespace JSON string that the server stores (e.g. `{"backend":"diskann","diskann_L":500,...}`).

### Maintenance
- Checkpoint and compact the SQLite index:
```bash
sqlite3 logging/votes/votes.sqlite3 "PRAGMA wal_checkpoint(FULL); VACUUM;"
```

- Backup the SQLite index safely:
```bash
sqlite3 logging/votes/votes.sqlite3 \
  ".backup 'logging/votes/votes-backup-$(date +%F).sqlite3'"
```

- Archiving JSONL (optional rotation):
  - Periodically move old `votes.jsonl` to `votes-YYYYMM.jsonl.zst` and start a new file.
  - The SQLite index remains current (latest votes only per key). If rebuilding the index from JSONL, replay the file from oldest to newest so the most recent vote wins.

### Changing storage location
- Set `VOTES_DIR=/path/to/dir` before starting the server to change where votes are written.

### Privacy/PII note
- The raw query text you submit is stored in both `votes.jsonl` and `votes.sqlite3`. Make sure that aligns with your data-handling policies.


