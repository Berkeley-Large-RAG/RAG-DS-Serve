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
We mirror the latest vote for each `(query, parameter set, passage_id)` into `votes.sqlite3` so the API can answer “what is the last vote?” instantly. Schema:
```sql
CREATE TABLE IF NOT EXISTS votes (
  time_stamp TEXT NOT NULL,
  query TEXT NOT NULL,
  passage_id TEXT NOT NULL,
  vote TEXT NOT NULL,
  relevant INTEGER NOT NULL,
  backend TEXT,
  parameters TEXT NOT NULL, -- canonical JSON string of the active knobs
  PRIMARY KEY (query, parameters, passage_id)
);
```
- `parameters` is the canonical JSON string (sorted keys, no whitespace) built from the same dictionary that is logged in JSONL. We use it to deduplicate votes for the same knob settings.
- `INSERT OR REPLACE` keeps only the latest vote for a `(query, parameters, passage_id)` triple, so lookups are O(1).

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
To see the latest relevance for a passage + parameter set:
```sql
SELECT vote, relevant, time_stamp
FROM votes
WHERE query = :query
  AND parameters = :canonical_parameters_json
  AND passage_id = :passage_id;
```

To rebuild reports, parse `votes.jsonl` sequentially (newest last) – it’s the source of truth.

### Maintenance tips
- Rotate `votes.jsonl` whenever needed (e.g., monthly) by moving it to `votes-YYYYMM.jsonl.zst` and letting the server recreate a fresh file.
- Occasionally run `sqlite3 logging/votes/votes.sqlite3 "PRAGMA wal_checkpoint(FULL); VACUUM;"` to compact the SQLite DB.
- Backups: `sqlite3 logging/votes/votes.sqlite3 ".backup 'votes-$(date +%F).sqlite3'"`.

