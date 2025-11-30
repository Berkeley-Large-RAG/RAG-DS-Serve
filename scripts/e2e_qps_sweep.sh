#!/usr/bin/env bash
set -euo pipefail

# Configurable parameters via env or defaults
HOST=${HOST:-http://http://128.208.4.44:30888/}
QUERIES=${QUERIES:-/mnt/md-256k/jinjian/DS/e2e_queries_20000.txt}
COUNT=${COUNT:-10000}
CONCURRENCY=${CONCURRENCY:-128}
K=${K:-10}
W=${W:-4}
THREADS=${THREADS:-128}
L_LIST=${L_LIST:-"500 1000 1500 2000 2500 3000"}
REQUEST_MODE=${REQUEST_MODE:-batch}  # Accepted values: batch, single

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (apt-get install -y jq)." >&2
  exit 1
fi

printf "%-6s %-8s %-10s %-8s %-12s %-12s %-12s %-15s %-12s\n" \
       "L" "Reqs" "Duration" "QPS" "embed_ms(avg)" "search_ms(avg)" "total_ms(avg)" "DA_batch_ms(avg)" "map_ms(avg)"
SAMPLE_FILE=$(mktemp)
shuf -n "$COUNT" "$QUERIES" > "$SAMPLE_FILE"
for L in $L_LIST; do
  START=$(date +%s)
  if [ "$REQUEST_MODE" = "batch" ]; then
    QUERIES_JSON=$(mktemp)
    # Build JSON array of queries from the fixed sample
    jq -R -s 'split("\n")[:-1]' "$SAMPLE_FILE" > "$QUERIES_JSON"
    # Single batched POST: stream payload to curl via stdin
    RESP=$(jq -n \
      --slurpfile queries "$QUERIES_JSON" \
      --argjson k "$K" --argjson L "$L" --argjson W "$W" --argjson T "$THREADS" \
      '{queries:$queries[0], n_docs:$k, backend:"diskann", diskann_L:$L, diskann_W:$W, diskann_threads:$T}' \
      | curl -s -X POST -H "Content-Type: application/json" --data-binary @- "$HOST"/search)
    END=$(date +%s)
    DUR=$((END-START))
    if [ "$DUR" -le 0 ]; then DUR=1; fi
    QPS=$(awk -v c="$COUNT" -v d="$DUR" 'BEGIN{ printf "%.2f", c/d }')
    # Extract per-query averages by dividing totals by COUNT
    READS=$(printf '%s' "$RESP" | jq -r --argjson n "$COUNT" '(.results.timings_ms) as $t | (.results.backend_timings_ms // {}) as $b | [($t.embed/$n), ($t.search/$n), ($t.total/$n), (($b.diskann_batch // 0)/$n), (($b.mapping // 0)/$n)] | @tsv')
    EMBED=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $1}')
    SEARCH=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $2}')
    TOTAL=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $3}')
    DABATCH=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $4}')
    MAP=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $5}')
    printf "%-6s %-8s %-10ss %-8s %-12s %-12s %-12s %-15s %-12s\n" "$L" "$COUNT" "$DUR" "$QPS" "$EMBED" "$SEARCH" "$TOTAL" "$DABATCH" "$MAP"
    rm -f "$QUERIES_JSON"
  else
    EMBED_TOTAL=0
    SEARCH_TOTAL=0
    TOTAL_TOTAL=0
    DABATCH_TOTAL=0
    MAP_TOTAL=0
    SUCCESS=0
    while IFS= read -r QUERY; do
      [ -z "$QUERY" ] && continue
      RESP=$(jq -n \
        --arg q "$QUERY" \
        --argjson k "$K" --argjson L "$L" --argjson W "$W" --argjson T "$THREADS" \
        '{queries:[$q], n_docs:$k, backend:"diskann", diskann_L:$L, diskann_W:$W, diskann_threads:$T}' \
        | curl -s -X POST -H "Content-Type: application/json" --data-binary @- "$HOST"/search)
      READS=$(printf '%s' "$RESP" | jq -r '(.results.timings_ms) as $t | (.results.backend_timings_ms // {}) as $b | [($t.embed // 0), ($t.search // 0), ($t.total // 0), (($b.diskann_batch // 0)), (($b.mapping // 0))] | @tsv' 2>/dev/null || true)
      if [ -z "$READS" ]; then
        echo "Warning: empty response for query '${QUERY:0:60}'" >&2
        continue
      fi
      IFS=$'\t' read -r EMBED_ONE SEARCH_ONE TOTAL_ONE DABATCH_ONE MAP_ONE <<<"$READS"
      EMBED_TOTAL=$(awk -v total="$EMBED_TOTAL" -v val="$EMBED_ONE" 'BEGIN{printf "%.6f", total + val}')
      SEARCH_TOTAL=$(awk -v total="$SEARCH_TOTAL" -v val="$SEARCH_ONE" 'BEGIN{printf "%.6f", total + val}')
      TOTAL_TOTAL=$(awk -v total="$TOTAL_TOTAL" -v val="$TOTAL_ONE" 'BEGIN{printf "%.6f", total + val}')
      DABATCH_TOTAL=$(awk -v total="$DABATCH_TOTAL" -v val="$DABATCH_ONE" 'BEGIN{printf "%.6f", total + val}')
      MAP_TOTAL=$(awk -v total="$MAP_TOTAL" -v val="$MAP_ONE" 'BEGIN{printf "%.6f", total + val}')
      SUCCESS=$((SUCCESS + 1))
      if [ "$SUCCESS" -ge "$COUNT" ]; then
        break
      fi
    done < "$SAMPLE_FILE"
    END=$(date +%s)
    if [ "$SUCCESS" -eq 0 ]; then
      echo "No successful queries for L=$L" >&2
      continue
    fi
    DUR=$((END-START))
    if [ "$DUR" -le 0 ]; then DUR=1; fi
    QPS=$(awk -v c="$SUCCESS" -v d="$DUR" 'BEGIN{ printf "%.2f", c/d }')
    EMBED=$(awk -v total="$EMBED_TOTAL" -v n="$SUCCESS" 'BEGIN{ printf "%.2f", total/n }')
    SEARCH=$(awk -v total="$SEARCH_TOTAL" -v n="$SUCCESS" 'BEGIN{ printf "%.2f", total/n }')
    TOTAL=$(awk -v total="$TOTAL_TOTAL" -v n="$SUCCESS" 'BEGIN{ printf "%.2f", total/n }')
    DABATCH=$(awk -v total="$DABATCH_TOTAL" -v n="$SUCCESS" 'BEGIN{ printf "%.2f", total/n }')
    MAP=$(awk -v total="$MAP_TOTAL" -v n="$SUCCESS" 'BEGIN{ printf "%.2f", total/n }')
    printf "%-6s %-8s %-10ss %-8s %-12s %-12s %-12s %-15s %-12s\n" "$L" "$SUCCESS" "$DUR" "$QPS" "$EMBED" "$SEARCH" "$TOTAL" "$DABATCH" "$MAP"
  fi
done
rm -f "$SAMPLE_FILE"


