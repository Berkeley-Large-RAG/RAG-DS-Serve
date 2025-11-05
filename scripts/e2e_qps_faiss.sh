#!/usr/bin/env bash
set -euo pipefail

# End-to-end FAISS (IVFPQ) QPS test against /search on a remote host.
# Supports batched (one POST with COUNT queries) or single (COUNT POSTs) modes.

# Config
HOST=${HOST:-http://api.ds-serve.org:30888}
QUERIES=${QUERIES:-/mnt/md-256k/jinjian/DS/e2e_queries_20000.txt}
COUNT=${COUNT:-4096}
CONCURRENCY=${CONCURRENCY:-128}   # used only in SINGLE mode
K=${K:-10}
NPROBE=${NPROBE:-128}
EXACT=${EXACT:-false}
DIVERSE=${DIVERSE:-false}
LAMBDA=${LAMBDA:-0.5}
BATCHED=${BATCHED:-1}              # 1 = single batched POST; 0 = many single POSTs
NPROBE_LIST=${NPROBE_LIST:-}

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (install via conda/apt)." >&2
  exit 1
fi

SAMPLE_FILE=$(mktemp)
shuf -n "$COUNT" "$QUERIES" > "$SAMPLE_FILE"

if [ -n "$NPROBE_LIST" ]; then
  printf "%-8s %-8s %-10s %-8s %-12s %-12s %-12s\n" \
         "nprobe" "Reqs" "Duration" "QPS" "embed_ms(avg)" "search_ms(avg)" "total_ms(avg)"
else
  printf "%-8s %-10s %-8s %-12s %-12s %-12s\n" \
         "Reqs" "Duration" "QPS" "embed_ms(avg)" "search_ms(avg)" "total_ms(avg)"
fi

if [ -n "$NPROBE_LIST" ] && [ "$BATCHED" = "1" ]; then
  # Sweep over NPROBE_LIST in batched mode, reusing the same sampled queries
  QUERIES_JSON=$(mktemp)
  jq -R -s 'split("\n")[:-1]' "$SAMPLE_FILE" > "$QUERIES_JSON"
  for NP in $NPROBE_LIST; do
    START=$(date +%s)
    RESP=$(jq -n \
      --slurpfile queries "$QUERIES_JSON" \
      --argjson k "$K" --argjson np "$NP" \
      --argjson ex "$EXACT" --argjson dv "$DIVERSE" --argjson lb "$LAMBDA" \
      '{queries:$queries[0], n_docs:$k, backend:"faiss", nprobe:$np, exact_search:$ex, diverse_search:$dv, lambda:$lb}' \
      | curl -s -X POST -H "Content-Type: application/json" --data-binary @- "$HOST"/search)
    END=$(date +%s)
    DUR=$((END-START)); if [ "$DUR" -le 0 ]; then DUR=1; fi
    QPS=$(awk -v c="$COUNT" -v d="$DUR" 'BEGIN{ printf "%.2f", c/d }')
    READS=$(printf '%s' "$RESP" | jq -r --argjson n "$COUNT" '(.results.timings_ms) as $t | [($t.embed/$n), ($t.search/$n), ($t.total/$n)] | @tsv')
    EMBED=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $1}')
    SEARCH=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $2}')
    TOTAL=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $3}')
    printf "%-8s %-8s %-10ss %-8s %-12s %-12s %-12s\n" "$NP" "$COUNT" "$DUR" "$QPS" "$EMBED" "$SEARCH" "$TOTAL"
  done
  rm -f "$QUERIES_JSON"
elif [ "$BATCHED" = "1" ]; then
  # Build JSON array of queries
  QUERIES_JSON=$(mktemp)
  jq -R -s 'split("\n")[:-1]' "$SAMPLE_FILE" > "$QUERIES_JSON"

  START=$(date +%s)
  RESP=$(jq -n \
    --slurpfile queries "$QUERIES_JSON" \
    --argjson k "$K" --argjson np "$NPROBE" \
    --argjson ex "$EXACT" --argjson dv "$DIVERSE" --argjson lb "$LAMBDA" \
    '{queries:$queries[0], n_docs:$k, backend:"faiss", nprobe:$np, exact_search:$ex, diverse_search:$dv, lambda:$lb}' \
    | curl -s -X POST -H "Content-Type: application/json" --data-binary @- "$HOST"/search)
  END=$(date +%s)
  rm -f "$QUERIES_JSON"

  DUR=$((END-START)); if [ "$DUR" -le 0 ]; then DUR=1; fi
  QPS=$(awk -v c="$COUNT" -v d="$DUR" 'BEGIN{ printf "%.2f", c/d }')
  READS=$(printf '%s' "$RESP" | jq -r --argjson n "$COUNT" '(.results.timings_ms) as $t | [($t.embed/$n), ($t.search/$n), ($t.total/$n)] | @tsv')
  EMBED=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $1}')
  SEARCH=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $2}')
  TOTAL=$(printf '%s' "$READS" | awk -F'\t' '{printf "%.2f", $3}')
  printf "%-8s %-10ss %-8s %-12s %-12s %-12s\n" "$COUNT" "$DUR" "$QPS" "$EMBED" "$SEARCH" "$TOTAL"
else
  START=$(date +%s)
  TMP=$(mktemp)
  jq -Rr @base64 < "$SAMPLE_FILE" | \
  xargs -P "$CONCURRENCY" -I{} bash -c '
    q=$(printf "%s" "$1" | base64 -d)
    jq -Rn --arg query "$q" --argjson k '"$K"' --argjson np '"$NPROBE"' \
      --argjson ex '"$EXACT"' --argjson dv '"$DIVERSE"' --argjson lb '"$LAMBDA"' \
      '"'"'{query:$query, n_docs:$k, backend:"faiss", nprobe:$np, exact_search:$ex, diverse_search:$dv, lambda:$lb}'"'"' \
    | curl -s -X POST -H "Content-Type: application/json" --data-binary @- '"$HOST"'/search
  ' _ {} | jq -r '(.results.timings_ms) as $t | [$t.embed, $t.search, $t.total] | @tsv' > "$TMP"
  END=$(date +%s)
  DUR=$((END-START)); if [ "$DUR" -le 0 ]; then DUR=1; fi
  QPS=$(awk -v c="$COUNT" -v d="$DUR" 'BEGIN{ printf "%.2f", c/d }')
  EMBED=$(awk -F'\t' '($1!="" && $1!="null"){sum+=$1; n++} END{ if(n>0) printf "%.2f", sum/n; else print "NA"}' "$TMP")
  SEARCH=$(awk -F'\t' '($2!="" && $2!="null"){sum+=$2; n++} END{ if(n>0) printf "%.2f", sum/n; else print "NA"}' "$TMP")
  TOTAL=$(awk -F'\t' '($3!="" && $3!="null"){sum+=$3; n++} END{ if(n>0) printf "%.2f", sum/n; else print "NA"}' "$TMP")
  printf "%-8s %-10ss %-8s %-12s %-12s %-12s\n" "$COUNT" "$DUR" "$QPS" "$EMBED" "$SEARCH" "$TOTAL"
  rm -f "$TMP"
fi

rm -f "$SAMPLE_FILE"

