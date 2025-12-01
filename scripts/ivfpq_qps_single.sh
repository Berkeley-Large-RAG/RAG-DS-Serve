#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FAISS_SCRIPT="${SCRIPT_DIR}/e2e_qps_faiss.sh"

if [[ ! -f "$FAISS_SCRIPT" ]]; then
  echo "Cannot find ${FAISS_SCRIPT}. Please ensure you are in the DS-Serve repo." >&2
  exit 1
fi

HOST=${HOST:-http://api.ds-serve.org:30888}
QUERIES=${QUERIES:-"$ROOT_DIR/e2e_queries_20000.txt"}
COUNT=${COUNT:-1000}
CONCURRENCY=${CONCURRENCY:-64}
K=${K:-10}
NPROBE=${NPROBE:-256}
NPROBE_LIST=${NPROBE_LIST:-}
EXACT=${EXACT:-false}
DIVERSE=${DIVERSE:-false}
LAMBDA=${LAMBDA:-0.5}

echo "[ivfpq-single] HOST=${HOST} COUNT=${COUNT} NPROBE=${NPROBE} CONCURRENCY=${CONCURRENCY}"

env \
  HOST="$HOST" \
  QUERIES="$QUERIES" \
  COUNT="$COUNT" \
  CONCURRENCY="$CONCURRENCY" \
  K="$K" \
  NPROBE="$NPROBE" \
  NPROBE_LIST="$NPROBE_LIST" \
  EXACT="$EXACT" \
  DIVERSE="$DIVERSE" \
  LAMBDA="$LAMBDA" \
  BATCHED=0 \
  bash "$FAISS_SCRIPT"
