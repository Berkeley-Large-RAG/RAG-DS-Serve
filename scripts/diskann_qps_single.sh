#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/e2e_qps_sweep.sh"

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
  echo "Cannot find ${SWEEP_SCRIPT}. Please ensure you are in the DS-Serve repo." >&2
  exit 1
fi

HOST=${HOST:-http://api.ds-serve.org:30888}
QUERIES=${QUERIES:-"$ROOT_DIR/e2e_queries_20000.txt"}
COUNT=${COUNT:-1000}
K=${K:-10}
W=${W:-8}
THREADS=${THREADS:-64}
L_LIST=${L_LIST:-"1000 1500 2000"}
WARMUP_SKIP=${WARMUP_SKIP:-100}

echo "[diskann-single] HOST=${HOST} COUNT=${COUNT} L_LIST=\"${L_LIST}\" WARMUP_SKIP=${WARMUP_SKIP}"

env \
  HOST="$HOST" \
  QUERIES="$QUERIES" \
  COUNT="$COUNT" \
  K="$K" \
  W="$W" \
  THREADS="$THREADS" \
  L_LIST="$L_LIST" \
  WARMUP_SKIP="$WARMUP_SKIP" \
  REQUEST_MODE=single \
  bash "$SWEEP_SCRIPT"
