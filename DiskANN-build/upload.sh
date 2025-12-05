#!/usr/bin/env bash
set -euo pipefail

# Content to upload
DST="/mnt/data/jinjian/DS-Serve/DiskANN-build/sharded_diskann_index"
cd "$DST"

export HF_HUB_ENABLE_HF_TRANSFER=1

# Login HF
read -s -p "Paste your HF token: " HF_TOKEN; echo
hf auth login --token "$HF_TOKEN"

# Use upload large folder tool
HF_HUB_ENABLE_HF_TRANSFER=1 \
hf upload-large-folder Berkeley-Large-RAG/DiskANN_index \
  /mnt/data/jinjian/DS-Serve/DiskANN-build/sharded_diskann_index \
  --repo-type=dataset