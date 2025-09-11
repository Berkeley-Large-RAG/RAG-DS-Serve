#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# --- Enable hf-transfer automatically if available ---
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print("[INFO] hf-transfer detected: using multi-connection downloads 🚀")
except ImportError:
    print("[WARN] hf-transfer not installed. Install with:")
    print("       pip install -U hf-transfer")
    print("       (downloads will still work, just slower)")

# --- Download ---
out_dir = Path("/home/yichuan_wang/data15t/embeddings").expanduser().resolve()
os.makedirs(out_dir, exist_ok=True)

path = snapshot_download(
    repo_id="alrope/cpds_embeddings",
    repo_type="dataset",
    local_dir=str(out_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=84   
)

print(f"[DONE] Downloaded to: {path}")
