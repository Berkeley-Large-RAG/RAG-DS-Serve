#!/usr/bin/env python3
"""
Download alrope/cpds_embeddings from Hugging Face with resume support.

Usage:
  python download_cpds_embeddings.py --dest ./cpds_embeddings \
    --patterns "*.pkl" "*.jsonl"  # optional filters
"""

import argparse
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="./cpds_embeddings", help="Target folder")
    ap.add_argument("--revision", default=None, help="Branch/tag/commit (optional)")
    ap.add_argument(
        "--patterns", nargs="*", default=None,
        help="Glob patterns to include (e.g., *.pkl *.jsonl). Omit to download all."
    )
    args = ap.parse_args()

    path = snapshot_download(
        repo_id="alrope/cpds_embeddings",
        repo_type="dataset",            # change to "model" if needed
        local_dir=args.dest,
        local_dir_use_symlinks=False,   # get real files instead of symlinks
        revision=args.revision,
        allow_patterns=args.patterns,   # None = no filter (download everything)
        resume_download=True,
        max_workers=8                   # tweak for your I/O
    )
    print(f"Downloaded to: {path}")

if __name__ == "__main__":
    main()
