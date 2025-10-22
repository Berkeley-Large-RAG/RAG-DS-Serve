#!/usr/bin/env python3
"""
Single-pass PKL shards → ONE DiskANN-ready vectors.bin (embeddings only).

Outputs:
  - vectors.bin  (little-endian header <uint32 N><uint32 D> + float32 row-major payload)

Layouts accepted per shard (we only use the vectors component):
  - (anything, embeddings)  → second tuple element used as embeddings
  - dict {id: 1-D vec}     → values stacked as embeddings
  - ndarray (N, D)         → embeddings directly
  - iterable of 1-D vecs   → stacked as embeddings

Performance:
  - Concurrent processing with configurable worker threads (default: 64)
  - Memory-efficient streaming for large datasets

Safety:
  - Atomic writes via *.tmp + os.replace
  - Validates shapes, consistent dim across shards, finite values only
  - SIGINT (Ctrl-C) cancels cleanly and removes temp files

Usage:
  python convert.py INPUT_DIR OUTPUT_DIR \
      --pattern "*.pkl" \
      [--workers 64] [--overwrite]
"""

from __future__ import annotations

import argparse
import os
import sys
import struct
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Cap BLAS threads BEFORE importing NumPy ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from tqdm.auto import tqdm

# ---- Deterministic shard ordering to match passage order ----
deprioritized_domains = ["massiveds-rpj_arxiv", "massiveds-rpj_github", "massiveds-rpj_book", "lb_full"]
deprioritized_domains_index = {d: i + 1 for i, d in enumerate(deprioritized_domains)}

def sort_key_embedding(fname: str):
    import re
    domain = fname.split('--', 1)[0]
    m = re.search(r'--passages(\d+)_(\d+)\.pkl$', fname)
    if m:
        rank = int(m.group(1))
        shard = int(m.group(2))
        return (deprioritized_domains_index.get(domain, 0), domain, rank, shard)
    # Non-matching names go last but remain stable by domain/name
    return (deprioritized_domains_index.get(domain, 0), domain, float('inf'), float('inf'))


# ---- Helper functions ----
def bytes_to_gb(bytes_size: int) -> str:
    """Convert bytes to GB with 2 decimal places."""
    gb = bytes_size / (1024 ** 3)
    return f"{gb:.2f}"


# ----------------------- helpers -----------------------

def _norm_str_from_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        return b.hex()

def _as_float32_matrix(x: Any) -> np.ndarray:
    """Return a C-order float32 2-D matrix; raise on wrong rank or non-finite."""
    arr = np.asarray(x, dtype=np.float32, order="C")
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite value (NaN/Inf) in embeddings")
    return arr

def _load_any_layout(path: Path) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Load a shard and normalize to (ids_or_None, embs32), where:
      - ids_or_None: (N,) int64 if present and integer-typed; otherwise None
      - embs32: (N, D) float32 C-order
    """
    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        _, embs = data  # Ignore first element
        E = _as_float32_matrix(embs)
        I = None  # Always set to None, ignoring any potential IDs
        return I, E

    if isinstance(data, dict):
        # Preserve insertion order (Py ≥3.7). If you need a stable order across Python versions,
        # consider sorting keys externally before creating the dict.
        vecs: List[np.ndarray] = []
        dim = None
        for v in data.values():
            v = np.asarray(v, dtype=np.float32)
            if v.ndim != 1:
                raise ValueError(f"{path}: dict values must be 1-D vectors, got {v.shape}")
            if dim is None:
                dim = v.shape[0]
            elif v.shape[0] != dim:
                raise ValueError(f"{path}: inconsistent vector dimensions in dict values")
            vecs.append(v)
        E = np.ascontiguousarray(np.vstack(vecs), dtype=np.float32)
        return None, E

    if isinstance(data, np.ndarray) and data.ndim == 2:
        E = _as_float32_matrix(data)
        return None, E  # no IDs present

    if hasattr(data, "__iter__"):
        # Materialize to determine N and D safely
        seq = [np.asarray(v, dtype=np.float32) for v in data]
        if not seq:
            raise ValueError(f"{path}: empty iterable")
        dim = seq[0].shape[0] if seq[0].ndim == 1 else None
        if dim is None:
            raise ValueError(f"{path}: sequence vectors must be 1-D")
        for i, v in enumerate(seq):
            if v.ndim != 1:
                raise ValueError(f"{path}: vector at index {i} must be 1-D, got {v.shape}")
            if v.shape[0] != dim:
                raise ValueError(f"{path}: inconsistent vector dimensions in sequence")
        E = np.ascontiguousarray(np.vstack(seq), dtype=np.float32)
        return None, E

    raise ValueError(f"{path}: unsupported pickle layout; expected (ids, embeddings) or similar")


# ----------------------- streaming core -----------------------

def _process_single_shard(shard_path: Path, shard_idx: int) -> Tuple[int, Optional[np.ndarray], np.ndarray]:
    """
    Process a single shard and return (shard_idx, ids, embeddings).
    Includes sanity checks and progress reporting.
    """
    I, E = _load_any_layout(shard_path)

    # Quick sanity checks
    n_vectors, n_dims = E.shape
    n_ids = len(I) if I is not None else 0

    # Check 1: no ID checks (we ignore IDs entirely now)

    # Check 2: Basic statistics
    finite_count = np.isfinite(E).sum()
    total_elements = E.size
    finite_ratio = finite_count / total_elements

    # Check 3: skip ID stats

    # Estimate shard size (vectors * dims * 4 bytes for float32)
    estimated_gb = (n_vectors * n_dims * 4) / (1024 ** 3)

    # Print sanity check results
    print(f"✓ Shard {shard_idx:3d}: {shard_path.name}")
    print(f"  Vectors: {n_vectors:,} | Dims: {n_dims} | Est. size: ~{estimated_gb:.2f} GB")
    print(f"  Finite: {finite_ratio:.1%}")
    # IDs ignored
    print()

    return shard_idx, I, E

def convert_stream(
    in_root: Path,
    out_root: Path,
    pattern: str = "*.pkl",
    overwrite: bool = False,
    workers: int = 64,
) -> None:
    """
    Single-pass conversion:
      - Write placeholder header to vectors.bin.tmp
      - For each shard (once): load → validate → append vectors
      - Patch header with final N, D
      - Save ids.npy from temporary raw int64 file
    """
    files = sorted(in_root.rglob(pattern), key=lambda p: sort_key_embedding(p.name))
    # --- Runtime verification against passage_order.txt ---
    pass_txt = in_root / 'passage_order.txt'
    if pass_txt.exists():
        import re
        try:
            with pass_txt.open('r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            expected = []
            for name in lines:
                if '--raw_passages_' not in name:
                    raise SystemExit(f'malformed line in passage_order.txt: {name}')
                domain, rest = name.split('--raw_passages_', 1)
                m = re.match(r'^(\d+)-(\d+)-of-(\d+)\.jsonl$', rest)
                if not m:
                    raise SystemExit(f'pattern mismatch in passage_order.txt: {name}')
                rank = int(m.group(1)); shard = int(m.group(2))
                expected.append(f"{domain}--passages{rank}_{shard:02d}.pkl")
            actual = [p.name for p in files]
            if expected != actual:
                print('Order verification failed: embeddings do not match passage_order.txt', file=sys.stderr)
                for i, (e, a) in enumerate(zip(expected, actual)):
                    if e != a:
                        print(f'First mismatch at index {i}: expected {e} | got {a}', file=sys.stderr)
                        break
                sys.exit(2)
            else:
                print('Order verification passed: embeddings match passage_order.txt')
        except Exception as e:
            print(f'Order verification error: {e}', file=sys.stderr)
            sys.exit(2)

    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(2)

    out_root.mkdir(parents=True, exist_ok=True)
    bin_path = out_root / "vectors.bin"
    bin_tmp  = out_root / "vectors.bin.tmp"
    # No ids output

    if not overwrite and (bin_path.exists() or bin_tmp.exists()):
        print("Output exists (use --overwrite).", file=sys.stderr)
        sys.exit(2)

    # Clean any stale temps
    for p in (bin_tmp,):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    N_total = 0
    D: Optional[int] = None
    # IDs ignored

    try:
        # Process shards in batches to control memory usage
        batch_size = 15  # Process in small batches
        processed_count = 0

        with open(bin_tmp, "wb") as fvec:
            # Placeholder header (little-endian uint32 N, D)
            fvec.write(struct.pack("<II", 0, 0))

            for batch_start in tqdm(range(0, len(files), batch_size),
                                  desc="Processing batches",
                                  unit="batch"):
                batch_end = min(batch_start + batch_size, len(files))
                batch_files = files[batch_start:batch_end]
                batch_results = {}

                # Process this batch concurrently
                with ThreadPoolExecutor(max_workers=min(workers, len(batch_files))) as executor:
                    future_to_shard = {
                        executor.submit(_process_single_shard, p, batch_start + i): (batch_start + i, p)
                        for i, p in enumerate(batch_files)
                    }

                    # Collect results for this batch
                    for future in as_completed(future_to_shard):
                        shard_idx, shard_path = future_to_shard[future]
                        try:
                            result_idx, I, E = future.result()
                            batch_results[shard_idx] = (shard_path, I, E)
                        except Exception as exc:
                            raise SystemExit(f"{shard_path}: {exc}")

                # Write this batch immediately to free memory
                for shard_idx in sorted(batch_results.keys()):
                    shard_path, I, E = batch_results[shard_idx]

                    # Determine/validate dimension
                    if D is None:
                        D = int(E.shape[1])
                    elif E.shape[1] != D:
                        raise SystemExit(f"{shard_path}: dim mismatch {E.shape[1]} != {D}")

                    # Stream-append vectors only
                    E.tofile(fvec)       # float32 row-major payload

                    N_total += E.shape[0]

                # Clear batch results to free memory
                del batch_results
                processed_count += len(batch_files)

            # Patch real header at the beginning of vectors.bin.tmp
            if D is None:
                raise SystemExit("No data found.")
            if N_total >= 2**32:
                raise SystemExit(f"Too many vectors ({N_total}); DiskANN header uses uint32.")
            fvec.flush(); os.fsync(fvec.fileno())
            fvec.seek(0)
            fvec.write(struct.pack("<II", N_total, D))

        # Atomic rename to final output
        os.replace(bin_tmp, bin_path)

        # Get final file sizes for reporting
        size_bin = os.path.getsize(bin_path)

        # No ids temp to remove

        # Final summary
        print("\n🎉 Conversion Complete!")
        print("📊 Final Statistics:")
        print(f"   Total vectors: {N_total:,}")
        print(f"   Dimensions: {D}")
        print(f"   vectors.bin: {bytes_to_gb(size_bin)} GB")
        print(f"   Shards processed: {len(files)}")
        print(f"   Workers used: {min(workers, len(files))}")

        # Verify final data integrity
        print(f"\n🔍 Final Integrity Check:")

        # Quick sample of vectors.bin
        with open(bin_path, "rb") as f:
            header = f.read(8)
            final_n, final_d = struct.unpack("<II", header)
            print(f"   Header check: N={final_n:,}, D={final_d}")
            print(f"   ✓ All checks passed!")

    except KeyboardInterrupt:
        print("\n^C received, aborting…", file=sys.stderr)
        # Best-effort cleanup
        for p in (bin_tmp,):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        sys.exit(130)

    except Exception as e:
        # Cleanup on error
        for p in (bin_tmp,):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        raise


# ----------------------- CLI -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Single-pass PKL → DiskANN vectors.bin (embeddings only)")
    ap.add_argument("input_dir",  type=str, help="Folder with .pkl shards")
    ap.add_argument("output_dir", type=str, help="Folder to write vectors.bin")
    ap.add_argument("--pattern",  default="*.pkl", help='Glob for inputs (default: "*.pkl")')
    ap.add_argument("--workers", type=int, default=64,
                    help="Number of worker threads for concurrent processing (default: 64)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    return ap.parse_args()

def main():
    args = parse_args()
    in_root  = Path(args.input_dir)
    out_root = Path(args.output_dir)
    convert_stream(
        in_root=in_root,
        out_root=out_root,
        pattern=args.pattern,
        overwrite=args.overwrite,
        workers=args.workers,
    )

if __name__ == "__main__":
    main()
