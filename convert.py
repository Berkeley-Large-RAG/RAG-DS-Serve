#!/usr/bin/env python3
"""
fp16/other → fp32 JSONL converter (simple, safe, fast).

- Input: .pkl files (dict id->vec, ndarray (N,D)/(D,), or iterable of vecs)
- Output: one .jsonl per input, each line: {"id": <id>, "embedding": [fp32,...]}
- Crash-safe via atomic write (temp + os.replace)
- Verifies first N rows by re-reading the output
- Rejects NaN/Inf (fail fast)
- Parallel across files with a single progress bar

Usage:
  python convert.py INPUT_DIR OUTPUT_DIR
  # options: --pattern "*.pkl" --workers 16 --verify-first 256 --overwrite
"""

from __future__ import annotations
import argparse, sys, json, pickle, os, tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Tuple
import numpy as np
from tqdm import tqdm

# ---- Fast JSON if available; safe fallback otherwise ----
try:
    import orjson
    _USE_ORJSON = True
    _ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE
except Exception:
    orjson = None
    _USE_ORJSON = False

def dumps_bytes(obj) -> bytes:
    if _USE_ORJSON:
        return orjson.dumps(obj, option=_ORJSON_OPTS)
    # stdlib json cannot serialize numpy arrays -> convert lists; add newline
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")

def norm_id(k):
    # Keep integers as ints; stringify everything else deterministically
    if isinstance(k, (int, np.integer)):
        return int(k)
    return str(k)

def iter_vecs(data) -> Iterable[Tuple[object, np.ndarray]]:
    """
    Yield (id, vec32) ensuring float32 dtype. Supports:
      - dict: {id -> vector-like}
      - ndarray: (N, D) or (D,)
      - generic iterables: [vector-like, ...]
    """
    if isinstance(data, dict):
        for k, v in data.items():
            yield k, np.asarray(v, dtype=np.float32, order="C")
        return

    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            for i in range(data.shape[0]):
                yield i, np.asarray(data[i], dtype=np.float32, order="C")
            return
        if data.ndim == 1:
            yield 0, np.asarray(data, dtype=np.float32, order="C")
            return

    if hasattr(data, "__iter__"):
        for i, v in enumerate(data):
            yield i, np.asarray(v, dtype=np.float32, order="C")
        return

    raise ValueError("Unsupported pickle layout for embeddings")

def ensure_finite(vec32: np.ndarray, where: str) -> np.ndarray:
    if np.all(np.isfinite(vec32)):
        return vec32
    raise ValueError(f"Non-finite value (NaN/Inf) in {where}")

def convert_one(
    pkl_path: Path, out_dir: Path, verify_first: int, overwrite: bool
) -> Tuple[str, bool, str]:
    out_path = out_dir / (pkl_path.stem + ".jsonl")
    if out_path.exists() and not overwrite:
        return (str(pkl_path), True, f"skip (exists) {out_path.name}")

    # Load pickle (DO NOT use on untrusted inputs)
    try:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return (str(pkl_path), False, f"pickle load error: {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    wrote_rows = 0
    v_ids, v_fp32 = [], []

    try:
        # temp file in same directory for atomic finalize
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=out_path.parent, suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)

        # big buffered write
        with open(tmp_path, "wb", buffering=8 * 1024 * 1024) as w:
            for k, vec32 in iter_vecs(data):
                if vec32.dtype != np.float32:
                    vec32 = vec32.astype(np.float32, copy=False)
                vec32 = ensure_finite(vec32, pkl_path.name)

                rec = {"id": norm_id(k), "embedding": (vec32 if _USE_ORJSON else vec32.tolist())}

                if verify_first > 0 and len(v_ids) < verify_first:
                    v_ids.append(rec["id"])
                    v_fp32.append(vec32.copy())

                w.write(dumps_bytes(rec))
                wrote_rows += 1

        if wrote_rows == 0:
            try:
                if tmp_path:
                    tmp_path.unlink(missing_ok=True)
            finally:
                return (str(pkl_path), False, "no rows")

        # atomic move -> never leaves partial final
        os.replace(tmp_path, out_path)
        tmp_path = None

        # verify prefix by re-reading output
        if verify_first > 0 and v_ids:
            need, ok = len(v_ids), 0
            with out_path.open("rb") as r:
                for i in range(need):
                    line = r.readline()
                    if not line:
                        break
                    obj = json.loads(line)
                    if obj.get("id") != v_ids[i]:
                        return (str(pkl_path), False, f"verify id mismatch at line {i}")
                    arr = np.asarray(obj.get("embedding"), dtype=np.float32)
                    if arr.shape == v_fp32[i].shape and np.allclose(arr, v_fp32[i], rtol=1e-6, atol=1e-6):
                        ok += 1
            if ok != need:
                return (str(pkl_path), False, f"verify failed {ok}/{need}")

        return (str(pkl_path), True, f"ok (rows={wrote_rows}, verified_first={len(v_ids)})")

    except Exception as e:
        # cleanup temps/partials
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return (str(pkl_path), False, f"error: {e}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="fp16/other → fp32 JSONL converter (atomic, verified, fast)."
    )
    ap.add_argument("input_dir", type=str, help="Folder with .pkl files")
    ap.add_argument("output_dir", type=str, help="Folder to write .jsonl files")
    ap.add_argument("--pattern", default="*.pkl", help="Glob to match inputs (default: *.pkl)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2),
                    help="Worker processes (default: ~half CPU cores)")
    ap.add_argument("--verify-first", type=int, default=256,
                    help="Verify only the first N rows per file (0 disables)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return ap.parse_args()

def main():
    # single-thread BLAS, we’re I/O-bound
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    files = sorted(in_dir.rglob(args.pattern))
    if not files:
        print("No input files found matching pattern.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(files)} file(s) | workers={args.workers} | verify-first={args.verify_first} | "
          f"{'orjson' if _USE_ORJSON else 'json'}")

    ok = fails = 0
    messages = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(convert_one, f, out_dir, args.verify_first, args.overwrite): f
            for f in files
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Converting", unit="file"):
            p, success, msg = fut.result()
            messages.append((success, p, msg))
            ok += int(success)
            fails += int(not success)

    if fails:
        print("\nFailures:")
        for success, p, msg in messages:
            if not success:
                print(f"  [FAIL] {p} -> {msg}")

    print(f"\nDone: {ok}/{len(files)} succeeded.")
    sys.exit(0 if ok == len(files) else 1)

if __name__ == "__main__":
    main()
