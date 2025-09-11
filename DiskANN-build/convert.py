#!/usr/bin/env python3
import argparse, os, sys, json, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# optional faster JSON
try:
    import orjson
    dumps = lambda o: orjson.dumps(o, option=orjson.OPT_SERIALIZE_NUMPY).decode()
except Exception:
    dumps = lambda o: json.dumps(o, separators=(",", ":"))

def iter_vecs(data):
    """Yield (id, vec16) from ndarray/list/dict; tolerate already-fp32."""
    if isinstance(data, dict):
        for k, v in data.items():
            a = np.asarray(v)
            if a.dtype != np.float16:
                a = a.astype(np.float16, copy=False)
            yield k, a
        return
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            for i in range(data.shape[0]):
                a = data[i]
                if a.dtype != np.float16:
                    a = a.astype(np.float16, copy=False)
                yield i, a
            return
        if data.ndim == 1:  # single vector
            a = data
            if a.dtype != np.float16:
                a = a.astype(np.float16, copy=False)
            yield 0, a
            return
    if hasattr(data, "__iter__"):
        for i, v in enumerate(data):
            a = np.asarray(v)
            if a.dtype != np.float16:
                a = a.astype(np.float16, copy=False)
            yield i, a
        return
    raise ValueError("Unsupported pickle layout")

def convert_one(pkl_path: Path, out_dir: Path, verify_first: int, force: bool):
    """Convert one .pkl → .jsonl(fp32). Verify only the first `verify_first` rows."""
    out_path = out_dir / (pkl_path.stem + ".jsonl")
    if out_path.exists() and not force:
        return str(pkl_path), True, f"skip (exists) {out_path.name}"

    try:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return str(pkl_path), False, f"pickle load error: {e}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    # stash first batch for quick verification
    v_ids, v_fp32 = [], []

    try:
        buf, CHUNK = [], 2048  # fewer syscalls
        with out_path.open("w", buffering=8*1024*1024) as w:
            for k, vec16 in iter_vecs(data):
                vec32 = vec16.astype(np.float32, copy=False)
                if len(v_ids) < verify_first:
                    v_ids.append(k)
                    v_fp32.append(vec32.copy())
                buf.append(dumps({"id": k, "embedding": vec32.tolist()}) + "\n")
                if len(buf) >= CHUNK:
                    w.write("".join(buf)); buf.clear()
                total += 1
            if buf:
                w.write("".join(buf))

        if total == 0:
            return str(pkl_path), False, "no rows"

        # Verify only the first batch by reading the first N lines back
        if verify_first > 0 and v_ids:
            need = len(v_ids)
            ok = 0
            with out_path.open("r") as r:
                for i in range(need):
                    line = r.readline()
                    if not line:
                        break
                    obj = json.loads(line)
                    if obj["id"] != v_ids[i]:
                        return str(pkl_path), False, f"verify id mismatch at line {i}"
                    arr = np.asarray(obj["embedding"], dtype=np.float32)
                    # tight check: exactly what we intended
                    if arr.shape == v_fp32[i].shape and np.allclose(arr, v_fp32[i], rtol=1e-6, atol=1e-6):
                        ok += 1
            if ok != need:
                return str(pkl_path), False, f"verify failed {ok}/{need}"

        return str(pkl_path), True, f"ok (rows={total}, verified_first={len(v_ids)})"

    except Exception as e:
        # clean up partials to avoid confusing reruns
        try: out_path.unlink(missing_ok=True)
        except Exception: pass
        return str(pkl_path), False, f"error: {e}"

def main():
    ap = argparse.ArgumentParser(description="Concurrent fp16 .pkl → fp32 JSONL converter (verify first batch).")
    ap.add_argument("input_dir", type=str)
    ap.add_argument("output_dir", type=str)
    ap.add_argument("--pattern", default="*.pkl")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--verify-first", type=int, default=1024, help="verify only the first N rows per file (0 disables)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    files = sorted(in_dir.rglob(args.pattern))
    if not files:
        print("No .pkl files found", file=sys.stderr); sys.exit(2)

    print(f"Found {len(files)} files | workers={args.workers} | verify-first={args.verify_first}")
    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_one, f, out_dir, args.verify_first, args.force): f for f in files}
        for fut in as_completed(futs):
            p, success, msg = fut.result()
            print(f"[{'OK' if success else 'FAIL'}] {p} -> {msg}")
            ok += int(success)
    print(f"\nDone: {ok}/{len(files)} succeeded.")
    sys.exit(0 if ok == len(files) else 1)

if __name__ == "__main__":
    main()
