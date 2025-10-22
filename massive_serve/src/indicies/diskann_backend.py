import os
import json
import sys
from pathlib import Path
import numpy as np

try:
    import diskannpy as dap  # type: ignore
except Exception as _e:
    dap = None


class DiskANNBackend:
    """Minimal, standalone DiskANN search backend with proven mapping logic.

    - Loads mapping arrays and filename list exactly like test_mapping.py
    - Uses diskannpy.StaticDiskIndex with explicit metadata (metric, dtype, dims, prefix)
    - Returns (scores, passages) to match server expectations
    """

    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]  # .../DS
        self._repo_root = repo_root
        self._passage_dir = str(repo_root / "data" / "passages")
        if not os.path.isdir(self._passage_dir):
            raise FileNotFoundError(f"PASSAGE_DIR not found: {self._passage_dir}")

        # Load arrays with mmap; load filename list with legacy alias
        self.position_array = np.load(str(repo_root / "position_array.npy"), mmap_mode="r")
        self.filename_index_array = np.load(str(repo_root / "filename_index_array.npy"), mmap_mode="r")
        try:
            sys.modules['numpy._core'] = np.core  # legacy alias, safe if present
        except Exception:
            pass
        self.filenames = np.load(str(repo_root / "filename_list.npy"), allow_pickle=True).tolist()
 
        # DiskANN params (env-overridable)
        self.index_dir = os.environ.get("DISKANN_INDEX_DIR", "/mnt/md-256k/jinjian/DiskANN_embeddings")
        self.index_prefix = os.environ.get("DISKANN_INDEX_PREFIX", "disk_index_compactDS_learn_R60_L80_B180_M600")
        # Metric: default to L2 (euclidean) to match L2-built indices
        self.metric = "l2"
        self.dimensions = int(os.environ.get("DISKANN_DIMENSIONS", "768"))
        self.vector_dtype = np.float32 if os.environ.get("DISKANN_VECTOR_DTYPE", "float32").lower() == "float32" else np.float16
        # Keep conservative default to avoid io_setup() EAGAIN on AIO
        self.num_threads = int(os.environ.get("DISKANN_NUM_THREADS", "2"))
        self.nodes_to_cache = int(os.environ.get("DISKANN_NODES_TO_CACHE", "100000"))

        self._index = None

    # ---------- mapping helpers ----------
    def _read_line_at(self, filename: str, byte_offset: int):
        path = os.path.join(self._passage_dir, filename)
        with open(path, 'rb') as f:
            f.seek(int(byte_offset))
            line = f.readline().decode('utf-8', errors='strict')
        obj = json.loads(line)
        return obj.get('text', ''), obj.get('id', None)

    def _ids_to_passages(self, ids, raw_query=None):
        out = []
        n = len(self.position_array)
        for idx in map(int, ids):
            if not (0 <= idx < n):
                continue
            pos = int(self.position_array[idx])
            fname_idx = int(self.filename_index_array[idx])
            if not (0 <= fname_idx < len(self.filenames)):
                continue
            fname = self.filenames[fname_idx]
            try:
                text, pid = self._read_line_at(fname, pos)
            except Exception:
                continue
            rec = {
                "passage_id": pid,
                "text": (text or "").strip(),
                "center_text": text,
                "source": fname.split('--')[0] if '--' in fname else 'unknown',
                "index_id": int(idx),
                "filename": fname,
                "position": int(pos),
            }
            if raw_query is not None:
                rec["raw_query"] = raw_query
            out.append(rec)
        return out

    # ---------- diskann ----------
    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        if dap is None:
            raise RuntimeError("diskannpy is not installed")
        self._index = dap.StaticDiskIndex(
            index_directory=self.index_dir,
            index_prefix=self.index_prefix,
            num_threads=self.num_threads,
            num_nodes_to_cache=self.nodes_to_cache,
            distance_metric=self.metric,
            vector_dtype=self.vector_dtype,
            dimensions=self.dimensions,
        )
        try:
            print(
                f"[DiskANN] Index loaded | dir={self.index_dir} prefix={self.index_prefix} "
                f"metric={self.metric} dim={self.dimensions} dtype={self.vector_dtype} "
                f"threads={self.num_threads} nodes_to_cache={self.nodes_to_cache}"
            )
        except Exception:
            pass

    def search(self, raw_query, query_embs: np.ndarray, k: int, L: int, W: int, threads: int | None = None, min_words: int | None = None):
        self._ensure_loaded()
        # Ensure float32 contiguous
        q = np.ascontiguousarray(query_embs.astype(np.float32))
        # Optional query normalization for L2/cosine
        try:
            do_norm = False
            if str(self.metric).lower() in ("cosine",):
                do_norm = True
            else:
                flag = os.environ.get("DISKANN_NORMALIZE_QUERY", "0")
                do_norm = flag.strip() in ("1", "true", "True")
            if do_norm:
                norms = np.linalg.norm(q, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                q = q / norms
                try:
                    print(f"[DiskANN] Applied unit L2 normalization to query embeddings (metric={self.metric}, shape={q.shape})")
                except Exception:
                    pass
        except Exception:
            # Best-effort normalization; proceed unnormalized on any error
            pass
        k = int(k)
        L = int(L)
        W = int(W) if W is not None else 2
        # threads: allow higher parallelism but keep an upper bound to avoid AIO saturation
        chosen = threads if (threads is not None) else self.num_threads
        try:
            eff_threads = max(1, min(int(chosen), 40))  # cap at 40
        except Exception:
            eff_threads = max(1, min(self.num_threads, 40))
        try:
            print(
                f"[DiskANN] Search request | batch={q.shape[0]} dim={q.shape[1]} k={k} "
                f"L={L} W={W} threads={eff_threads} min_words={min_words} dtype={q.dtype} "
                f"contiguous={q.flags['C_CONTIGUOUS']}"
            )
        except Exception:
            pass
        # Fetch a candidate pool, then slice to k after filtering (tunable via env)
        try:
            kfetch_env = os.environ.get("DISKANN_K_FETCH")
            if kfetch_env is not None and kfetch_env != "":
                K_FETCH = max(1, int(kfetch_env))
            else:
                # Heuristic: 4x requested k, bounded [64, 1000]
                K_FETCH = max(64, min(1000, int(max(k * 4, k))))
        except Exception:
            K_FETCH = 1000
        ids, dists = self._index.batch_search(
            queries=q,
            k_neighbors=K_FETCH,
            complexity=L,
            beam_width=W,
            num_threads=eff_threads,
        )
        try:
            ex = dists[0][:3].tolist() if hasattr(dists, "shape") and dists.size else []
            print(f"[DiskANN] batch_search -> K_FETCH={K_FETCH} ids.shape={getattr(ids,'shape',None)} dists.shape={getattr(dists,'shape',None)} dists_head={ex}")
        except Exception:
            pass
        all_passages = []
        all_scores = []
        for i in range(ids.shape[0]):
            q_text = raw_query[i] if isinstance(raw_query, list) else raw_query
            mapped = self._ids_to_passages(ids[i], raw_query=q_text)
            # Filter by min_words before slicing to k
            selected = []
            selected_scores = []
            for j, rec in enumerate(mapped):
                if min_words is not None:
                    try:
                        mw = int(min_words)
                    except Exception:
                        mw = 0
                    if mw > 0:
                        text = (rec.get("text") or "").strip()
                        if len(text.split()) < mw:
                            continue
                selected.append(rec)
                selected_scores.append(dists[i][j])
                if len(selected) >= k:
                    break

            all_passages.append(selected)
            all_scores.append(np.array(selected_scores))

        # Light debug to confirm parameters match expectations
        try:
            first_len = len(all_passages[0]) if all_passages else 0
            print(f"[DiskANN] L={L} W={W} threads={eff_threads} K_FETCH={K_FETCH} -> returned ~{first_len} after filter; target k={k}; min_words={min_words}")
        except Exception:
            pass

        return all_scores, all_passages


