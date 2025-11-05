import os
import json
import sys
import threading
from pathlib import Path
import numpy as np

try:
    import diskannpy as dap  # type: ignore
except Exception as _e:
    dap = None


class DiskANNBackend:
    """DiskANN search backend.

    - Loads mapping arrays and filename list exactly like test_mapping.py
    - Uses diskannpy.StaticDiskIndex with explicit metadata (metric, dtype, dims, prefix)
    - Returns (scores, passages) to match server expectations
    """

    def __init__(self) -> None:
        # need to fix the path to the download path TODO @yichaun
        ds_root_env = os.environ.get("DATASTORE_PATH")
        repo_root = Path(os.path.expanduser(ds_root_env)) if ds_root_env else Path("/mnt/md-256k/jinjian/DS")
        self._repo_root = repo_root
        # Allow override; otherwise try common layouts under DATASTORE_PATH
        passage_dir_env = os.environ.get("PASSAGE_DIR")
        if passage_dir_env and os.path.isdir(passage_dir_env):
            self._passage_dir = passage_dir_env
        else:
            candidate1 = repo_root / "data" / "passages"
            candidate2 = repo_root / "passages"
            self._passage_dir = str(candidate1 if candidate1.is_dir() else candidate2)
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
        # Metric: default to MIPS ('mips'); allow env override
        _metric_env = (os.environ.get("DISKANN_DISTANCE", "mips") or "").strip().lower()
        if _metric_env in ("ip", "inner", "inner_product", "mips"):
            self.metric = "mips"
        elif _metric_env in ("cos", "cosine"):
            self.metric = "cosine"
        else:
            self.metric = "l2"
        self.dimensions = int(os.environ.get("DISKANN_DIMENSIONS", "768"))
        self.vector_dtype = np.float32 if os.environ.get("DISKANN_VECTOR_DTYPE", "float32").lower() == "float32" else np.float16
        # Keep conservative default to avoid io_setup() EAGAIN on AIO
        self.num_threads = int(os.environ.get("DISKANN_NUM_THREADS", "2"))
        self.nodes_to_cache = int(os.environ.get("DISKANN_NODES_TO_CACHE", "100000"))

        # Startup log for quick sanity
        try:
            print(
                f"[DiskANN] Startup config | metric={self.metric} dim={self.dimensions} dtype={self.vector_dtype} "
                f"threads={self.num_threads} nodes_to_cache={self.nodes_to_cache} dir={self.index_dir} prefix={self.index_prefix}"
            )
        except Exception:
            pass

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

        # Optional warm-up to populate OS page cache and internal buffers
        try:
            warm = os.environ.get("DISKANN_WARMUP", "0").strip().lower() in ("1", "true", "yes")
            nq = int(os.environ.get("DISKANN_WARMUP_QUERIES", "0"))
            qfile = os.environ.get("DISKANN_WARMUP_QUERY_FILE", "").strip()
            if warm or nq > 0:
                if nq <= 0:
                    nq = 2000
                k = 1
                L = int(os.environ.get("DISKANN_L", "150"))
                W = int(os.environ.get("DISKANN_W", "4"))
                T = int(os.environ.get("DISKANN_NUM_THREADS", str(self.num_threads)))
                bs = max(1, int(os.environ.get("DISKANN_WARMUP_BATCH", "256")))
                print(f"[DiskANN] Warm-up: queries={nq} L={L} W={W} T={T} batch={bs} qfile={'set' if qfile else 'random'}")
                remaining = nq
                # Preload warmup queries
                def _yield_batches(total: int):
                    if qfile:
                        # DiskANN .bin layout: uint32 n, uint32 d, followed by n*d float32
                        with open(qfile, 'rb') as f:
                            hdr = np.fromfile(f, dtype=np.uint32, count=2)
                            n_total, d = int(hdr[0]), int(hdr[1])
                            if d != self.dimensions:
                                d = self.dimensions  # guard
                            to_read = min(total, n_total)
                            for off in range(0, to_read, bs):
                                cur = min(bs, to_read - off)
                                f.seek(8 + (off * d * 4))
                                arr = np.fromfile(f, dtype=np.float32, count=cur * d)
                                if arr.size != cur * d:
                                    break
                                yield arr.reshape(cur, d)
                    else:
                        remaining_local = total
                        while remaining_local > 0:
                            cur = bs if remaining_local >= bs else remaining_local
                            yield np.random.normal(size=(cur, self.dimensions)).astype(np.float32)
                            remaining_local -= cur
                while remaining > 0:
                    for q in _yield_batches(remaining):
                        cur = q.shape[0]
                        try:
                            _ids, _d = self._index.batch_search(
                                queries=q,
                                k_neighbors=k,
                                complexity=L,
                                beam_width=W,
                                num_threads=T,
                            )
                        except Exception:
                            remaining = 0
                            break
                        remaining -= cur

                # Optional keep-warm background loop
                keep = int(os.environ.get("DISKANN_WARMUP_KEEPALIVE", "0"))
                if keep > 0:
                    def _keep_warm():
                        try:
                            base_q = next(_yield_batches(max(bs, 256)))
                        except Exception:
                            base_q = np.random.normal(size=(max(bs, 256), self.dimensions)).astype(np.float32)
                        while True:
                            try:
                                self._index.batch_search(
                                    queries=base_q,
                                    k_neighbors=k,
                                    complexity=L,
                                    beam_width=W,
                                    num_threads=max(1, min(T, 32)),
                                )
                            except Exception:
                                pass
                            # sleep in seconds
                            try:
                                import time as _t
                                _t.sleep(float(keep))
                            except Exception:
                                break
                    th = threading.Thread(target=_keep_warm, daemon=True)
                    th.start()
        except Exception:
            pass

    def search(self, raw_query, query_embs: np.ndarray, k: int, L: int, W: int, threads: int | None = None, min_words: int | None = None):
        self._ensure_loaded()
        # Ensure float32 contiguous
        q = np.ascontiguousarray(query_embs.astype(np.float32))
        # Queries are used as-is (MIPS build); no query normalization.
        k = int(k)
        L = int(L)
        W = int(W) if W is not None else 2
        # threads: use exactly what is requested (no capping)
        eff_threads = max(1, int(threads if threads is not None else self.num_threads))
        try:
            print(
                f"[DiskANN] Search request | batch={q.shape[0]} dim={q.shape[1]} k={k} "
                f"L={L} W={W} threads={eff_threads} dtype={q.dtype} "
                f"contiguous={q.flags['C_CONTIGUOUS']}"
            )
        except Exception:
            pass
        # Determine K_FETCH (candidate oversampling)
        try:
            kfetch_env = os.environ.get("DISKANN_K_FETCH")
            if kfetch_env is not None and kfetch_env.strip() != "":
                K_FETCH = max(1, int(kfetch_env))
            else:
                K_FETCH = 1000
        except Exception:
            K_FETCH = 1000
        # Log a concise, clean effective config line for sanity
        try:
            print(
                f"[DiskANN] Effective config | metric={self.metric} L={L} W={W} K_FETCH={K_FETCH} "
                f"threads={eff_threads} dims={self.dimensions} dtype={q.dtype} "
                f"nodes_to_cache={self.nodes_to_cache} index={self.index_dir}/{self.index_prefix}"
            )
        except Exception:
            pass

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
            # No min_words filtering; take the first k mapped results
            # Old min_words filter block (commented out):
            # selected = []
            # selected_scores = []
            # for j, rec in enumerate(mapped):
            #     if min_words is not None:
            #         try:
            #             mw = int(min_words)
            #         except Exception:
            #             mw = 0
            #         if mw > 0:
            #             text = (rec.get("text") or "").strip()
            #             if len(text.split()) < mw:
            #                 continue
            #     selected.append(rec)
            #     selected_scores.append(dists[i][j])
            #     if len(selected) >= k:
            #         break
            selected = []
            selected_scores = []
            for j, rec in enumerate(mapped):
                selected.append(rec)
                selected_scores.append(dists[i][j])
                if len(selected) >= k:
                    break

            all_passages.append(selected)
            all_scores.append(np.array(selected_scores))

        # Light debug to confirm parameters match expectations
        try:
            first_len = len(all_passages[0]) if all_passages else 0
            print(f"[DiskANN] L={L} W={W} threads={eff_threads} K_FETCH={K_FETCH} -> returned ~{first_len}; target k={k}")
        except Exception:
            pass

        return all_scores, all_passages


