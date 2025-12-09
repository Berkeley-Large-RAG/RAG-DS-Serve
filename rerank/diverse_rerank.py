import os
import numpy as np
import logging
import time
import struct
import threading
import torch
from types import SimpleNamespace
from rerank.exact_rerank import embed_passages
from normalize_text import normalize as nm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
CACHE_DIR = "./embedding_cache"

PRECOMPUTED_EMB_PATH = os.environ.get(
    "DISKANN_EMBED_BIN",
    "/mnt/data/jinjian/DS-Serve/DiskANN-build/converted/vectors.bin",
)
_PRECOMP_STORE = None
_PRECOMP_ERROR = False
_STORE_LOCK = threading.Lock()


class PrecomputedEmbeddingStore:
    def __init__(self, path: str) -> None:
        self.path = path
        with open(self.path, "rb") as f:
            header = f.read(8)
            if len(header) != 8:
                raise ValueError(f"{self.path}: invalid vectors.bin header")
            self.total_rows, self.dim = struct.unpack("<II", header)
        if self.total_rows <= 0 or self.dim <= 0:
            raise ValueError(f"{self.path}: invalid shape ({self.total_rows}, {self.dim})")
        self._memmap = np.memmap(
            self.path,
            dtype=np.float32,
            mode="r",
            offset=8,
            shape=(self.total_rows, self.dim),
        )
        print(
            f"[DIVERSE-EMB] Precomputed store ready "
            f"(rows={self.total_rows:,}, dim={self.dim}) from {self.path}"
        )

    def get(self, index_id):
        if index_id is None:
            return None
        try:
            idx = int(index_id)
        except Exception:
            return None
        if idx < 0 or idx >= self.total_rows:
            return None
        return np.array(self._memmap[idx], copy=True)


def get_precomputed_store():
    global _PRECOMP_STORE, _PRECOMP_ERROR
    if _PRECOMP_ERROR:
        return None
    with _STORE_LOCK:
        if _PRECOMP_STORE is not None:
            return _PRECOMP_STORE
        path = PRECOMPUTED_EMB_PATH
        if not path or not os.path.exists(path):
            _PRECOMP_ERROR = True
            logging.warning(f"[DIVERSE-EMB] Precomputed vectors not found at {path}. Skipping diverse rerank.")
            return None
        try:
            _PRECOMP_STORE = PrecomputedEmbeddingStore(path)
        except Exception as exc:
            _PRECOMP_ERROR = True
            logging.warning(f"[DIVERSE-EMB] Failed to load precomputed embeddings ({exc}). Skipping diverse rerank.")
            return None
        return _PRECOMP_STORE


def fetch_precomputed_embeddings(raw_passages):
    store = get_precomputed_store()
    if store is None:
        return None
    vectors = []
    for p in raw_passages:
        idx = p.get("index_id")
        vec = store.get(idx)
        if vec is None:
            logging.warning(f"[DIVERSE-EMB] Missing embedding for index_id {idx}; skipping rerank.")
            return None
        vectors.append(vec)
    if not vectors:
        return None
    return np.vstack(vectors)


def _run_diverse_mmr(raw_passages, ctx_embeddings, query_emb, lambda_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx_embeddings = torch.tensor(ctx_embeddings, dtype=torch.float32, device=device)  # (N, D)
    N = ctx_embeddings.shape[0]

    if "exact score" in raw_passages[0]:
        rel_scores = torch.tensor(
            [p["exact score"] for p in raw_passages], dtype=torch.float32, device=device
        )
        logging.info("Using precomputed 'exact score' as relevance score.")
    else:
        query_emb = torch.tensor(query_emb, dtype=torch.float32, device=device)  # (1, D)
        rel_scores = torch.matmul(ctx_embeddings, query_emb.T).squeeze(-1)  # (N,)
        logging.info("Computed relevance score via dot product with query.")

    sim_matrix = torch.matmul(ctx_embeddings, ctx_embeddings.T)  # (N, N)

    selected = []
    selected_idxs = []
    candidate_idxs = list(range(N))

    for _ in range(N):
        if not candidate_idxs:
            break

        if selected_idxs:
            div_penalty = sim_matrix[candidate_idxs][:, selected_idxs].sum(dim=1)
        else:
            div_penalty = torch.zeros(len(candidate_idxs), device=device)

        scores = rel_scores[candidate_idxs] - lambda_val * div_penalty
        best_idx_local = torch.argmax(scores).item()
        best_idx = candidate_idxs[best_idx_local]

        selected.append(raw_passages[best_idx])
        selected_idxs.append(best_idx)
        candidate_idxs.remove(best_idx)

    return selected


def diverse_rerank_topk(raw_passages, query_encoder, lambda_val=0.5):
    total_start = time.time()
    logging.info(f"Doing diverse reranking for {len(raw_passages)} total evaluation samples using lambda {lambda_val}...")

    raw_query = raw_passages[0]['raw_query']

    # Try to use precomputed embeddings first (fast path)
    ctx_embeddings = fetch_precomputed_embeddings(raw_passages)
    if ctx_embeddings is not None:
        args = SimpleNamespace(
            per_gpu_batch_size=128,
            get=lambda k, default=None: default 
        )
        query_only = embed_passages(
            args=args,
            raw_query=raw_query,
            texts=[],
            passage_ids=[],
            model=query_encoder,
        )
        query_emb = query_only[raw_query].reshape(1, -1)
        reranked = _run_diverse_mmr(raw_passages, ctx_embeddings, query_emb, lambda_val)
        logging.info(f"[DIVERSE-EMB] Precomputed rerank finished in {time.time() - total_start:.2f}s")
        return reranked

    # Fallback to live re-embedding if the store is unavailable
    logging.warning("[DIVERSE-EMB] Precomputed embeddings unavailable; re-embedding passages on the fly.")
    step_start = time.time()
    ctxs = [nm(p["text"].lower()) for p in raw_passages]
    ctx_ids = [p["passage_id"] for p in raw_passages]
    args = SimpleNamespace(
        per_gpu_batch_size=128,
        get=lambda k, default=None: default 
    )
    text_to_embeddings = embed_passages(
        args=args,
        raw_query=raw_query,
        texts=ctxs,
        passage_ids=ctx_ids,
        model=query_encoder,
    )
    query_embedding = text_to_embeddings[raw_query].reshape(1, -1)
    ctx_embeddings = np.vstack([text_to_embeddings[text] for text in ctxs])
    logging.info(f"[DIVERSE-EMB] On-the-fly embedding took {time.time() - step_start:.2f}s")

    reranked = _run_diverse_mmr(raw_passages, ctx_embeddings, query_embedding, lambda_val)
    logging.info(f"Total rerank time: {time.time() - total_start:.2f}s")
    return reranked


def diverse_rerank_precomputed(raw_passages, query_embedding=None, query_encoder=None, lambda_val=0.5):
    ctx_embeddings = fetch_precomputed_embeddings(raw_passages)
    if ctx_embeddings is None:
        logging.warning("[DIVERSE-EMB] Precomputed embeddings unavailable; returning ANN order.")
        return raw_passages
    if query_embedding is not None:
        query_emb = np.asarray(query_embedding)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
    else:
        if query_encoder is None:
            logging.warning("[DIVERSE-EMB] Missing query encoder; returning ANN order.")
            return raw_passages
        args = SimpleNamespace(
            per_gpu_batch_size=128,
            get=lambda k, default=None: default 
        )
        query_only = embed_passages(
            args=args,
            raw_query=raw_passages[0]['raw_query'],
            texts=[],
            passage_ids=[],
            model=query_encoder,
        )
        query_emb = query_only[raw_passages[0]['raw_query']].reshape(1, -1)
    return _run_diverse_mmr(raw_passages, ctx_embeddings, query_emb, lambda_val)
