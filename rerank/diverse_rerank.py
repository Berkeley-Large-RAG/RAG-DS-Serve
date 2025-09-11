import os
import numpy as np
import logging
import time
import torch
from sklearn.metrics.pairwise import cosine_similarity
from types import SimpleNamespace
from rerank.exact_rerank import embed_passages
from normalize_text import normalize as nm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
CACHE_DIR = "./embedding_cache"

def diverse_rerank_topk(raw_passages, query_encoder, lambda_val=0.5):
    total_start = time.time()
    logging.info(f"Doing diverse reranking for {len(raw_passages)} total evaluation samples using lambda {lambda_val}...")

    step_start = time.time()
    raw_query = raw_passages[0]['raw_query']
    ctxs = [nm(p["text"].lower()) for p in raw_passages]
    ctx_ids = [p["passage_id"] for p in raw_passages]
    logging.info(f"Step 1: Normalization done in {time.time() - step_start:.2f}s")

    args = SimpleNamespace(
        per_gpu_batch_size=128,
        get=lambda k, default=None: default 
    )

    step_start = time.time()
    text_to_embeddings = embed_passages(
        args=args,
        raw_query=raw_query,
        texts=ctxs,
        passage_ids=ctx_ids,
        model=query_encoder
    )
    logging.info(f"Step 2: Embedding done in {time.time() - step_start:.2f}s")

    step_start = time.time()
    query_emb = text_to_embeddings[raw_query].reshape(1, -1)
    ctx_embeddings = [text_to_embeddings[t] for t in ctxs]
    ctx_embeddings = np.vstack(ctx_embeddings)
    logging.info(f"Step 3: Embedding vector preparation done in {time.time() - step_start:.2f}s")

    # === Optimized greedy reranking ===
    step_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx_embeddings = torch.tensor(ctx_embeddings, dtype=torch.float32, device=device)  # (N, D)
    N = ctx_embeddings.shape[0]

    # === Use precomputed exact score if available ===
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

    logging.info(f"Step 4: Greedy selection loop done in {time.time() - step_start:.2f}s")
    logging.info(f"Total rerank time: {time.time() - total_start:.2f}s")

    return selected
