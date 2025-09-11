import os
import json
import pickle as pkl
import logging
from tqdm import tqdm
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rerank import contriever
from normalize_text import normalize as nm
os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_embedding_from_cache(passage_id):
    cache_path = os.path.join(CACHE_DIR, f"{passage_id}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None

def save_embedding_to_cache(passage_id, embedding):
    cache_path = os.path.join(CACHE_DIR, f"{passage_id}.npy")
    np.save(cache_path, embedding)

# Old
# def embed_passages(args, raw_query, texts, passage_ids, model):
#     print(f"Embedding {len(texts)} passages (after caching)...")
#     texts_to_embed = []
#     ids_to_embed = []
#     embeddings = {}

#     texts_to_embed.append(raw_query) # Always skip caching for raw_query
#     ids_to_embed.append(None)
#     for text, pid in zip(texts, passage_ids):
#         cached = load_embedding_from_cache(pid)
#         if cached is not None:
#             embeddings[text] = cached
#         else:
#             texts_to_embed.append(text)
#             ids_to_embed.append(pid)

#     if texts_to_embed:
#         encoded = model.encode(texts_to_embed, batch_size=min(128, args.per_gpu_batch_size), instruction="<|embed|>\n")
#         for pid, emb, original_text in zip(ids_to_embed, encoded, texts_to_embed):
#             if np.isnan(emb).any():
#                 logging.warning(f"[WARNING] Skipping text due to NaN in embedding: {original_text}")
#                 continue
#             embeddings[original_text] = emb
#             if pid is not None:
#                 save_embedding_to_cache(pid, emb)  # Cache only if it's a passage embedding, not the query -- query has None as pid to differ
#     return embeddings

def embed_passages(args, raw_query, texts, passage_ids, model):
    # Decide what we can pull from cache vs. what we need to compute
    texts_to_embed = []
    ids_to_embed = []
    embeddings = {}

    # Always (re)compute the query; do not cache queries
    texts_to_embed.append(raw_query)
    ids_to_embed.append(None)

    cached_hit_ids = []
    cached_miss_ids = []

    for text, pid in zip(texts, passage_ids):
        cached = load_embedding_from_cache(pid)
        if cached is not None:
            embeddings[text] = cached
            cached_hit_ids.append(pid)
        else:
            texts_to_embed.append(text)
            ids_to_embed.append(pid)
            cached_miss_ids.append(pid)

    total = len(texts)
    hits = len(cached_hit_ids)
    misses = len(cached_miss_ids)

    # Clean, single-line summary of cache usage
    print(f"[CACHE] Using cached embeddings for {hits}/{total} passages; "
          f"computing {misses} new passage embeddings + 1 query.")
    if hits > 0:
        print(f"[CACHE] Cache directory: {os.path.abspath(CACHE_DIR)}")

    # Actually embed the (query + cache-miss passages)
    if texts_to_embed:
        encoded = model.encode(
            texts_to_embed,
            batch_size=min(128, args.per_gpu_batch_size),
            instruction="<|embed|>\n"
        )
        saved_count = 0
        for pid, emb, original_text in zip(ids_to_embed, encoded, texts_to_embed):
            if np.isnan(emb).any():
                logging.warning(f"[WARNING] Skipping text due to NaN in embedding: {original_text}")
                continue
            embeddings[original_text] = emb
            # Cache only passage embeddings (pid != None). Query has pid=None.
            if pid is not None:
                save_embedding_to_cache(pid, emb)
                saved_count += 1

        if saved_count > 0:
            print(f"[CACHE] Saved {saved_count} new passage embeddings to cache.")

    return embeddings


# Defaults to memory reranking with live passages
def exact_rerank_topk(raw_passages, query_encoder):

    logging.info(f"Doing exact reranking for {len(raw_passages)} total evaluation samples...")
    raw_query = raw_passages[0]['raw_query']
    #pre normalize
    ctxs = [nm(p["text"].lower()) for p in raw_passages]
    # print("Keys in passage[0]:", raw_passages[0].keys())
    ctx_ids = [p["passage_id"] for p in raw_passages]



    from types import SimpleNamespace

    args = SimpleNamespace(
        per_gpu_batch_size=128,
        get=lambda k, default=None: default 
    )
    print("Calling embed passages now")
    text_to_embeddings = embed_passages(
        args=args,
        raw_query=raw_query,
        texts=ctxs,
        passage_ids=ctx_ids,
        model=query_encoder
    )
    
    print("Calculating the exact similarity scores...")


    query_embedding = text_to_embeddings[raw_query].reshape(1, -1)
    for text in ctxs:
        if text not in text_to_embeddings:
            print(f"A text is not found in text_to_embeddings")
            print(f"[INFO] Embedding cache directory: {os.path.abspath(CACHE_DIR)}")
    ctx_embeddings = [text_to_embeddings[text] for text in ctxs]
    ctxs_embedding = np.vstack(ctx_embeddings)


    print("Query embedding shape:", query_embedding.shape)
    print("Ctxs embedding shape:", ctxs_embedding.shape)
    print("Query dtype:", query_embedding.dtype)
    print("Ctxs dtype:", ctxs_embedding.dtype)

    similarities = cosine_similarity(query_embedding, ctxs_embedding)
    print("Similarity calculation finished successfully!")

    for i, p in enumerate(raw_passages):
        p["exact score"] = float(similarities[0][i]) if not np.isnan(similarities[0][i]) else -1.0

    raw_passages.sort(key=lambda p: p["exact score"], reverse=True)
        

    return raw_passages
    


def normalize_text(text):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))

def search_topk(cfg):
    exact_rerank_topk(cfg)
