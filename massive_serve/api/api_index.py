import os
import time
import json
from massive_serve.contriever import Contriever, load_retriever
from transformers import AutoModel
import torch

from massive_serve.src.search import embed_queries
from massive_serve.src.indicies.base import Indexer
from massive_serve.src.indicies.diskann_backend import DiskANNBackend

import psutil

# Function to measure memory usage
def print_mem_use(label=""):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / 1024 / 1024
    print(f"[{label}] Memory usage: {mem_mb:.2f} MB")


device = 'cuda' if torch.cuda.is_available()  else 'cpu'

class DatastoreAPI():
    def __init__(self, cfg) -> None:
        self.domain_name = cfg.domain_name
        self._index = Indexer(cfg)
        self.index = self._index.datastore
        self.diskann = DiskANNBackend()
        
        # Hardcode Contriever for ANN embedding to match the built index
        model_name_or_path = "facebook/contriever-msmarco"
        tokenizer_name_or_path = "facebook/contriever-msmarco"
        if "contriever" in model_name_or_path:
            query_encoder, query_tokenizer, _ = load_retriever(model_name_or_path)
        elif "dragon" in model_name_or_path or "drama" in model_name_or_path or "ReasonIR" in model_name_or_path:
            query_encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        elif "GRIT" in model_name_or_path:
            from gritlm import GritLM
            query_tokenizer = None
            query_encoder = GritLM(model_name_or_path, torch_dtype="auto", mode="embedding")
        else:
            print(f"Trying to load {model_name_or_path} as a sentence transformer...")
            from sentence_transformers import SentenceTransformer
            query_tokenizer = None
            query_encoder = SentenceTransformer(model_name_or_path)
        self.query_encoder = query_encoder
        self.query_tokenizer = query_tokenizer
        self.query_encoder = self.query_encoder.to(device)
        self.cfg = cfg
    
    def search(self, query, n_docs, nprobe=None, expand_index_id=None, expand_offset=1, exact_rerank=False, use_diverse=False, lambda_val=0.5, backend=None, diskann_L=None, diskann_W=None, diskann_threads=None, min_words=None):
        print("✅ START OF search()")
        query_embedding = self.embed_query(query)
        if backend == 'diskann':
            L = int(diskann_L or 150)
            W = int(diskann_W or 2)
            threads = int(diskann_threads) if diskann_threads is not None else None
            searched_scores, searched_passages = self.diskann.search(query, query_embedding, int(n_docs), L, W, threads=threads)
            # Apply length filter similar to FAISS path
            if isinstance(searched_passages, list) and len(searched_passages) > 0 and min_words is not None:
                try:
                    mw = int(min_words)
                    for i in range(len(searched_passages)):
                        filtered = []
                        for p in searched_passages[i]:
                            text = (p.get('text') or '').strip()
                            if mw <= 0 or len(text.split()) >= mw:
                                filtered.append(p)
                        searched_passages[i] = filtered
                except Exception:
                    pass
        else:
            searched_scores, searched_passages  = self.index.search(
                query,
                query_embedding,
                n_docs,
                nprobe,
                expand_index_id,
                expand_offset,
                exact_rerank,
                use_diverse,
                lambda_val,
                use_diskann=False,
                min_words=min_words,
            )
        results = {'scores': searched_scores, 'passages': searched_passages}
        return results

    def embed_query(self, query):
        if isinstance(query, str):
            query_embedding = embed_queries(self.cfg, [query], self.query_encoder, self.query_tokenizer, self.cfg.query_encoder)
        elif isinstance(query, list):
            query_embedding = embed_queries(self.cfg, query, self.query_encoder, self.query_tokenizer, self.cfg.query_encoder)
        else:
            raise AttributeError(f"Query is not a string nor list!")
        return query_embedding
    

def get_datastore(cfg):
    print_mem_use("Before DSAPI")
    ds = DatastoreAPI(cfg)
    print_mem_use("After API before test search")
    test_search(ds)
    print_mem_use("After test search")
    return ds


def main(cfg):
    get_datastore(cfg, 0)


def test_search(ds):
    query = 'when was the last time anyone was on the moon?'  # 'scores': array([[44.3889  , 44.770973, 45.956238]], dtype=float32), 'IDs': [[[5, 45516], [6, 2218998], [5, 897337]]
    query2 = "who wrote he ain't heavy he's my brother lyrics?"  # 'scores': array([[33.60194 , 41.798004, 43.465225]], dtype=float32) 'IDs': [[[2, 361677], [5, 1717105], [2, 361675]]]
    search_results = ds.search(query, 1)
    print(search_results)



if __name__ == '__main__':
    main()
    
    