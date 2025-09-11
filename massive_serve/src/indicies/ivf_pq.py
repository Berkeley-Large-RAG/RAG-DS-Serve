import os
import json
import random
import logging
import pickle
import numpy as np
import time
import glob
from tqdm import tqdm
from rerank.exact_rerank import exact_rerank_topk
from rerank.diverse_rerank import diverse_rerank_topk
import pdb
import re
import faiss
import numpy as np
import torch
from massive_serve.api.serve import query_encoder
try:
    from massive_serve.api.serve import QUERY_LOGGER  # optional query logger
except Exception:
    QUERY_LOGGER = None

# Allow server to inject a logger post-import to avoid circular import timing
def set_query_logger(logger):
    global QUERY_LOGGER
    QUERY_LOGGER = logger

# Enforce single-threaded FAISS/BLAS for reproducibility during testing
try:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass
except Exception:
    pass

import psutil
import os
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger().setLevel(logging.INFO)

# Function to measure memory usage
def print_mem_use(label=""):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / 1024 / 1024
    print(f"[{label}] Memory usage: {mem_mb:.2f} MB")



def extract_keywords(text):
    # Remove punctuation but preserve words like "GPT-3" or "smart-home"
    text = re.sub(r'[^\w\s\-]', '', text)
    words = text.lower().split()
    generic = {"tell", "me", "more", "about", "what", "is", "are", "the", "a", "an", "explain", "does", "how"}
    return [w for w in words if w not in generic]

# Function to measure delaydef log_latency_breakdown(t0, t1, t2, t3, t4):
def log_latency_breakdown(t0, t1, t2, t3, t4):
    try:
        logging.info("\n Latency breakdown:")
        logging.info(f"  (1) Pre-search prep          : {round((t1 - t0)*1000, 2)} ms")
        logging.info(f"  (2) index.search             : {round((t2 - t1)*1000, 2)} ms")
        logging.info(f"  (3) get_retrieved_passages   : {round((t3 - t2)*1000, 2)} ms")
        logging.info(f"  (4) post-processing & return : {round((t4 - t3)*1000, 2)} ms")
        logging.info(f"  🔁 Total                     : {round((t4 - t0)*1000, 2)} ms\n")
    except Exception as e:
        logging.warning(f"⚠️ Failed to log latency: {e}")



os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class IVFPQIndexer(object):

    def __init__(self, 
                embed_paths,
                index_path,
                meta_file,
                trained_index_path,
                passage_dir=None,
                pos_map_save_path=None,
                sample_train_size=1000000,
                prev_index_path=None,
                dimension=768,
                dtype=np.float16,
                ncentroids=4096,
                probe=2048,
                num_keys_to_add_at_a_time=1000000,
                DSTORE_SIZE_BATCH=51200000,
                n_subquantizers=16,
                code_size=8,
                ):
    
        self.embed_paths = embed_paths  # list of paths where saved the embedding of all shards
        self.index_path = index_path  # path to store the final index
        self.meta_file = meta_file  # path to save the index id to db id map
        self.prev_index_path = prev_index_path  # add new data to it instead of training new clusters
        self.trained_index_path = trained_index_path  # path to save the trained index
        self.passage_dir = passage_dir
        self.pos_map_save_path = "/home/ubuntu/Jinjian/retrieval-scaling/built_index/index_mapping.np.npy"
        self.cuda = False

        self.sample_size = sample_train_size
        self.dimension = dimension
        self.ncentroids = ncentroids
        self.probe = 32
        self.num_keys_to_add_at_a_time = num_keys_to_add_at_a_time
        self.n_subquantizers = n_subquantizers
        self.code_size = code_size
        self.position_array = np.load("/home/ubuntu/massive-serve-dev/index_dev/position_array.npy")
        self.filename_index_array = np.load("/home/ubuntu/massive-serve-dev/index_dev/filename_index_array.npy")
        self.filenames = self._load_filenames()

        print("Loading index...")
        start_idx = time.time()
        # try:
        #     self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        #     print("[IVFPQIndexer] Loaded FAISS index with mmap (on-disk)")
        # except Exception as e:
        # print(f"[IVFPQIndexer] mmap failed ({e}), falling back to in-memory load")
        self.index = faiss.read_index(index_path)
        end_idx = time.time()
        #print_mem_use("load_index after faiss.read_index")
        self.index.nprobe = self.probe
        print(f"[load_index] Set nprobe to {self.probe}")
        print(f"Index load time: {end_idx - start_idx:.2f} seconds")

        if self.passage_dir is not None:
            #print_mem_use("load_index before loading position arrays")
            start_map = time.time()
            self.position_array = np.load("/home/ubuntu/massive-serve-dev/index_dev/position_array.npy")
            self.filename_index_array = np.load("/home/ubuntu/massive-serve-dev/index_dev/filename_index_array.npy")
            self.filenames = self._load_filenames()
            end_map = time.time()
            print(f"Position map load time: {end_map - start_map:.2f} seconds")
            #print_mem_use("load_index after loading position arrays")



    def _load_filenames(self):
        # This must match the ordering used when building the arrays
        print(f"passage dir is {self.passage_dir}")
        filenames = [f for f in os.listdir(self.passage_dir) if f.endswith(".jsonl")]
        filenames = sorted(filenames, key=self._sort_func)
        return filenames

    def _sort_func(self, filename):
        domain = filename.split('raw_passages')[0].split('--')[0]
        rank = int(filename.split('passages_')[-1].split("-")[0])
        shard_idx = int(filename.split("-of-")[0].split("-")[-1])
        deprioritized_domains = ["massiveds-rpj_arxiv", "massiveds-rpj_github", "massiveds-rpj_book", "lb_full"]
        deprioritized_domains_index = {domain: i + 1 for i, domain in enumerate(deprioritized_domains)}
        deprioritized = deprioritized_domains_index.get(domain, 0)
        return (deprioritized, domain, rank, shard_idx)     

    def load_embeds(self, shard_id=None):
        all_ids, all_embeds = [], []
        offset = 0
        for embed_path in self.embed_paths:
            loaded_shard_id = int(re.search(r'_(\d+).pkl$', embed_path).group(1))
            if shard_id is not None and loaded_shard_id != shard_id:
                continue
            print(f"Loading pickle embedding from {embed_path}...")
            with open(embed_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)
            all_ids.extend([i + offset for i in ids])
            all_embeds.extend(embeddings)
            offset += len(ids)
        all_embeds = np.stack(all_embeds).astype(np.float32)
        datastore_size = len(all_ids)
        return all_embeds
        
    def get_embs(self, indices=None, shard_id=None):
        if indices is not None:
            embs = self.embs[indices]
        elif shard_id is not None:
            embs = self.load_embeds(shard_id)
        return embs

    def get_knn_scores(self, query_emb, indices):
        assert self.embeds is not None
        embs = self.get_embs(indices) # [batch_size, k, dimension]
        scores = - np.sqrt(np.sum((np.expand_dims(query_emb, 1)-embs)**2, -1)) # [batch_size, k]
        return scores

    def _sample_and_train_index(self,):
        print(f"Sampling {self.sample_size} examples from {len(self.embed_paths)} files...")
        per_shard_sample_size = self.sample_size // len(self.embed_paths)
        all_sampled_embs = []
        for embed_path in self.embed_paths:
            print(f"Loading pickle embedding from {embed_path}...")
            with open(embed_path, "rb") as fin:
                _, embeddings = pickle.load(fin)
            shard_size = len(embeddings)
            print(f"Finished loading, sampling {per_shard_sample_size} from {shard_size} for training...")
            random_samples = np.random.choice(np.arange(shard_size), size=[min(per_shard_sample_size, shard_size)], replace=False)
            sampled_embs = embeddings[random_samples]
            all_sampled_embs.extend(sampled_embs)
        all_sampled_embs = np.stack(all_sampled_embs).astype(np.float32)
        
        print ("Training index...")
        start_time = time.time()
        self._train_index(all_sampled_embs, self.trained_index_path)
        print ("Finish training (%ds)" % (time.time()-start_time))

    def _train_index(self, sampled_embs, trained_index_path):
        quantizer = faiss.IndexFlatIP(self.dimension)
        start_index = faiss.IndexIVFPQ(quantizer,
                                       self.dimension,
                                       self.ncentroids,
                                       self.n_subquantizers,
                                       self.code_size,
                                       faiss.METRIC_INNER_PRODUCT
                                       )
        start_index.nprobe = self.probe
        np.random.seed(1)

        if self.cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False

            # Train on GPU and back to CPU
            gpu_index.train(sampled_embs)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(sampled_embs)
        faiss.write_index(start_index, trained_index_path)

    def _add_keys(self, index_path, trained_index_path):
        index = faiss.read_index(trained_index_path)
        assert index.is_trained and index.ntotal == 0
        
        start_time = time.time()
        # NOTE: the shard id is a absolute id defined in the name
        for embed_path in self.embed_paths:
            filename = os.path.basename(embed_path)
            match = re.search(r"passages_(\d+)\.pkl", filename)
            shard_id = int(match.group(1))
                
            to_add = self.get_embs(shard_id=shard_id).copy()
            index.add(to_add)
            ids_toadd = [[shard_id, chunk_id] for chunk_id in range(len(to_add))]  #TODO: check len(to_add) is correct usage
            self.index_id_to_db_id.extend(ids_toadd)
            print ('Added %d / %d shards, (%d min)' % (shard_id+1, len(self.embed_paths), (time.time()-start_time)/60))
        
        faiss.write_index(index, index_path)
        with open(self.meta_file, 'wb') as fout:
            pickle.dump(self.index_id_to_db_id, fout)
        print ('Adding took {} s'.format(time.time() - start_time))
        return index
    # Old
    # def get_retrieved_passages(self, all_indices):
    #     all_passages = []
    #     for q_idx, query_indices in enumerate(all_indices):
    #         passages_per_query = []
    #         seen_keys = set()  # set of (filename, position) tuples

    #         # print(f"[DEBUG] Query {q_idx} top-{len(query_indices)} indices:")

    #         for i, index_id in enumerate(query_indices):
    #             position = int(self.position_array[index_id])
    #             filename_idx = int(self.filename_index_array[index_id])
    #             filename = self.filenames[filename_idx]
    #             file_path = os.path.join(self.passage_dir, filename)
    #             key = (filename, position)

    #             # Optional deduplication warning
    #             if key in seen_keys:
    #                 print(f"[WARN] index_id={index_id} (#{i}) points to duplicate passage at {filename} offset {position}")
    #             else:
    #                 seen_keys.add(key)

    #             with open(file_path, 'r') as f:
    #                 f.seek(position)
    #                 line = f.readline()

    #             passage = json.loads(line)
    #             source = filename.split('--')[0] if '--' in filename else 'unknown'

    #             passages_per_query.append({
    #                 "text": passage["text"],
    #                 "source": source,
    #                 "index_id": int(index_id),
    #                 "filename": filename,
    #                 "position": position
    #             })

    #             # print(f"[INFO] index_id={index_id} (#{i}) → {filename}:{position} → \"{passage['text'][:50]}\"")
    #             # print(f"[INFO] FAISS index size (index.ntotal): {self.index.ntotal}")


    #         all_passages.append(passages_per_query)
    #     return all_passages


    def read_text(self, index_id):
        pos = int(self.position_array[index_id])
        fname_idx = int(self.filename_index_array[index_id])
        fname = self.filenames[fname_idx]
        file_path = os.path.join(self.passage_dir, fname)
        try:
            with open(file_path, 'r') as f:
                f.seek(pos)
                line = f.readline()
                parsed = json.loads(line)
                return parsed.get("text", ""), parsed.get("id", None)

        except Exception as e:
            print(f"[WARN] Failed to read index_id={index_id} → {fname}:{pos} — {e}")
            return None, None

    # def get_retrieved_passages(self, all_indices):
    #     all_passages = []
    #     for query_indices in all_indices:
    #         passages_per_query = []
    #         for center_id in query_indices:
    #             idx = int(center_id)
    #             position = int(self.position_array[idx])
    #             filename_idx = int(self.filename_index_array[idx])
    #             filename = self.filenames[filename_idx]
    #             center_text = self.read_text(idx)
    #             passages_per_query.append({
    #                 "text": center_text.strip(),
    #                 "center_text": center_text,
    #                 "source": filename.split('--')[0],
    #                 "index_id": idx,
    #                 "filename": filename,
    #                 "position": position
    #             })
    #         all_passages.append(passages_per_query)
    #     return all_passages

    def get_retrieved_passages(self, indices, raw_query=None):
        """
        Given a flat list of FAISS index IDs (for a single query), return structured passage dicts.
        """
        passages = []
        for idx in indices:
            idx = int(idx)  # ensure it's an int
            position = int(self.position_array[idx])
            filename_idx = int(self.filename_index_array[idx])
            length = len(self.filenames)
            # print(f"File name idx is {filename_idx} and filename length is {length}")
            filename = self.filenames[filename_idx]
            center_text, passage_id = self.read_text(idx)
            if passage_id is None:
                print("[ERROR] Failed to parse passage id.")

            passage = {
                "passage_id": passage_id,
                "text": center_text.strip(),
                "center_text": center_text,
                "source": filename.split('--')[0],
                "index_id": idx,
                "filename": filename,
                "position": position
            }
            if raw_query is not None:
                passage["raw_query"] = raw_query
            passages.append(passage)

        return passages  # List[Dict]

    def expand_passage(self, idx, offset):
        position = int(self.position_array[idx])
        fname_idx = int(self.filename_index_array[idx])
        filename = self.filenames[fname_idx]

        center_text, _ = self.read_text(idx)
        text_blocks = [center_text]
        print(f"[INFO] Expansion offset is {offset}")

        for o in range(1, offset + 1):
            for sign in [-1, 1]:
                neighbor_id = idx + sign * o
                if 0 <= neighbor_id < len(self.position_array):
                    neighbor_text, _ = self.read_text(neighbor_id)
                    if neighbor_text:
                        if sign == -1:
                            text_blocks.insert(0, neighbor_text)
                        else:
                            text_blocks.append(neighbor_text)

        full_text = " ".join(text_blocks)
        result = {
            "text": full_text.strip(),
            "source": filename.split('--')[0],
            "index_id": int(idx),
            "filename": str(filename),
            "position": int(position)
        }

        if offset == 1:
            result["original_text"] = center_text.strip()

        print(f"[INFO] Return value with offset {offset} is: {result}")

        return result

    def _get_passage(self, index_id):
        try:
            position = int(self.position_array[index_id])
            filename_idx = int(self.filename_index_array[index_id])
            filename = self.filenames[filename_idx]
            file_path = os.path.join(self.passage_dir, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing file: {file_path}")

            with open(file_path, 'r') as f:
                f.seek(position)
                line = f.readline()

            passage = json.loads(line)
            source = filename.split('--')[0] if '--' in filename else 'unknown'

            return {
                "text": passage.get("text", ""),
                "source": source
            }
        except Exception as e:
            raise RuntimeError(f"Index {index_id} (pos={position}) failed: {e}")
        
    def is_redundant(self, existing_texts, new_text):
        return any(t in new_text or new_text in t for t in existing_texts)

    
    # Batched search from concurrent.futures import ThreadPoolExecutor
    def search(self, raw_query, query_embs, k, nprobe, expand_index_id=None, expand_offset=1, exact_rerank=False, diverse_rerank=False, lambda_val=0.5):
        if expand_index_id is not None:
            print(f"[EXPANSION MODE] Expanding index_id={expand_index_id} with offset={expand_offset}")
            result = self.expand_passage(expand_index_id, offset=expand_offset)
            return None, [[result]]

        if isinstance(raw_query, str):
            raw_queries = [raw_query]
        else:
            raw_queries = raw_query

        t0 = time.time()
        query_embs = query_embs.astype(np.float32)
        if nprobe is not None:
            self.index.nprobe = nprobe
            print(f"[IVFPQIndexer] nprobe dynamically set to {nprobe}")
        else:
            print("nprobe is set to None")

        # Optimize for speed: cap at 3200 for maximum speed
        MAX_K = 3200  # Capped for speed
        base_Ks = [1000]  # Always search with K=1000 for testing; slice later
        
        print(f"[DEBUG] Input k={k}, type={type(k)}")
        # Ensure k is an integer
        k = int(k)
        
        # Find starting K: smallest K that's >= k
        start_idx = len(base_Ks) - 1  # Default to last (MAX_K)
        for i, K in enumerate(base_Ks):
            if K >= k:
                start_idx = i
                break
        
        K_ranges = base_Ks[start_idx:]
        print(f"[DEBUG] Requested k={k}, starting from K={K_ranges[0] if K_ranges else 'None'}")
        print(f"[DEBUG] K_ranges: {K_ranges}")
        all_final_scores = []
        all_final_passages = []

        best_passages = None
        best_scores = None
        best_total = 0

        used_big_k = None
        for attempt in range(len(K_ranges)):
            K = K_ranges[attempt]
            print(f"[SEARCH] Attempt {attempt + 1}: K is {K}")
            used_big_k = K
            all_scores, all_indices = self.index.search(query_embs, K)


            # === Parallel get_retrieved_passages ===
            def retrieve_for_query(i):
                q_text = raw_queries[i]
                top_indices = all_indices[i]
                return self.get_retrieved_passages(top_indices, raw_query=q_text)

            with ThreadPoolExecutor() as executor:
                all_raw_passages = list(executor.map(retrieve_for_query, range(len(raw_queries))))

            all_batch_passages = []


            for i, raw_passages in enumerate(all_raw_passages):
                if exact_rerank:
                    try:
                        raw_passages = exact_rerank_topk(raw_passages, query_encoder)
                        print(f"[SEARCH] [QUERY {i}] Used Exact Rerank")
                    except Exception as e:
                        print(f"[SEARCH] [QUERY {i}] Exact Rerank failed: {e}")
                if diverse_rerank:
                    try:
                        raw_passages = diverse_rerank_topk(raw_passages, query_encoder, lambda_val=lambda_val)
                        print(f"[SEARCH] [QUERY {i}] Used Diverse Rerank")
                    except Exception as e:
                        print(f"[SEARCH] [QUERY {i}] Diverse Rerank failed: {e}")


                unique = []
                # seen_texts = set()  # Use set for O(1) lookup instead of list
                for passage in raw_passages:
                    #text = passage.get("text", "").strip()
                    
                    # Keep word limit consistent as requested
                    # min_words = 0 
                    # if len(text.split()) < min_words:
                    #     continue
                    
                    # Use set for faster redundancy check
                    # if text in seen_texts:
                    #     continue
                    
                    # # Removed keyword matching filter for speed and simplicity
                    
                    # seen_texts.add(text)
                    unique.append(passage)
                    if len(unique) >= k:
                        break

                all_batch_passages.append(unique)

            # Track best attempt so far
            total_found = sum(len(p) for p in all_batch_passages)
            if total_found > best_total:
                best_total = total_found
                best_passages = all_batch_passages
                best_scores = [s[:len(p)] for s, p in zip(all_scores, all_batch_passages)]

            if all(len(p) >= k for p in all_batch_passages):
                print(f"[SEARCH] Succeeded on attempt {attempt + 1}")
                all_final_passages = all_batch_passages
                # print(f"All final passages are: {all_final_passages}")
                all_final_scores = [s[:len(p)] for s, p in zip(all_scores, all_batch_passages)]
                break
            else:
                print(f"[SEARCH] Insufficient results on attempt {attempt + 1}")

        # Fallback to best available if we never reached k
        if not all_final_passages:
            print("[SEARCH] Falling back to best available results")
            all_final_passages = best_passages or []
            all_final_scores = best_scores or []
        elif any(len(p) < k for p in all_final_passages):
            print(f"[SEARCH] Completed with fewer than requested top-{k} results")

        t1 = time.time()

        # Log queries (moved from serve.py)
        try:
            if QUERY_LOGGER is not None:
                latency = round(t1 - t0, 4)
                cfg = {
                    'nprobe': nprobe,
                    'exact_search': bool(exact_rerank),
                    'diverse_search': bool(diverse_rerank),
                }
                if bool(diverse_rerank) and (lambda_val is not None):
                    try:
                        cfg['lambda'] = round(float(lambda_val), 2)
                    except Exception:
                        pass
                # raw_query may be str or list
                if isinstance(raw_query, list):
                    for q in raw_query:
                        QUERY_LOGGER.log(q, cfg, k, latency, expand_index_id, expand_offset, big_k=used_big_k)
                else:
                    QUERY_LOGGER.log(raw_query, cfg, k, latency, expand_index_id, expand_offset, big_k=used_big_k)
        except Exception:
            pass
        print(f"[SEARCH] Completed batch of {len(raw_queries)} in {t1 - t0:.2f}s")
        print(f"[DEBUG] Returning scores of shape: {len(all_final_scores)} x {len(all_final_scores[0]) if all_final_scores else 0}")
        print(f"[DEBUG] Returning passages of shape: {len(all_final_passages)} x {len(all_final_passages[0]) if all_final_passages else 0}")

        return all_final_scores, all_final_passages






        


def test_build_multi_shard_dpr_wiki():
    embed_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/embeddings/facebook/contriever-msmarco/dpr_wiki/1-shards'
    embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
    sample_train_size = 6000000
    projection_size = 768
    ncentroids = 4096
    n_subquantizers = 16
    code_size = 8
    formatted_index_name = f"index_ivf_pq_ip.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
    index_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/embeddings/facebook/contriever-msmarco/dpr_wiki/1-shards/index_ivf_pq/'
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, formatted_index_name)
    meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
    trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
    pos_map_save_path = os.path.join(index_dir, 'passage_pos_id_map.pkl')
    passage_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/passages/dpr_wiki/1-shards'
    index = IVFPQIndexer(
        embed_paths,
        index_path,
        meta_file,
        trained_index_path,
        passage_dir=passage_dir,
        pos_map_save_path=pos_map_save_path,
        sample_train_size=sample_train_size,
        dimension=projection_size,
        ncentroids=ncentroids,
        n_subquantizers=n_subquantizers,
        code_size=code_size,
        )


def test_build_single_shard_dpr_wiki():
    embed_dir = '/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards'
    embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
    sample_train_size = 6000000
    projection_size = 768
    ncentroids = 4096
    n_subquantizers = 16
    code_size = 8
    formatted_index_name = f"index_ivf_pq_ip.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
    index_dir = '/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards/index_ivf_pq/'
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, formatted_index_name)
    meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
    trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
    pos_map_save_path = os.path.join(index_dir, 'passage_pos_id_map.pkl')
    passage_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/passages/dpr_wiki/8-shards'
    index = IVFPQIndexer(
        embed_paths,
        index_path,
        meta_file,
        trained_index_path,
        passage_dir=passage_dir,
        pos_map_save_path=pos_map_save_path,
        sample_train_size=sample_train_size,
        dimension=projection_size,
        ncentroids=ncentroids,
        n_subquantizers=n_subquantizers,
        code_size=code_size,
        )

if __name__ == '__main__':
    test_build_multi_shard_dpr_wiki()