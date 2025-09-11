import os
import re
import logging
import numpy as np
import torch

from .flat import FlatIndexer
from .ivf_flat import IVFFlatIndexer
from .ivf_pq import IVFPQIndexer


class Indexer(object):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.index_type = cfg.index_type
        
        data_root = os.path.expanduser(self.cfg.data_root)
        
        index_dir = os.path.join(data_root, cfg.domain_name, 'index')
        index_files = [f for f in os.listdir(index_dir) if f.endswith('.faiss')]
        if len(index_files) != 1:
            raise ValueError(f"Expected exactly one .faiss file in {index_dir}, found {len(index_files)}")
        index_path = os.path.join(index_dir, index_files[0])
        #meta_file = os.path.join(index_dir, index_files[0].replace('.faiss', '.faiss.meta'))
        meta_file = "/home/ubuntu/Jinjian/retrieval-scaling/built_index/index/index_id_to_db_id.npy"
        #assert os.path.exists(index_path) and os.path.exists(meta_file)
        
        passage_dir = os.path.join(data_root, cfg.domain_name, 'passages')
        embedding_paths = None  # assume index is already built
        trained_index_path = None
        pos_map_save_path = os.path.join(data_root, cfg.domain_name, 'id_to_passage.jsonl')
        
        if self.index_type == "Flat":
            self.datastore = FlatIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                passage_dir=passage_dir,
                pos_map_save_path=pos_map_save_path,
                dimension=None,
            )
        elif self.index_type == "IVFFlat":
            self.datastore = IVFFlatIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                trained_index_path=trained_index_path,
                passage_dir=passage_dir,
                pos_map_save_path=pos_map_save_path,
                sample_train_size=None,
                prev_index_path=None,
                dimension=None,
                ncentroids=None,
                probe=self.cfg.nprobe,
            )
        elif self.index_type == "IVFPQ":
            self.datastore = IVFPQIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                trained_index_path=trained_index_path,
                passage_dir=passage_dir,
                pos_map_save_path=pos_map_save_path,
                sample_train_size=None,
                prev_index_path=None,
                dimension=None,
                ncentroids=None,
                probe=self.cfg.nprobe,
                n_subquantizers=None,
                code_size=None,
            )
        else:
            raise NotImplementedError
        
        
    def search(self, query_embs, k=5):
        all_scores, all_passages, db_ids = self.datastore.search(query_embs, k)
        return all_scores, all_passages, db_ids
    
