import os
import json
import csv
import pickle
import gzip
import logging
import pdb

from massive_serve.src.indicies.index_utils import get_passage_pos_ids


############################## Training ##############################
def fast_load_jsonl_shard(passages_dir, shard_index, return_all_passages=True):
    """
    This function is designed to handle large datasets by only loading the specific portion of data (shard) that 
    corresponds to the given shard index.

    Shards are determined by dividing the total size of all files in the directory evenly by `num_shards`. 
    This function reads only the data portion of the `shard_index` shard, chunks the text from each line 
    based on `chunk_sz`, and appends each chunk to a list with an incremental ID.
    """
    use_passage_pos_id_map = True
    
    if not return_all_passages:
        assert use_passage_pos_id_map, f"You must set `use_passage_pos_id_map=True` to enable efficient passage loading!"

    # Check the existance of processed data
    passage_shard_save_path = os.path.join(passages_dir, f'raw_passages-{shard_index}-of-{num_shards}.jsonl')
    pos_map_save_path = os.path.join(passages_dir, 'passage_pos_id_map.pkl')

    if not return_all_passages:
        with open(pos_map_save_path, 'rb') as f:
            passage_pos_ids = pickle.load(f)
        return  passage_pos_ids
    
    elif os.path.exists(passage_shard_save_path):
        passages = []
        with open(passage_shard_save_path, 'r') as fin:
            for line in fin:
                passages.append(json.loads(line))
        return passages
