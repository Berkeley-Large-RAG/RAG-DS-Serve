import os
import sys
import json
import pickle as pkl
import logging
import time
import copy
import random
from tqdm import tqdm
import re
import pdb
import string
from collections import Counter

import torch


os.environ["TOKENIZERS_PARALLELISM"] = "true"


device = 'cuda' if torch.cuda.is_available()  else 'cpu'


def embed_queries(cfg, queries, model, tokenizer, model_name_or_path):
    if "sentence-transformers" in model_name_or_path or "e5" in model_name_or_path or "Qwen" in model_name_or_path:  # temp fix
        all_question = []
        for k, q in enumerate(queries):
            all_question.append(q)
        
        embeddings = model.encode(all_question, batch_size=min(128, cfg.per_gpu_batch_size))  # sentence-transformer has extra memory overhead and can only support a smaller batch size
    
    else:
        model.eval()
        embeddings, batch_question = [], []
        with torch.no_grad():

            for k, q in tqdm(enumerate(queries)):
                batch_question.append(q)

                if len(batch_question) == cfg.per_gpu_batch_size or k == len(queries) - 1:
                    
                    if "drama" in model_name_or_path:
                        output = model.encode_queries(batch_question, batch_question, dim=768)  # TODO: change this to align with index.projection_size
                    elif "ReasonIR" in model_name_or_path or "GRIT" in model_name_or_path:
                        output = model.encode(batch_question, instruction="", batch_size=cfg.per_gpu_batch_size)
                        output = torch.tensor(output, device='cpu')
                    else:
                        encoded_batch = tokenizer.batch_encode_plus(
                            batch_question,
                            return_tensors="pt",
                            max_length=cfg.question_maxlength,
                            padding=True,
                            truncation=True,
                        )

                        encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                        output = model(**encoded_batch)
                        if "contriever" not in model_name_or_path:
                            output = output.last_hidden_state[:, 0, :]
                    
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0).numpy()
    
    print(f"Questions embeddings shape: {embeddings.shape}")

    return embeddings
