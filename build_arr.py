import os
import json
import numpy as np
from tqdm import tqdm

INPUT_DIR = "/home/ubuntu/Jinjian/massive-serve/built_index_full/passages"
POSITION_ARRAY_PATH = "position_array.npy"
FILENAME_INDEX_ARRAY_PATH = "filename_index_array.npy"
FILENAME_LIST_PATH = "filename_list.npy"

deprioritized_domains = ["massiveds-rpj_arxiv", "massiveds-rpj_github", "massiveds-rpj_book", "lb_full"]
deprioritized_domains_index = {domain: i + 1 for i, domain in enumerate(deprioritized_domains)}

def sort_func(filename):
    import re
    domain = filename.split('--', 1)[0]
    m = re.search(r'--passages(\d+)_(\d+)\.pkl$', filename)
    if m:
        rank = int(m.group(1))
        shard_idx = int(m.group(2))
        deprioritized = deprioritized_domains_index.get(domain, 0)
        return (deprioritized, domain, rank, shard_idx)

    m = re.search(r'passages_(\d+)-of-(\d+)', filename)
    if m:
        rank = int(m.group(1))
        try:
            shard_idx = int(filename.split('-of-')[0].split('-')[-1])
        except Exception:
            shard_idx = 0
        deprioritized = deprioritized_domains_index.get(domain, 0)
        return (deprioritized, domain, rank, shard_idx)

    deprioritized = deprioritized_domains_index.get(domain, 0)
    return (deprioritized, domain, float('inf'), float('inf'))
def sort_jsonl_files(jsonl_files):
    return sorted(jsonl_files, key=sort_func)

def check_jsonl_file_validity(file_path, num_lines_to_check=3):
    try:
        with open(file_path, 'r') as f:
            for _ in range(num_lines_to_check):
                line = f.readline()
                if not line:
                    break
                json.loads(line)
        return True
    except Exception as e:
        print(f"❌ Corrupted or malformed file: {file_path} | Error: {e}")
        return False

def build_dual_array_index_mapping():
    print(f"📂 Scanning sorted .jsonl files from: {INPUT_DIR}")
    filenames = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    sorted_filenames = sort_jsonl_files(filenames)

    position_array = []
    filename_index_array = []
    filename_list = []

    valid_count, corrupted_count = 0, 0

    for idx, fname in enumerate(tqdm(sorted_filenames, desc="📍 Processing files")):
        fpath = os.path.join(INPUT_DIR, fname)

        if not check_jsonl_file_validity(fpath):
            corrupted_count += 1
            continue

        valid_count += 1
        filename_list.append(fname)

        with open(fpath, "r") as f:
            position = f.tell()
            line = f.readline()
            while line:
                position_array.append(position)
                filename_index_array.append(idx)
                position = f.tell()
                line = f.readline()

    # Save arrays
    print(f"💾 Saving arrays with {len(position_array)} entries")
    np.save(POSITION_ARRAY_PATH, np.array(position_array, dtype=np.int64))
    np.save(FILENAME_INDEX_ARRAY_PATH, np.array(filename_index_array, dtype=np.int32))
    np.save(FILENAME_LIST_PATH, np.array(filename_list, dtype=object))  # list of strings

    print("✅ Mapping complete.")
    print(f"✅ {valid_count} valid files processed.")
    if corrupted_count > 0:
        print(f"⚠️  {corrupted_count} corrupted files skipped.")

if __name__ == "__main__":
    build_dual_array_index_mapping()
