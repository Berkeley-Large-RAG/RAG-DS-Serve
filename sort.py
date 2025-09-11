import os

INPUT_DIR = "/home/ubuntu/Jinjian/retrieval-scaling/built_index_full/passages"
OUTPUT_PATH = "./Jinjian_sorted_filenames.txt"
deprioritized_domains = ["massiveds-rpj_arxiv", "massiveds-rpj_github", "massiveds-rpj_book", "lb_full"]
deprioritized_domains_index = {domain: i + 1 for i, domain in enumerate(deprioritized_domains)}

def sort_func(filename):
    domain = filename.split('raw_passages')[0].split('--')[0]
    rank = int(filename.split('passages_')[-1].split("-")[0])
    shard_idx = int(filename.split("-of-")[0].split("-")[-1])
    deprioritized = deprioritized_domains_index.get(domain, 0)
    return (deprioritized, domain, rank, shard_idx)

def sort_jsonl_files(jsonl_files):
    return sorted(jsonl_files, key=sort_func)


if __name__ == "__main__":
    filenames = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    sorted_filenames = sort_jsonl_files(filenames)
    with open(OUTPUT_PATH, "w") as f:
        for fname in sorted_filenames:
            f.write(fname + "\n")
    print(f"!!! Files sorted in this order: {sorted_filenames}")