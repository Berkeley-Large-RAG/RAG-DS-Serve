import os
import json
import hashlib  # [disk-vote] query hashing for compact storage
import sqlite3  # [disk-vote-sqlite] efficient lookup index
import traceback
import datetime
import socket
import getpass
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue
from flask import send_from_directory
try:
    import psutil as _psutil
except Exception:
    _psutil = None
import time
try:
    from colorama import Fore, Style
except Exception:
    class _DummyStyle:
        RESET_ALL = ""
    class _DummyFore:
        CYAN = ""; GREEN = ""; YELLOW = ""; WHITE = ""
    Fore = _DummyFore()
    Style = _DummyStyle()
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gritlm import GritLM


startup_start = time.time()
query_tokenizer  = None
query_encoder = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")
query_encoder.eval()
query_encoder = query_encoder.to(device)

# Function to measure memory usage (no-op if psutil unavailable)
def print_mem_use(label=""):
    try:
        if _psutil is None:
            return
        process = _psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_mb = mem_bytes / 1024 / 1024
        print(f"[{label}] Memory usage: {mem_mb:.2f} MB")
    except Exception:
        pass

from massive_serve.api.api_index import get_datastore
try:
    # Inject logger into ivf_pq after QUERY_LOGGER is defined
    from massive_serve.src.indicies import ivf_pq as _ivf
except Exception:
    _ivf = None

# ANSI color codes
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
WHITE = '\033[37m'
RESET = '\033[0m'


class DatastoreConfig():
    def __init__(self,):
        super().__init__()
        self.data_root = os.environ.get('DATASTORE_PATH', os.path.expanduser('~'))
        self.domain_name = os.environ.get('MASSIVE_SERVE_DOMAIN_NAME', 'demo')
        self.config = json.load(open(os.path.join(self.data_root, self.domain_name, 'config.json')))
        assert self.config['domain_name'] == self.domain_name, f"Domain name in config.json ({self.config['domain_name']}) does not match the domain name in the environment variable ({self.domain_name})"
        self.query_encoder = self.config['query_encoder']
        self.query_tokenizer = self.config['query_tokenizer']
        self.index_type = self.config['index_type']
        self.per_gpu_batch_size = self.config['per_gpu_batch_size']
        self.question_maxlength = self.config['question_maxlength']
        self.nprobe = self.config['nprobe']

ds_cfg = DatastoreConfig()
        

app = Flask(__name__)
CORS(app)


# ============================ [disk-vote] BEGIN additions ============================
# Persist passage relevance votes to disk for later analysis and across restarts.
# One JSON line per vote: { ts, query_hash, query_norm, passage_id, relevant, raw_query }

class DiskVoteStore:
    """Append-only JSONL vote store with bounded tail scans for lookup.

    - Writes are serialized via a process-level lock
    - get_vote scans from file end up to a fixed number of lines to find latest
    """

    def __init__(self, base_dir: str, file_name: str = "votes.jsonl", tail_scan_limit: int = 20000) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.file_path = os.path.join(self.base_dir, file_name)
        self.tail_scan_limit = int(tail_scan_limit)
        self._lock = threading.Lock()
        # [disk-vote-sqlite] initialize sqlite sidecar for fast lookups
        self.db_path = os.path.join(self.base_dir, "votes.sqlite3")
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("PRAGMA synchronous=NORMAL;")
        self._create_tables()

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            return ""
        return " ".join(query.strip().lower().split())

    @classmethod
    def _hash_query(cls, query: str) -> str:
        norm = cls._normalize_query(query)
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()

    @staticmethod
    def _canonicalize_ctx(ctx: dict | None) -> dict:
        if not isinstance(ctx, dict):
            return {}
        out = {}
        if "nprobe" in ctx and ctx["nprobe"] is not None:
            try:
                out["nprobe"] = int(ctx["nprobe"])  # ensure int
            except Exception:
                pass
        if "exact_search" in ctx:
            out["exact_search"] = bool(ctx["exact_search"])  # required boolean
        if "diverse_search" in ctx:
            out["diverse_search"] = bool(ctx["diverse_search"])  # required boolean
        # Only persist lambda if diverse_search is true
        if out.get("diverse_search") is True and ("lambda" in ctx) and (ctx["lambda"] is not None):
            try:
                out["lambda"] = round(float(ctx["lambda"]), 2)  # 2dp
            except Exception:
                pass
        return out

    @classmethod
    def _hash_ctx(cls, ctx: dict | None) -> str:
        canon = cls._canonicalize_ctx(ctx)
        # stable JSON string for hashing
        canon_str = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(canon_str.encode("utf-8")).hexdigest()

    def _create_tables(self) -> None:
        # Minimal normalized schema
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
              query_hash TEXT PRIMARY KEY,
              query_norm TEXT
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS contexts (
              ctx_hash TEXT PRIMARY KEY,
              nprobe INTEGER,
              exact_search INTEGER,
              diverse_search INTEGER,
              lambda REAL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS votes (
              query_hash TEXT NOT NULL,
              ctx_hash TEXT NOT NULL,
              passage_id TEXT NOT NULL,
              relevant INTEGER NOT NULL,
              ts INTEGER NOT NULL,
              PRIMARY KEY (query_hash, ctx_hash, passage_id)
            )
            """
        )
        self._db.commit()

    def record_vote(self, query: str, passage_id: str, relevant: bool, ctx: dict | None = None) -> None:
        payload = {
            # [disk-vote] compact: integer seconds
            "ts": int(time.time()),
            "query_hash": self._hash_query(query),
            "query_norm": self._normalize_query(query),
            "passage_id": str(passage_id),
            "relevant": bool(relevant),
        }
        # [disk-vote] optional compact search config (nprobe, exact_search, diverse_search, lambda)
        compact_ctx = self._canonicalize_ctx(ctx)
        if compact_ctx:
            payload["config"] = compact_ctx
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            # [disk-vote-sqlite] upsert normalized rows for efficient lookup
            qh = payload["query_hash"]
            cn = payload["query_norm"]
            ch = self._hash_ctx(compact_ctx)
            ts = payload["ts"]
            rel = 1 if bool(payload["relevant"]) else 0
            pid = payload["passage_id"]
            # queries map
            self._db.execute(
                "INSERT OR IGNORE INTO queries(query_hash, query_norm) VALUES (?, ?)",
                (qh, cn),
            )
            # contexts map
            if compact_ctx:
                self._db.execute(
                    "INSERT OR IGNORE INTO contexts(ctx_hash, nprobe, exact_search, diverse_search, lambda) VALUES (?, ?, ?, ?, ?)",
                    (
                        ch,
                        compact_ctx.get("nprobe"),
                        1 if compact_ctx.get("exact_search") else 0 if "exact_search" in compact_ctx else None,
                        1 if compact_ctx.get("diverse_search") else 0 if "diverse_search" in compact_ctx else None,
                        compact_ctx.get("lambda"),
                    ),
                )
            else:
                # ensure there is at least a row for empty ctx
                self._db.execute(
                    "INSERT OR IGNORE INTO contexts(ctx_hash, nprobe, exact_search, diverse_search, lambda) VALUES (?, NULL, NULL, NULL, NULL)",
                    (ch,),
                )
            # votes upsert
            self._db.execute(
                "INSERT OR REPLACE INTO votes(query_hash, ctx_hash, passage_id, relevant, ts) VALUES (?, ?, ?, ?, ?)",
                (qh, ch, pid, rel, ts),
            )
            self._db.commit()

    def get_vote(self, query: str, passage_id: str):
        """Return latest vote True/False for (query, passage_id) or None if unseen."""
        qh = self._hash_query(query)
        pid = str(passage_id)
        try:
            if not os.path.exists(self.file_path):
                return None
            # Read tail blocks backwards without loading entire file
            with open(self.file_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                buffer = bytearray()
                pos = f.tell()
                lines_seen = 0
                while pos > 0 and lines_seen < self.tail_scan_limit:
                    step = min(4096, pos)
                    pos -= step
                    f.seek(pos)
                    chunk = f.read(step)
                    buffer[:0] = chunk
                    while True:
                        nl = buffer.rfind(b"\n")
                        if nl == -1:
                            break
                        line = buffer[nl+1:]
                        del buffer[nl:]
                        if not line:
                            continue
                        try:
                            rec = json.loads(line.decode("utf-8"))
                            if rec.get("query_hash") == qh and str(rec.get("passage_id")) == pid:
                                return bool(rec.get("relevant"))
                        except Exception:
                            pass
                        lines_seen += 1
                # Check remaining buffer (start of file)
                if buffer:
                    try:
                        rec = json.loads(buffer.decode("utf-8"))
                        if rec.get("query_hash") == qh and str(rec.get("passage_id")) == pid:
                            return bool(rec.get("relevant"))
                    except Exception:
                        pass
            return None
        except Exception:
            return None


# Instantiate vote store under VM root folder (override with VOTES_DIR if set)
_votes_dir = os.environ.get('VOTES_DIR', '/home/ubuntu/votes')
vote_store = DiskVoteStore(_votes_dir)
# ============================ [disk-vote] END additions ============================

# ============================ [query-log] BEGIN additions ============================
class QueryLogger:
    """Thread-safe JSONL logger for every query with canonicalized config."""

    def __init__(self, base_dir: str, file_name: str = "queries.jsonl") -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.file_path = os.path.join(self.base_dir, file_name)
        self._lock = threading.Lock()

    def log(self, query: str, canon_cfg: dict, n_docs: int, latency_s: float, expand_index_id=None, expand_offset=1, big_k=None) -> None:
        rec = {
            "ts": int(time.time()),
            "query_hash": DiskVoteStore._hash_query(query),
            "query_norm": DiskVoteStore._normalize_query(query),
            "config": canon_cfg,
            "n_docs": int(n_docs),
            "latency": float(latency_s),
            "expand_index_id": expand_index_id,
            "expand_offset": expand_offset,
        }
        if big_k is not None:
            try:
                rec["K"] = int(big_k)
            except Exception:
                pass
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())


_query_log_dir = os.environ.get('QUERY_LOG_DIR', '/home/ubuntu/query_logs')
QUERY_LOGGER = QueryLogger(_query_log_dir)
try:
    if _ivf and hasattr(_ivf, 'set_query_logger'):
        _ivf.set_query_logger(QUERY_LOGGER)
except Exception:
    pass
# ============================ [query-log] END additions ============================

 
class Item:
    def __init__(self, query=None, query_embed=None, domains=ds_cfg.domain_name, n_docs=1, nprobe=None, expand_index_id=None, expand_offset=None, exact_rerank=False, use_diverse=False, lambda_val=None) -> None:
        self.query = query
        self.query_embed = query_embed
        self.domains = domains
        self.n_docs = n_docs
        self.nprobe = nprobe
        self.searched_results = None
        self.expand_index_id = expand_index_id
        self.expand_offset = expand_offset
        self.exact_rerank = exact_rerank
        self.use_diverse = use_diverse
        self.lambda_val = lambda_val
    
    def get_dict(self,):
        dict_item = {
            'query': self.query,
            'query_embed': self.query_embed,
            'domains': self.domains,
            'n_docs': self.n_docs,
            'nprobe': self.nprobe,
            'searched_results': self.searched_results,
            'expand_index_id' : self.expand_index_id,
            'expand_offset' : self.expand_offset,
            'exact_rerank' : self.exact_rerank,
        }
        return dict_item


class SearchQueue:
    def __init__(self, log_queries=False):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.current_search = None
        print_mem_use("Before get datastore")
        self.datastore = get_datastore(ds_cfg)
        print_mem_use("After get datastore")
        self.log_queries = log_queries
        self.query_log = 'cached_queries.jsonl'
    
    def search(self, item):
        with self.lock:
            if self.current_search is None:
                self.current_search = item
                if item.nprobe is not None:
                    self.datastore.index.nprobe = item.nprobe
                results = self.datastore.search(item.query, item.n_docs, item.nprobe, item.expand_index_id, item.expand_offset, item.exact_rerank, item.use_diverse, item.lambda_val)
                self.current_search = None
                return results
            else:
                future = threading.Event()
                self.queue.put((item, future))
                future.wait()
                return item.searched_results
    
    def process_queue(self):
        while True:
            item, future = self.queue.get()
            with self.lock:
                self.current_search = item
                item.searched_results = self.datastore.search(item)
                self.current_search = None
            future.set()
            self.queue.task_done()

search_queue = SearchQueue()
threading.Thread(target=search_queue.process_queue, daemon=True).start()

@app.route('/search', methods=['POST'])
def search():
    try:
        if "queries" in request.json:
            print(f"[Batched Search] Received {len(request.json['queries'])} queries.")
            query_input = request.json["queries"]
        else:
            print("Processing a single query at once. ")
            query_input=request.json['query']
        # Build canonical config for caching key
        request_cfg = {
            'nprobe': request.json.get('nprobe', None),
            'exact_search': request.json.get('exact_search', False),
            'diverse_search': request.json.get('diverse_search', False),
            'lambda': request.json.get('lambda', 0.5),
        }
        canon_cfg = DiskVoteStore._canonicalize_ctx(request_cfg)
        item = Item(
            query=query_input,
            n_docs=request.json.get('n_docs', 1),
            nprobe=request.json.get('nprobe', None),
            expand_index_id = request.json.get('expand_index_id'),
            expand_offset = request.json.get('expand_offset', 1),
            domains=ds_cfg.domain_name,
            # exact_rerank = request.json.get('use_rerank', False),  # ORIGINAL - commented out
            # use_diverse=request.json.get('use_diverse', False),  # ORIGINAL - commented out
            exact_rerank = request.json.get('exact_search', False),
            use_diverse=request.json.get('diverse_search', False),
            lambda_val=request.json.get('lambda', 0.5),
        )

        # Perform the search synchronously with 600s timeout
        timer = threading.Timer(600.0, lambda: (_ for _ in ()).throw(TimeoutError('Search timed out after 600 seconds')))
        timer.start()
        try:
            start_time = time.time()
            results = search_queue.search(item)
            timer.cancel()



            def convert_ndarrays(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_ndarrays(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: convert_ndarrays(v) for k, v in obj.items()}
                else:
                    return obj

            results = convert_ndarrays(results)
            # Query logging moved into ivf_pq.search

            return jsonify({
                "message": f"Search completed for '{item.query}' from {item.domains}",
                "query": query_input,
                "n_docs": item.n_docs,
                "nprobe": item.nprobe,
                "expand_index_id" : request.json.get('expand_index_id'),
                "expand_offset" : request.json.get('expand_offset', 1),
                "results": results,
            }), 200
        except TimeoutError as e:
            timer.cancel()
            return jsonify({
                "message": str(e),
                "query": item.query,
                "error": "timeout"
            }), 408
    except Exception as e:
        tb_lines = traceback.format_exception(e.__class__, e, e.__traceback__)
        error_message = f"An error occurred: {str(e)}\n{''.join(tb_lines)}"
        return jsonify({"message": error_message}), 500
    
@app.route('/vote', methods=['POST'])
def vote():
    """[disk-vote] Record a relevance vote for a passage tied to a specific query.

    Body: { query: str, passage_id: str|int, relevant: bool, config?: { ... } }
    Always pass the original passage_id (even if UI shows expanded context).
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = data.get('query')
        passage_id = data.get('passage_id')
        relevant = data.get('relevant')
        ctx = data.get('config') 
        if not query or passage_id is None or relevant is None:
            return jsonify({"status": "error", "message": "Missing query, passage_id, or relevant"}), 400
        vote_store.record_vote(query, str(passage_id), bool(relevant), ctx=ctx if isinstance(ctx, dict) else None)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/vote/peek', methods=['POST'])
def vote_peek():
    """[disk-vote] Optional helper to retrieve last vote for a (query, passage_id)."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = data.get('query')
        passage_id = data.get('passage_id')
        if not query or passage_id is None:
            return jsonify({"status": "error", "message": "Missing query or passage_id"}), 400
        val = vote_store.get_vote(query, str(passage_id))
        return jsonify({"status": "ok", "vote": val}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/current_search')
def current_search():
    with search_queue.lock:
        current = search_queue.current_search
        if current:
            return jsonify({
                "current_search": current.query,
                "domains": current.domains,
                "n_docs": current.n_docs,
            }), 200
        else:
            return jsonify({"message": "No search currently in progress"}), 200

@app.route('/queue_size')
def queue_size():
    size = search_queue.queue.qsize()
    return jsonify({"queue_size": size}), 200

@app.route("/")
def home():
    return jsonify("Hello! What you are looking for?")

@app.route('/ui')
def serve_ui():
    return send_from_directory('.', 'backup.html')

def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.

@app.route('/<path:filename>')
def serve_static_file(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(script_dir, filename)

def main():
    #port = find_free_port()
    port = int(os.environ.get('MASSIVE_SERVE_PORT', 30888))
    server_id = socket.gethostname()
    domain_name = ds_cfg.domain_name
    serve_info = {'server_id': server_id, 'port': port}
    username = os.environ.get('USER') or os.environ.get('USERNAME') or getpass.getuser()
    endpoint = f'{username}@{server_id}:{port}/search'  # use actual username
    

    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    WHITE = Fore.WHITE
    RESET = Style.RESET_ALL
    box_width = 63  

    def pad(label, value):
        total_padding = box_width - len(label)
        return label + value.ljust(total_padding)

    banner = f"""
    {CYAN}╔{'═' * box_width}╗
    ║{GREEN}{'MASSIVE SERVE SERVER'.center(box_width)}{CYAN}║
    ╠{'═' * box_width}╣
    ║{YELLOW}{pad(" Domain:   ", WHITE + domain_name)}{CYAN}║
    ║{YELLOW}{pad(" Server:   ", WHITE + server_id)}{CYAN}║
    ║{YELLOW}{pad(" Port:     ", WHITE + str(port))}{CYAN}║
    ║{YELLOW}{pad(" Endpoint: ", WHITE + endpoint)}{CYAN}║
    ╚{'═' * box_width}╝{RESET}
    """
    print(banner)
    
    # Print test request with colors
    test_request = f"""
{GREEN}Test your server with this curl command:\n{RESET}
{CYAN}curl -X POST {endpoint} -H "Content-Type: application/json" -d '{YELLOW}{{"query": "Tell me more about the stories of Einstein.", "n_docs": 1, "domains": "{domain_name}"}}{CYAN}'{RESET}
"""
    print(test_request)
    print(f"Deployment completed in {time.time() - startup_start:.2f} seconds")

    app.run(host='0.0.0.0', port=port)
    # app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)



if __name__ == '__main__':
    main()