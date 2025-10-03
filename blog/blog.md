---
layout: page
title: DS Serve
---

<h2 align="center">🚀 <b>DS SERVE: A Framework for Efficient and Scalable Neural Retrieval</b></h2>

<p align="center">Jinjian Liu<sup>1*</sup>, Yichuan Wang<sup>1*</sup>, <a href="https://github.com/lyuxinxi">Xinxi Lyu</a><sup>2</sup>, Rulin Shao<sup>3</sup>, Joseph E. Gonzalez<sup>1</sup>, Matei Zaharia<sup>1</sup>, Sewon Min<sup>1</sup></p>
<p align="center"><sup>1</sup>University of California, Berkeley <sup>2</sup>University of Illinois Urbana–Champaign <sup>3</sup>University of Washington</p>

<p align="center">[<a href="https://tinyurl.com/compact-ds-dive">Web Interface</a>] [<a href="../API_DOCUMENTATION.md">API Endpoint</a>] [<a href="../README.md">Server & CLI</a>] [<a href="../VOTES_DOCUMENTATION.md">Votes Storage</a>] [<a href="https://github.com/Berkeley-Large-RAG/RAG-DS-Serve">Code</a>] [<a href="https://arxiv.org/abs/2507.01297">Paper</a>]</p>

**[✨NEW]** DiskANN integration: ~1000 QPS at 500B-token scale with low RAM.

**[✨NEW]** Exact + Diverse modes: tune accuracy–diversity–latency on demand.

---
<br/>

We introduce DS Serve, a framework that transforms large-scale text corpus into a high-performance neural retrieval system, aka search engines without daunting hardware requirements. DS Serve achieves low latency with modest memory overhead, and it also supports inference time tradeoff across accuracy, diversity, and latency.

<p align="center">
  <img src="Figure%201.png" style="width: 80%;" />
</p>

**Key Performance**
- DiskANN-backed serving: ~1000 QPS at 500B-token scale with low RAM footprint
- FAISS IVFPQ backbone: ~200 ms inference at ~100 GB memory overhead
- Toggle accuracy/diversity with Exact and Diverse search options

<p align="center">
  <img src="table.png" style="width: 80%;" />
  <br/>
  <i>Figure: Summary table of evaluation highlights.</i>
</p>

---
<br/>

<h2 align="center">What can I use DS Serve for?</h2>

We use DS Serve for fast, controllable retrieval in RAG and search applications.

It’s actively useful for:
1. **Data attribution & curation**: semantic search beyond keyword overlap; complements tools like OLMoTrace
2. **Training search agents**: fully controllable latency–accuracy tradeoffs without external rate limits
3. **Advancing search quality**: excels on longer, complex queries where keyword engines falter

Beyond these, DS Serve helps with:
1. **Understanding corpora** and building tailored subsets
2. **Reducing redundancy** via diversity-aware reranking (MMR)
3. **Collecting realistic benchmarks** through built-in voting

---
<br/>

<h2 align="center">Why Exact and Diverse Modes?</h2>

Because IVFPQ inevitably sacrifices accuracy, we introduce Exact Search as an optional reranking mode. On top of ANN, we employ GritLM to compute exact similarities between queries and passage embeddings, which are then used to rerank top-k passages. As shown in our evaluation results (Table 1), Exact Search effectively enhances accuracy across all five tasks. On a cold start, embedding the results can be slow, so we've built an embedding cache to allow ~1000ms latency on similar queries in Exact Search.

<p align="center">
  <img src="parameter panel.png" style="width: 70%;" />
  <br/>
  <i>Figure: Control panel with tunable parameters and tooltips.</i>
</p>

Search results often suffer from information overlap, like nearly identical text chunks, so we offer a Diverse Search option to improve overall coverage of the results. To do this, we apply maximal marginal relevance (MMR) on candidates returned by ANN to penalize redundant information. In our use cases, we find Diverse Search substantially improves user experience by eliminating redundant texts.

For queries that are less common or rely on very recent knowledge — for example “Jensen Huang” — the datastore may only contain a handful of truly relevant passages, due to its frozen state. In these situations, Exact Search performs well because it prioritizes those relevant results at the top. By contrast, Diverse Search doesn’t necessarily improve performance since it risks surfacing less accurate passages when there aren’t enough strong candidates to begin with.

During search, DS SERVE initially fetches a very large pool of K candidates (K = 1000), so it always easily fills the top-k (e.g. k = 5) passage list. On the very rare occasion that retrieval does fail, a clear alert message pops up to give informative suggestions for the user.

<p align="center">
  <img src="failure mode.png" style="width: 80%;" />
  <br/>
  <i>Figure: Example failure mode and user guidance.</i>
</p>

---
<br/>

<h2 align="center">How DS Serve Works</h2>

- **Backbone**: FAISS IVFPQ index for billion-scale ANN search
- **On-disk scale**: DiskANN integration for high QPS with low RAM
- **Passage lookup**: position arrays map FAISS ids back to text shards
- **Reranking**: optional Exact (accuracy) and Diverse (coverage) modes
- **Controls**: tunable nprobe, exact/diverse toggles, λ for MMR, etc.

---
<br/>

## Citation

If you find DS Serve useful, please kindly cite:
```bibtex
@article{lyu2025compactds,
  title={Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks},
  author={Xinxi Lyu and Michael Duan and Rulin Shao and Pang Wei Koh and Sewon Min},
  journal={arXiv preprint arXiv:2507.01297},
  year={2025}
}
```
