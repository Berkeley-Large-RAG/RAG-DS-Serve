---
layout: page
title: DS Serve
---

<style>
@import url('{{ "assets/infini-gram.css" | relative_url }}');
p { font-size: 16px; margin: 12px 0; }
.small-note { font-size: 13px; color: #666; margin-top: 2px; }
/* Constrain page width for readability */
.page-content { max-width: 1100px; margin: 0 auto; padding: 0 16px; }
/* Compact callout note for Overview */
.callout-note {
  font-size: 13px;
  color: #6b7280;            /* gray-500 */
  background: #fcfcfd;       /* very light */
  border-left: 3px solid #e5e7eb;
  padding: 6px 10px;
  margin: 8px 0 0;
  border-radius: 4px;
  font-style: normal;
}
.callout-note a { font-weight: 600; }
/* Compact superscript anchor for inline note references */
.note-ref {
  font-size: 12px;
  line-height: 0;
  vertical-align: super;
  margin-left: 2px;
}
.note-ref a {
  color: #6b7280;
  text-decoration: none;
}
.note-ref a:hover { text-decoration: underline; }
/* Inline note reference for balanced text flow */
.note-inline {
  font-size: 12px;
  color: #6b7280;
  margin-left: 4px;
  white-space: nowrap;
  text-decoration: none;
}
.note-inline:hover { text-decoration: underline; }
/* Superscript note marker placed tight to the word */
.note-sup {
  font-size: 0.55em;
  line-height: 0;
  vertical-align: super;
}
.note-sup a {
  color: #9ca3af;           /* gray-400 */
  text-decoration: none;
}
.note-sup a:hover { color: #6b7280; text-decoration: underline; }
/* Overview highlight box */
.overview-box {
  background: #ffffb3;
  border-left: 4px solid #facc15;
  padding: 12px 16px;
  margin: 12px 0 16px;
  border-radius: 6px;
  color: #1f2937;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
/* Make details summaries match paragraph sizing */
details > summary {
  font-size: 16px;
  margin: 6px 0;
  cursor: pointer;
}
/* Responsive helpers */
.affiliations {
  font-size: 13px;
  line-height: 1.2;
  margin: 0 auto 4px;
  max-width: 100%;
  text-align: center;
}
.affiliations span {
  white-space: normal;
}
.affiliations sup {
  margin-right: 2px;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
/* Hide theme-injected page title on homepage to avoid duplicate 'DS Serve' */
.post-title, .page-title { display: none; }
/* Hide the Minima footer on the homepage to avoid duplicate site title */
.site-footer { display: none !important; }
/* Center the top navigation and enlarge links on the homepage */
.site-header .wrapper { justify-content: center; flex-wrap: wrap; gap: 8px; }
.site-header .site-title { font-size: 20px !important; font-weight: 700 !important; color: #111827 !important; margin-right: 12px !important; text-transform: uppercase; }
.site-header .site-nav .page-link { display: inline-block; margin: 2px 6px; font-size: 16px; }
.site-header .site-nav .trigger { justify-content: center; gap: 10px; flex-wrap: wrap; }
/* Stronger header divider and single-line layout */
.site-header { border-bottom: 3px solid #111827 !important; }
/* Header logo sizing */
.site-header .site-title .site-logo {
  height: 18px;
  width: 18px;
  object-fit: contain;
  margin-right: 8px;
  vertical-align: middle;
}
@media (max-width: 768px) {
  .affiliations {
    max-width: 100%;
    padding: 0 12px;
  }
}
@media (max-width: 640px) {
  .site-header .site-title {
    display: none !important;
  }
  .site-header .wrapper {
    justify-content: flex-end;
  }
  .site-header .site-nav .trigger {
    display: none;
    flex-direction: column;
    gap: 8px;
    padding: 12px 16px;
    margin-top: 10px;
    background: #ffffff;
    border: 1px solid rgba(15,23,42,0.1);
    border-radius: 12px;
    box-shadow: 0 12px 25px rgba(15,23,42,0.12);
    text-align: left;
  }
  .site-header .site-nav input.nav-trigger:checked ~ .trigger {
    display: flex;
  }
  .site-header .site-nav .page-link {
    font-size: 16px;
    margin: 4px 0;
  }
}
/* Performance table */
.perf-table { overflow-x: auto; margin: 8px 0 12px; }
.perf-table table { width: 100%; border-collapse: collapse; font-size: 14px; }
.perf-table th, .perf-table td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: center; white-space: nowrap; }
.perf-table thead th { background: #f9fafb; }
.perf-table caption { caption-side: top; font-weight: 600; margin-bottom: 6px; text-align: left; }
</style>



<h2 align="center" style="margin-top: 10px; margin-bottom: 5px; font-size: 28px;">🚀 DS SERVE: A Framework for Efficient and Scalable Neural Retrieval</h2>

<p align="center" class="authors" style="margin-bottom: 2px;">
  <a href="https://github.com/berkeleyljj" target="_blank">Jinjian Liu</a><sup>1*</sup>, 
  <a href="https://yichuan-w.github.io/" target="_blank">Yichuan Wang</a><sup>1*</sup>, 
  <a href="https://alrope123.github.io/" target="_blank">Xinxi Lyu</a><sup>2</sup>, 
  <a href="https://rulinshao.github.io/" target="_blank">Rulin Shao</a><sup>3</sup>,<br/>
  <a href="https://joeygonzalez.com/" target="_blank">Joseph E. Gonzalez</a><sup>1</sup>, 
  <a href="https://people.eecs.berkeley.edu/~matei/" target="_blank">Matei Zaharia</a><sup>1</sup>, 
  <a href="https://www.sewonmin.com/" target="_blank">Sewon Min</a><sup>1</sup>
</p>
<p align="center" class="affiliations">
  <sup>1</sup>University of California, Berkeley &nbsp;
  <sup>2</sup>University of Illinois Urbana–Champaign &nbsp;
  <sup>3</sup>University of Washington
</p>
<p align="center" style="font-size: 13px; margin-top: 2px;"><sup>*</sup>Equal contribution.</p>
<p align="center" style="margin-top: 6px;">[<a href="http://api.ds-serve.org:30888/ui">Web Interface</a>] [<a href="{{ 'API_DOCUMENTATION.html' | relative_url }}">API Endpoint</a>] [<a href="{{ 'VOTES_DOCUMENTATION.html' | relative_url }}">Voting System</a>] [<a href="https://github.com/Berkeley-Large-RAG/RAG-DS-Serve">Code</a>] [<a href="{{ 'assets/DS_SERVE_Camera_Ready.pdf' | relative_url }}">Paper</a>]</p>

<!-- **[✨NEW]** DiskANN integration: >2000 index-level QPS and up to 10000 QPS at 500B-token scale with ~200 GB RAM.

**[✨NEW]** Dual ANN backends: Choose between IVFPQ (~100 GB RAM, ~100 QPS) and DiskANN (+200 GB RAM, 10000 QPS) based on your throughput and memory requirements.

**[✨NEW]** Exact + Diverse modes: tune accuracy–diversity–latency on demand. -->

---
<br/>

<div class="overview-box">
  <ol style="margin: 0; padding-left: 20px;">
    <li style="margin-bottom: 8px;">You can turn any large in-house dataset (<1T tokens) into a <b>high-throughput (up to 10000 QPS)</b>, <b>memory-efficient (<200 GB RAM)</b> retrieval system with a <b>web UI and API</b>.</li>
    <li>Our <b>prototype</b>, built on <b>400B words</b> of high-quality LLM pre-training data, is readily available and provides downstream gains comparable to commercial search engine endpoints.</li>
  </ol>
</div>

<p align="center">
  <img src="{{ 'assets/good_ui_example.gif' | relative_url }}"
       alt="DS-Serve UI"
       style="width: 48%; margin: 5px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
  <img src="{{ 'assets/panel_example.gif' | relative_url }}"
       alt="DS-Serve control panel"
       style="width: 48%; margin: 5px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
</p>
<p align="center" class="small-note"><b>DS-Serve UI & control panel</b></p>

### Why was it previously challenging?

- **Scaling neural retrieval is hard.** Achieving high throughput, low memory use, and strong accuracy on very large datasets is non-trivial—traditional linear scan is simply infeasible.
- **Widely used ANN methods don’t scale gracefully.** At large scales, IVFPQ suffers from inefficient latency–performance tradeoffs and quantization errors, while HNSW demands substantial RAM, making modest deployments impractical.
- **End-to-end tooling is lacking.** Few frameworks offer a ready-to-use retrieval stack with a web UI, API endpoints, and built-in feedback collection.

As an example, most users default to search engines for general knowledge queries even when high-quality web data (e.g., LLM pre-training corpora) is publicly available. However, those engines are costly, low-throughput, and often unreliable at scale.

DS Serve addresses these challenges by making it easy to transform any large-scale in-house dataset into a high-throughput, memory-efficient neural retrieval system backed by DiskANN—complete with a web UI, API endpoints, and mechanisms for collecting search-result feedback. Our prototype (400B tokens, 2B vectors, 5 TB embeddings) matches the downstream gains of commercial search endpoints and, to the best of our knowledge, is the largest publicly accessible vector store.

<p align="center">
  <img src="{{ 'plots/Figure-1.png' | relative_url }}" style="width: 70%;" />
</p>
<p>
DS SERVE converts the largest pretraining dataset into an efficient neural retrieval system: a query q retrieves relevant text via ANN (IVFPQ or DiskANN), optionally reranks with exact and/or diverse search, and returns the top-k chunks with voting options for user feedback.
</p>

<!-- <div id="overview-note" class="callout-note"><span class="note-sup" aria-hidden="true">*</span> Note: For detailed technical explanations of the algorithms, see the <a href="#technical-design">Technical Design</a> section.</div> -->

<p>
See below for a set of <a href="#application">new applications our framework enables</a>, our <a href="#technical-design">detailed system design</a>, and <a href="#performance">performance benchmarks</a>!
</p>

---
<br/>

## Application

We envision DS Serve enabling a range of high-impact applications:

1. **Retrieval-Augmented Generation (RAG)**: DS Serve powers efficient RAG by feeding high-quality search results into LLMs. As shown in the [Performance section](#performance), it delivers superior accuracy and latency compared to both open-source baselines (like IVFPQ) and commercial search endpoints.
2. **Data Attribution & Curation**: By indexing entire pre-training corpora, DS Serve enables semantic data attribution, complementing n-gram based systems like OLMoTrace. It also facilitates advanced curation—allowing semantic deduplication, decontamination, and customized filtering for query-specific datasets.
3. **Training Search Agents**: Training deep-research agents requires high-frequency search rollouts that are often cost-prohibitive on commercial engines. DS Serve provides a free, high-throughput backend where developers can control latency-accuracy tradeoffs without rate limits.
4. **Pushing the Frontier of Search**: While traditional search engines struggle with long or complex queries, our vector-based approach handles them effectively. Additionally, the built-in voting system collects real-world labeled data to help build realistic benchmarks for retrieval research. 

---
<br/>

## Technical design 

### Datastore
Prior work showed that retrieval over large pre‑training corpora can improve RAG accuracy (see <a href="https://arxiv.org/abs/2112.04426" target="_blank">RETRO</a>, <a href="https://arxiv.org/abs/2407.12854" target="_blank">MassiveDS</a>, <a href="https://arxiv.org/abs/2507.01297" target="_blank">CompactDS</a>, <a href="https://arxiv.org/abs/2005.11401" target="_blank">RAG</a>, <a href="https://arxiv.org/abs/2002.08909" target="_blank">REALM</a>); however, accessible frameworks with modest resources have been lacking. DS SERVE is built on CompactDS, a 380‑billion‑word corpus (~2B vectors) spanning web crawl data, Wikipedia, research papers, and more.

This represents a significantly larger datastore than most prior work, and to the best of our knowledge is the largest pretraining dataset that users can access in open source for free. Typical evaluations run at much smaller scales (often ≤ tens of millions of vectors), e.g., <a href="https://microsoft.github.io/msmarco/" target="_blank">MS&nbsp;MARCO</a>, and <a href="https://dumps.wikimedia.org/" target="_blank">Wikipedia</a>, as well as consolidated leaderboards such as <a href="https://arxiv.org/abs/2104.08663" target="_blank">BEIR</a>. Even advanced commercial vector databases commonly impose per‑namespace/index limits well below the billion‑vector regime; see pricing/capacity notes for <a href="https://turbopuffer.com/pricing?namespaces=1&namespace=0&docs=1000000000&doc=7&writes=0&write=0" target="_blank">Turbopuffer</a>.

### Scalable and efficient search

**What is Approximate Nearest Neighbor (ANN) search?**

First, a database is embedded into a collection of vectors named datastore *D*. Then an index over *D* is built for easy lookup. Given a user query *q*, ANN returns the nearest neighbors—the vectors in *D* most semantically similar to *q*—through approximation. By visiting only part of the index, ANN retrieves faster than exhaustive search, which is infeasible at a billion‑vector scale. Therefore, ANN optimizes for latency with a small tradeoff in recall.

**The Challenge: IVFPQ and the Accuracy-Latency-Memory Tradeoff**

Most researchers and existing frameworks (including the <a href="https://arxiv.org/abs/2407.12854" target="_blank">CompactDS paper</a>) relying on **IVFPQ** (Inverted File with Product Quantization) have been struggling with the **accuracy-latency-memory** tradeoffs. At the billion-vector scale, IVFPQ requires heavy quantization to fit in RAM (sacrificing accuracy) or consumes excessive memory. Furthermore, increasing accuracy (e.g., larger `nprobe`) drastically reduces throughput.

**The Solution: DiskANN**

DS Serve addresses this by incorporating <a href="https://github.com/microsoft/DiskANN" target="_blank">DiskANN</a>, which significantly outperforms IVFPQ. By storing compressed vectors in RAM and full-precision vectors with the navigation graph on NVMe SSDs, DiskANN breaks traditional bottlenecks through implicit reranking during graph traversal.

**IVFPQ vs. DiskANN**

| Feature | IVFPQ (Traditional) | DiskANN (DS Serve) |
| :--- | :--- | :--- |
| **Accuracy** | **Lower**: Quantization noise reduces recall. | **Higher**: Full-precision vectors on disk ensure high recall. |
| **Throughput** | **~100 QPS**: More distance computations. | **>10,000 QPS**: Fewer distance computations using advanced data structure (navigation graph) and massively parallel I/O. |
| **Latency** | **Higher**: Sequential inverted list scanning. | **Lower**: Efficient graph traversal. |

We support both backends, but **DiskANN is the recommended default** for all high-performance deployments. DiskANN achieves >10,000 index-level QPS and 200+ end-to-end QPS with ~200 GB RAM, making it ideal for high-throughput deployments while maintaining competitive accuracy. In our internal evaluations, DiskANN's implicit reranking improved downstream accuracy compared to pure ANN and, on some tasks (e.g., MMLU), matched or exceeded exact search. Detailed benchmarks are provided below.

<details>
<summary><b>How DiskANN works (details)</b></summary>
<p>DiskANN stores compressed vectors in RAM and full-precision vectors with graph adjacency lists on SSD. Search begins with a candidate queue ordered by approximate distances from compressed vectors. At each step, the algorithm selects the nearest unvisited node, fetches its exact vector and adjacency list from SSD to refine the distance, then evaluates its neighbors using compressed representations. This alternates between compressed-distance frontier expansion and selective SSD loading for refinement until convergence. By loading only essential high precision data per query, DiskANN scales to billion-scale datasets while maintaining strong recall.</p>
</details>

<details>
<summary><b>Diverse Search &amp; Exact Search</b></summary>
<p><b>Diverse Search</b> (available on UI) applies MMR to reduce redundancy: <code>Score(i) = λ·sim(q,d_i) − (1−λ)·max<sub>j∈S</sub> sim(d_i,d_j)</code>. Enable this when results contain noticeable duplicates or near-duplicates.</p>
<p><b>Exact Search</b> (requires GPU, not on public UI) reranks ANN candidates by recomputing exact similarity scores using GritLM. This improves accuracy for harder queries and benefits from caching for repeated or similar queries. To enable Exact Search, build from source with a dedicated GPU.</p>
<p>For fastest latency/QPS, keep both toggles off. Use Diverse Search to de-duplicate results; use Exact Search when accuracy is critical and you have GPU compute available.</p>
</details>

## Performance 
<hr />

<h3 align="center">Interpolating DS Serve with LLM</h3>
<p>Downstream accuracy (%) with LLaMa 3.1 8B Instruct. DS Serve consistently improves accuracy across reasoning-intensive benchmarks; adding Exact Search provides further gains. We use <i>K</i>=1000, <i>k</i>=10, and <i>n</i><sub>probe</sub>=256.</p>
<p align="center">
  <img src="{{ 'plots/llm_interpolation_accuracy.png' | relative_url }}" alt="LLM interpolation accuracy" style="width: 75%; margin: 8px;" />
</p>

<h3 align="center">DiskANN vs IVFPQ</h3>
<p>DiskANN is more accurate <i>and</i> faster than IVFPQ. At recommended configs (DiskANN L=2000, IVFPQ nprobe=256), DiskANN achieves <b>~2.3× higher throughput</b> and <b>~2.2× lower latency</b>. The internal DiskANN index reaches up to 10,000 QPS; DiskANN is the recommended default. Accuracy comparison uses DiskANN L=2000 and IVFPQ nprobe=256 on <a href="https://arxiv.org/abs/1705.03551" target="_blank">TriviaQA</a> and <a href="https://arxiv.org/abs/1906.00300" target="_blank">Natural Questions</a>.</p>
<p align="center">
  <img src="{{ 'plots/diskann_vs_ivfpq_four_panel.png' | relative_url }}" alt="DiskANN vs IVFPQ: throughput, latency, TriviaQA and NaturalQS accuracy" style="width: 95%; margin: 8px;" />
</p>
<p>Accuracy metrics: <b>Recall</b> measures whether the LLM's answer contains the correct answer; <b>EM</b> (Exact Match) measures whether the LLM's answer exactly matches the correct answer after normalization; <b>F1</b> measures word-level overlap between the LLM's answer and the correct answer. IVFPQ remains available as a legacy option.</p>
<hr />


<h3 align="center">DS Serve vs Google Custom Search (CSE) API</h3>
<p>Latency (single-request), throughput (batched), and downstream accuracy comparison. Accuracy is averaged over MMLU, MMLU Pro, AGI Eval, GPQA, and MATH using LLaMa 3.1 8B Instruct (see <a href="https://arxiv.org/pdf/2507.01297" target="_blank">CompactDS Table 8</a>). Google CSE results use the official <a href="https://developers.google.com/custom-search/v1/overview" target="_blank">Custom Search JSON API</a>, please check the linked documentation for use guide.</p>
<p align="center">
  <img src="{{ 'plots/search_engine_latency_throughput.png' | relative_url }}" alt="Latency and throughput: Google API vs DS Serve Database" style="width: 48%; margin: 4px;" />
  <img src="{{ 'plots/search_engine_accuracy_avg.png' | relative_url }}" alt="Acccuracy: Search Engine vs DS Serve Database" style="width: 48%; margin: 4px;" />
</p>
<p>DS Serve (backed by CompactDS) achieves better downstream accuracy than Google CSE and it's also capable to offer <b>~30× higher throughput</b> (batched) and <b>~2× lower latency</b> (single-request)—all free of any API costs. We aim to reach 10,000 QPS end-to-end, matching the internal index-only throughput.</p>
<hr />

<h3 align="center">DiskANN Search Complexity (L) Ablation</h3>
<p>The search list size <b>L</b> controls the accuracy–latency tradeoff in DiskANN. L≈100 is sufficient for most queries; higher L improves accuracy for harder queries while remaining fast. Internal QPS reflects raw index throughput without embedding or network overhead—our goal is to close the gap between internal and end-to-end QPS through future optimizations.</p>
<p align="center">
  <img src="{{ 'plots/diskann_ablation_three_panel.png' | relative_url }}" alt="DiskANN L ablation: internal QPS, batched e2e QPS, and single-request latency" style="width: 95%; margin: 8px;" />
</p>

---
<br/>

## Acknowledgements

We thank the following open‑source projects and communities:

- <a href="https://github.com/RulinShao/massive-serve" target="_blank">Massive Serve</a> — for the serving infrastructure and deployment utilities that power DS Serve.
- <a href="https://github.com/facebookresearch/faiss" target="_blank">IVFPQ</a> and <a href="https://github.com/microsoft/DiskANN" target="_blank">DiskANN</a> — for enabling high‑performance ANN search at scale.
