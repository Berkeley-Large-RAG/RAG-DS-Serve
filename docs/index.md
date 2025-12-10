---
layout: page
title: DS Serve
---

<style>
@import url('{{ "assets/infini-gram.css" | relative_url }}');
p { font-size: 16px; margin: 12px 0; }
.small-note { font-size: 13px; color: #666; margin-top: 2px; }
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
.site-header .site-nav { display: none !important; } /* hide header links; keep title centered */
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



<h2 align="center" style="margin-top: 10px; margin-bottom: 5px; font-size: 28px;">🚀 <b>DS SERVE: A Framework for Efficient and Scalable Neural Retrieval</b></h2>

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
    <li>DiskANN delivers the best balance of accuracy, latency, and RAM cost in our deployments, outperforming IVFPQ on real workloads.</li>
  </ol>
</div>

<p align="center">
  <img src="{{ 'assets/ds-serve-ui.webp' | relative_url }}"
       alt="UI snippet"
       style="width: 36%; margin: 5px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
  <img src="{{ 'assets/panel.png' | relative_url }}" alt="Parameter panel snippet" style="width: 24%; margin: 5px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
</p>
<p align="center" class="small-note"><i>UI (gif) and parameter panel snippets</i></p>

### Why was it previously challenging?

- **Scaling neural retrieval is hard.** Achieving high throughput, low memory use, and strong accuracy on very large datasets is non-trivial—traditional linear scan is simply infeasible.
- **Widely used ANN methods don’t scale gracefully.** At large scales, IVFPQ suffers from inefficient latency–performance tradeoffs and quantization errors, while HNSW demands substantial RAM, making modest deployments impractical.
- **End-to-end tooling is lacking.** Few frameworks offer a ready-to-use retrieval stack with a web UI, API endpoints, and built-in feedback collection.

As an example, most users default to search engines for general knowledge queries even when high-quality web data (e.g., LLM pre-training corpora) is publicly available. However, those engines are costly, low-throughput, and often unreliable at scale.

**DS Serve** addresses these challenges by making it easy to transform any large-scale in-house dataset into a high-throughput, memory-efficient neural retrieval system backed by DiskANN—complete with a web UI, API endpoints, and mechanisms for collecting search-result feedback. Our prototype (400B tokens, 2B vectors, 5 TB embeddings) matches the downstream gains of commercial search endpoints and, to the best of our knowledge, is the largest publicly accessible vector store.

<p align="center">
  <img src="{{ 'plots/Figure-1.png' | relative_url }}" style="width: 70%;" />
</p>
<p>
DS SERVE converts the largest pretraining dataset into an efficient neural retrieval system: a query q retrieves relevant text via ANN (IVFPQ or DiskANN), optionally reranks with exact and/or diverse search, and returns the top-k chunks with voting options for user feedback.
</p>

<!-- <div id="overview-note" class="callout-note"><span class="note-sup" aria-hidden="true">*</span> Note: For detailed technical explanations of the algorithms, see the <a href="#technical-design">Technical Design</a> section.</div> -->

<p>
See below for a set of <a href="#application">new applications our framework enables</a>, <a href="#user-guide">documentation for using DS-Serve</a>, our <a href="#technical-design">detailed system design</a>, and <a href="#performance">performance benchmarks</a>!
</p>

---
<br/>

## Application

We envision **DS Serve** enabling a range of high-impact applications:

1. **Retrieval-Augmented Generation (RAG)**: DS Serve powers efficient RAG by feeding high-quality search results into LLMs. As shown in the [Performance section](#performance), it delivers superior accuracy and latency compared to both open-source baselines (like IVFPQ) and commercial search endpoints.
2. **Data Attribution & Curation**: By indexing entire pre-training corpora, DS Serve enables semantic data attribution, complementing n-gram based systems like OLMoTrace. It also facilitates advanced curation—allowing semantic deduplication, decontamination, and customized filtering for query-specific datasets.
3. **Training Search Agents**: Training deep-research agents requires high-frequency search rollouts that are often cost-prohibitive on commercial engines. DS Serve provides a free, high-throughput backend where developers can control latency-accuracy tradeoffs without rate limits.
4. **Pushing the Frontier of Search**: While traditional search engines struggle with long or complex queries, our vector-based approach handles them effectively. Additionally, the built-in voting system collects real-world labeled data to help build realistic benchmarks for retrieval research. 

---
<br/>

## User guide

We provide two ways to use **DS Serve**: API calls and a web UI.

<details>
<summary><b>API</b></summary>
<p><b>DS Serve</b> provides a free API for programmatic access via HTTP requests, enabling seamless integration into your applications and workflows. The API accepts configurable parameters and returns responses with retrieved passages and metadata. For detailed API documentation and usage examples, please refer to the <a href="{{ 'API_DOCUMENTATION.html' | relative_url }}">API Documentation</a> page.</p>
</details>

<details>
<summary><b>Web Interface</b></summary>
<p align="left"><i>Figure 2: Control panel with tunable parameters and tooltips.</i></p>
<p align="center">
  <img src="{{ 'panel.png' | relative_url }}" style="width: 90%;" />
</p>
<p>Use the control panel to adjust search behavior (Figure 2):</p>
<ul>
  <li><b>nprobe</b>: Higher values increase accuracy but marginally add latency. Therefore a large value is generally recommended.</li>
  <li><b>k (max: 1000)</b>: number of top passages to display.</li>
  <li><b>Min words</b>: filter out passages shorter than this before display to encourage more context-rich results.</li>
  <li><b>Exact Search</b>: improves accuracy at the cost of increased compute and overhead.</li>
  <li><b>Diverse Search</b>: reduces redundant results for better coverage.</li>
  <li><b>λ (lambda)</b>: diversity weight used only for <b>Diverse Search</b>. Higher values favor diversity, and lower relevance.</li>
  <li><b>? icon</b>: click to reveal inline tooltips explaining each control parameter.</li>
</ul>
<p><b>Quick Walkthrough</b></p>
<ul>
  <li>Type a query. Optionally enable <b>Exact Search</b> to prioritize accuracy and <b>Diverse Search</b> to prioritize diversity. Then press "Enter" or click the arrow icon to search with either IVF_PQ ANN or DiskANN backend.</li>
  <li>After results are shown, click the expand/collapse button to control the displayed chunk. Users can also vote <b>YES/NO</b> on the relevance of each result.</li>
</ul>
</details>

---
<br/>

## Technical design 

### Datastore
While prior work shows that retrieval over large pre‑training corpora can improve RAG accuracy (see <a href="https://arxiv.org/abs/2112.04426" target="_blank">RETRO</a>, <a href="https://arxiv.org/abs/2407.12854" target="_blank">MassiveDS</a>, <a href="https://arxiv.org/abs/2507.01297" target="_blank">CompactDS</a>,<a href="https://arxiv.org/abs/2005.11401" target="_blank">RAG</a>, <a href="https://arxiv.org/abs/2002.08909" target="_blank">REALM</a>), accessible frameworks for non‑experts to build and operate billion‑scale indexes have been lacking. Here, we demonstrate DS SERVE on CompactDS, a 380‑billion‑word corpus (~2B vectors) spanning web crawl data, Wikipedia, research papers, and more.

This represents a significantly larger datastore than most prior work, and to the best of our knowledge is the largest pretraining dataset that users can access in open source for free. Typical evaluations run at much smaller scales (often ≤ tens of millions of vectors), e.g., <a href="https://microsoft.github.io/msmarco/" target="_blank">MS&nbsp;MARCO</a>, <a href="https://ai.google.com/research/NaturalQuestions" target="_blank">Natural Questions</a>, and <a href="https://hotpotqa.github.io/" target="_blank">HotpotQA</a>, as well as consolidated leaderboards such as <a href="https://arxiv.org/abs/2104.08663" target="_blank">BEIR</a>. Even advanced commercial vector databases commonly impose per‑namespace/index limits well below the billion‑vector regime; see pricing/capacity notes for <a href="https://turbopuffer.com/pricing?namespaces=1&namespace=0&docs=1000000000&doc=7&writes=0&write=0" target="_blank">Turbopuffer</a>.

### Scalable and efficient search
<details>
<summary><b>What is Approximate Nearest Neighbor (ANN) search?</b></summary>
<p>First, a database is embedded into a colleciton of vectors named datastore <i>D</i>. Then an index over <i>D</i> is built for easy lookup. Given a user query <i>q</i>, ANN returns the nearest neighbors — the vectors in <i>D</i> most semantically similar to <i>q</i> -- through approximation. By visiting only part of the index, ANN retrieves faster than exhaustive search, which is infeasible at a billion‑vector scale. Therefore, ANN optimizes for latency with a small tradeoff in recall.<sup class="note-sup"><a href="#technical-design" aria-label="See Technical design">*</a></sup></p>
</details>
<details>
<summary><b>How we integrate Approximate Nearest Neighbor (ANN) search</b></summary>
<p>Real‑world vector datasets can contain billions of vectors and occupy terabytes. Keeping all vectors in DRAM is expensive. Two practical strategies reduce cost while preserving accuracy:</p>
<ul>
  <li>Quantization with in‑memory ANN (e.g., IVFPQ)</li>
  <li>Disk‑based ANN that stores vectors on SSDs with a small RAM cache (~10–20% of dataset)</li>
</ul>
<p>In DS Serve we support both backends:</p>
<ol>
  <li><b>IVFPQ</b><br/>
     We use <a href="https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#ivfpq" target="_blank">IVFPQ</a> to reduce memory and latency by clustering and product quantization.<br/>
     In our setting, IVFPQ supports inference within ~200 ms at ~100 GB RAM, achieving <b>~100 QPS</b> end‑to‑end.
  </li>
  <li><b>DiskANN</b><br/>
     For higher throughput, we integrate <a href="https://github.com/microsoft/DiskANN" target="_blank">DiskANN</a>, a disk‑based ANN system.<br/>
     DiskANN achieves <b>&gt;10000</b> index‑level QPS and <b>~200+ end‑to‑end QPS</b> at ~200 GB RAM, making it suitable for high‑throughput deployments while maintaining competitive accuracy.<br/>
     In our internal evaluations, DiskANN’s implicit reranking improved downstream accuracy compared to pure ANN and, on some tasks (e.g., MMLU), matched or exceeded Exact Search.
  </li>
</ol>
</details>

<details>
<summary><b>How DiskANN works</b></summary>
<p>DiskANN keeps a compressed copy of vectors in memory to compute approximate distances, while SSDs store full‑precision vectors and the proximity‑graph index. During search, the system fetches a node’s original vector (to refine distances) and its adjacency list (to continue traversal) from disk.</p>
</details>

## Performance 
<hr />

<div class="perf-table">
  <h3 style="margin: 6px 0 2px;">Interpolating DS-Serve with neural LLM</h3>
  <table>
    <caption>Table 1. LLaMa 3.1 8B results. <i>Acc</i> is accuracy (%); <i>t</i> is end‑to‑end retrieval latency (s). For Exact Search, <i>t</i> is without cache and <i>t</i><sub>cache</sub> with cache. We use <i>K</i>=1000, <i>k</i>=10, and <i>n</i><sub>probe</sub>=256.</caption>
    <thead>
      <tr>
        <th rowspan="2">Task</th>
        <th colspan="1">No DS Serve</th>
        <th colspan="2">DS Serve</th>
        <th colspan="3">DS Serve + Exact</th>
      </tr>
      <tr>
        <th>Acc</th>
        <th>Acc</th><th>t (s)</th>
        <th>Acc</th><th>t (s)</th><th>t<sub>cache</sub> (s)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="text-align:left">MMLU</td><td>68.9</td><td>73.5</td><td>0.17</td><td>73.7</td><td>16.44</td><td>0.30</td></tr>
      <tr><td style="text-align:left">MMLU Pro</td><td>39.8</td><td>47.5</td><td>0.19</td><td>49.4</td><td>16.54</td><td>0.32</td></tr>
      <tr><td style="text-align:left">AGI Eval</td><td>56.2</td><td>56.2</td><td>0.21</td><td>58.3</td><td>15.03</td><td>0.34</td></tr>
      <tr><td style="text-align:left">MATH</td><td>46.9</td><td>50.0</td><td>0.18</td><td>53.1</td><td>16.51</td><td>0.33</td></tr>
      <tr><td style="text-align:left">GPQA</td><td>29.9</td><td>31.7</td><td>0.17</td><td>36.6</td><td>16.57</td><td>0.32</td></tr>
    </tbody>
  </table>
</div>


<details>
<summary><b>What are post‑ANN Exact and Diverse Search?</b></summary>
<p><b>Exact Search</b> reranks ANN candidates by computing exact similarities between the query embedding and passages. We enable it on demand; with our embedding cache, latency remains practical for repeated or similar queries.</p>
<p><b>Diverse Search</b> reduces redundancy using maximal marginal relevance (MMR) on the ANN pool:</p>
<p align="center"><code>Score(i) = λ · sim(q, d<sub>i</sub>) − (1 − λ) · max<sub>j ∈ S</sub> sim(d<sub>i</sub>, d<sub>j</sub>)</code></p>
<p><small><code>sim(·,·)</code> is cosine similarity; <code>λ</code> (lambda) balances relevance and diversity.</small></p>
<p>In practice, Diverse Search eliminates redundant texts and improves overall coverage.</p>
</details>

<details>
<summary><b>Performance Analysis</b></summary>
The approximate nature of the ANN backend inevitably sacrifices accuracy, thus we integrate <b>Exact Search</b> as an optional reranking mode. To do so, we compute exact similarities, instead of using approximation, between queries and passages. Then search results are reranked according to the newly computed exact scores. In our evaluation results, <b>Exact Search</b> effectively enhances accuracy across all five tasks. On a cold start, embedding the results can be slow, so we've built an embedding cache to allow ~1000ms latency on similar queries in Exact Search. (Table 1)</p>

<p>Additionally, search results often suffer from information overlap, i.e. nearly identical text chunks. To address this problem we offer a Diverse Search option that penalizes redundant information. In our use cases, we find <b>Diverse Search</b> substantially improves user experience.</p>

<p>For queries that are less common or rely on very recent knowledge — for example “Jensen Huang” — the datastore may only contain a handful of truly relevant passages due to its frozen state. In these situations, <b>Exact Search</b> performs well because it ranks those relevant results to the top. In contrast, <b>Diverse Search</b> risks surfacing less accurate results.</p>

<p>During search, <b>DS Serve</b> initially oversamples a pool of candidates, and then finds top results among them. To make it easier for users, a with improvement suggestions shows in the case of failing to retrieve enough results, even though this is very rare.</p> 
</details>


<h3 align="center">Batched Retrieval</h3>
<p><b>QPS:</b> IVFPQ vs DiskANN when batching queries.</p>
<p align="center">
  <img src="{{ 'plots/ivfpq_qps_batched.png' | relative_url }}" alt="IVFPQ QPS batched" style="width: 44%; margin: 8px;" />
  <img src="{{ 'plots/diskann_qps_vs_L.png' | relative_url }}" alt="DiskANN batched QPS vs L" style="width: 44%; margin: 8px;" />
</p>
<p><b>Latency/Breakdown:</b> IVFPQ vs DiskANN batched latency.</p>
<p align="center">
  <img src="{{ 'plots/ivfpq_latency_batched.png' | relative_url }}" alt="IVFPQ latency batched" style="width: 44%; margin: 8px;" />
  <img src="{{ 'plots/diskann_latency_breakdown_vs_L.png' | relative_url }}" alt="DiskANN batched latency breakdown" style="width: 44%; margin: 8px;" />
</p>
<hr />

<h3 align="center">Single-request Retrieval</h3>
<p><b>QPS:</b> IVFPQ vs DiskANN for single-query serving.</p>
<p align="center">
  <img src="{{ 'plots/ivfpq_qps_single.png' | relative_url }}" alt="IVFPQ QPS single" style="width: 44%; margin: 8px;" />
  <img src="{{ 'plots/diskann_single_request_qps_vs_L.png' | relative_url }}" alt="DiskANN single QPS vs L" style="width: 44%; margin: 8px;" />
</p>
<p><b>Latency/Breakdown:</b> IVFPQ vs DiskANN single-request latency.</p>
<p align="center">
  <img src="{{ 'plots/ivfpq_latency_single.png' | relative_url }}" alt="IVFPQ latency single" style="width: 44%; margin: 8px;" />
  <img src="{{ 'plots/diskann_single_request_latency_vs_L.png' | relative_url }}" alt="DiskANN single latency vs L" style="width: 44%; margin: 8px;" />
</p>
<hr />

<h3 align="center">Accuracy comparisons</h3>
<p>Recall, Exact Match, and F1 for DiskANN vs IVFPQ (L=5000, nprobe=256) on TriviaQA and NaturalQS.</p>
<p align="center">
  <img src="{{ 'accuracy_ivfpq_vs_diskann_triviaqa.png' | relative_url }}" alt="TriviaQA accuracy DiskANN vs IVFPQ" style="width: 44%; margin: 8px;" />
  <img src="{{ 'accuracy_ivfpq_vs_diskann_naturalqs.png' | relative_url }}" alt="NaturalQS accuracy DiskANN vs IVFPQ" style="width: 44%; margin: 8px;" />
</p>
<p>Single-request search is used mainly for the interactive UI; batched search is recommended for higher-throughput API workloads.</p>
<hr />

<h3 align="center">Throughput comparison vs Google API</h3>
<p>Measured average QPS for single-request and batched modes.</p>
<p align="center">
  <img src="{{ 'plots/search_engine_qps_single.png' | relative_url }}" alt="Single-request QPS: Google API vs DS Serve" style="width: 44%; margin: 8px;" />
  <img src="{{ 'plots/search_engine_qps_batched.png' | relative_url }}" alt="Batched QPS: Google API vs DS Serve" style="width: 44%; margin: 8px;" />
</p>
<p class="small-note"><b>Note:</b> The latency number shown on the UI measures end-to-end wall-clock time (request setup, network travel, JSON encode/decode, rendering). QPS and latency can have small fluctuations depending on network speed.</p>

---
<br/>

## Acknowledgements

We thank the following open‑source projects and communities:

- <a href="https://github.com/RulinShao/massive-serve" target="_blank">Massive Serve</a> — for the serving infrastructure and deployment utilities that power DS Serve.
- <a href="https://github.com/facebookresearch/faiss" target="_blank">IVFPQ</a> and <a href="https://github.com/microsoft/DiskANN" target="_blank">DiskANN</a> — for enabling high‑performance ANN search at scale.
