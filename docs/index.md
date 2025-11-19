---
layout: page
title: DS Serve
---

<style>
@import url('{{ "assets/infini-gram.css" | relative_url }}');
p { font-size: 18px; margin: 6px 0; }
.small-note { font-size: 14px; color: #666; margin-top: 2px; }
/* Compact callout note for Overview */
.callout-note {
  font-size: 14px;
  color: #4b5563;
  background: #f9fafb;
  border-left: 4px solid #e5e7eb;
  padding: 8px 12px;
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
  font-size: 0.65em;
  line-height: 0;
  vertical-align: super;
}
.note-sup a {
  color: #6b7280;
  text-decoration: none;
}
.note-sup a:hover { text-decoration: underline; }
/* Make details summaries match paragraph sizing */
details > summary {
  font-size: 18px;
  margin: 6px 0;
  cursor: pointer;
}
/* Hide theme-injected page title on homepage to avoid duplicate 'DS Serve' */
.post-title, .page-title { display: none; }
/* Hide the Minima footer on the homepage to avoid duplicate site title */
.site-footer { display: none !important; }
/* Center the top navigation and enlarge links on the homepage */
.site-header .wrapper { justify-content: center; }
.site-header .site-title { font-size: 18px !important; font-weight: 600 !important; color: #111827 !important; margin-right: 12px !important; }
.site-header .site-nav .page-link { font-size: 18px; font-weight: 600; color: #111827; }
.site-header .site-nav .trigger { justify-content: center; gap: 10px; }
</style>



<h2 align="center">🚀 <b>DS SERVE: A Framework for Efficient and Scalable Neural Retrieval</b></h2>

<p align="center" class="authors">
  <a href="https://github.com/berkeleyljj" target="_blank">Jinjian Liu</a><sup>1*</sup>, 
  <a href="https://yichuan-w.github.io/" target="_blank">Yichuan Wang</a><sup>1*</sup>, 
  <a href="https://alrope123.github.io/" target="_blank">Xinxi Lyu</a><sup>2</sup>, 
  <a href="https://rulinshao.github.io/" target="_blank">Rulin Shao</a><sup>3</sup>,<br/>
  <a href="https://joeygonzalez.com/" target="_blank">Joseph E. Gonzalez</a><sup>1</sup>, 
  <a href="https://people.eecs.berkeley.edu/~matei/" target="_blank">Matei Zaharia</a><sup>1</sup>, 
  <a href="https://www.sewonmin.com/" target="_blank">Sewon Min</a><sup>1</sup>
</p>
<p align="center">
  <sup>1</sup>University of California, Berkeley &nbsp;
  <sup>2</sup>University of Illinois Urbana–Champaign &nbsp;
  <sup>3</sup>University of Washington
</p>
<p align="center"><sup>*</sup>Equal contribution.</p>

<p align="center">[<a href="{{ 'API_DOCUMENTATION.html' | relative_url }}">API Endpoint</a>] [<a href="https://tinyurl.com/compact-ds-dive">Web Interface</a>] [<a href="{{ 'VOTES_DOCUMENTATION.html' | relative_url }}">Voting System</a>] [<a href="https://github.com/Berkeley-Large-RAG/RAG-DS-Serve">Code</a>] [<a href="https://openreview.net/forum?id=nQBZKcF2bo">Paper</a>]</p>

<!-- **[✨NEW]** DiskANN integration: >2000 index-level QPS and ~200+ end-to-end QPS at 500B-token scale with ~200 GB RAM.

**[✨NEW]** Dual ANN backends: Choose between IVFPQ (~100 GB RAM, ~100 QPS) and DiskANN (+200 GB RAM, 200+ end-to-end QPS) based on your throughput and memory requirements.

**[✨NEW]** Exact + Diverse modes: tune accuracy–diversity–latency on demand. -->

---
<br/>

## Overview

The design of **DS Serve** is motivated by current challenges in information retrieval:
- Commercial search engines struggle with long and complex queries while being costly to deploy at scale, so a powerful yet affordable search framework is needed
- Exponential growth of information database obsoletes traditional linear search, urging more efficient neural retrieval.
- A gap persists between the NLP and database search community, preventing effective uses of search tools and algorithms like ANN<sup class="note-sup"><a href="#overview-note" aria-label="See note">*</a></sup>
- User labels for search results have been difficult to collect and curate.

To address these challenges, we introduce **DS Serve**, a framework that transforms a large-scale text corpus into a high-performance neural retrieval system that's:
- **[✨NEW]** blazing fast with high throughput 🚀 
- **[✨NEW]** built upon the largest datastore (~500B tokens, ~2B vectors, ~5T vector embeddings)
- **[✨NEW]** featuring customizable and efficient search backends -- DiskANN, Exact, and Diverse Search<sup class="note-sup"><a href="#overview-note" aria-label="See note">*</a></sup>
- **[✨NEW]** providing high-performance neural retrieval through free public endpoints and gathers user feedback in real-time

<p align="left"><i>Figure 1: DS SERVE converts the largest pretraining dataset into an efficient neural retrieval system: a query q retrieves relevant text via ANN (IVFPQ or DiskANN), optionally reranks with exact and/or diverse search, and returns the top-k chunks with voting options for user feedback.</i></p>
<p align="center">
  <img src="{{ 'plots/Figure-1.png' | relative_url }}" style="width: 70%;" />
</p>

<div id="overview-note" class="callout-note"><span class="note-sup" aria-hidden="true">*</span> Note: For detailed technical explanations of the algorithms, see the <a href="#technical-design">Technical design</a> section.</div>
---
<br/>



**Key Contributions**
- We present the **DS Serve** framework to convert any text corpus into a high-performance, fully controllable in-house neural datastore, with a web interface and API endpoints.  
- Through this framework, we enable access to and controlled experiemtation on the largest publicly-deployed datastore, featuring 500B tokens and achieving **10000+** index-level QPS with DiskANN integration and **200+** end-to-end QPS.
- We demonstrate DiskANN as a scalable and more accurate alternative to IVFPQ, achieving **10000+** QPS at the index level while maintaining manageable memory footprint (~200 GB RAM).
- Furthermore, **DS Serve** contributes to practical applications including data attribution, training search agents, and advancing search methods. For more details please refer to the Application section below. 

---
<br/>

## Application

We envision the use of **DS Serve** for fast, controllable retrieval in RAG and search applications:
1. **Data attribution & curation**: **DS Serve** can readily be used for training data attribution by indexing the entire pretraining corpus, as a complementary or improvement over OLMoTrace (). In addition, the framework enables improved data curation through semantic deduplication, decontamination, and customized filtering. 
2. **Training search agents**: **DS Serve** addresses difficulties in search agent training by providing a fully controllable search backend, allowing developers to set their own latency-accuracy tradeoffs without incurring costs or rate limits.
3. **Pushing the frontier of search**: Our vector-based framework is more effective for long and complex inputs than commercial search engines, and it also collects labeled data in real-time with the voting option. 

---
<br/>

## Performance 
<details>
<summary><b>What is Approximate Nearest Neighbor (ANN) search?</b></summary>
<p><b>ANN</b> quickly finds near neighbors by searching only part of the index instead of scanning exhaustively. It trades a bit of accuracy for much lower latency. That said, ANN retrieval still achieves noticeable accuracy gain compared to no-retrieval baselines.</p>
</details>

<details>
<summary><b>What are post‑ANN Exact and Diverse Search?</b></summary>
<p><b>Exact Search</b> reranks ANN candidates by computing exact similarities between the query embedding and passages. We enable it on demand; with our embedding cache, latency remains practical for repeated or similar queries.</p>
<p><b>Diverse Search</b> reduces redundancy using maximal marginal relevance (MMR) on the ANN pool:</p>
<p align="center"><code>Score(i) = λ · sim(q, d<sub>i</sub>) − (1 − λ) · max<sub>j ∈ S</sub> sim(d<sub>i</sub>, d<sub>j</sub>)</code></p>
<p><small><code>sim(·,·)</code> is cosine similarity; <code>λ</code> (lambda) balances relevance and diversity.</small></p>
<p>In practice, Diverse Search eliminates redundant texts and improves overall coverage.</p>
</details>

<details>
<summary><b>Why use Exact and Diverse Search?</b></summary>
<p>The approximate nature of the search backend inevitably sacrifices accuracy, thus we introduce <b>Exact Search</b> as an optional reranking mode. To do so, we compute exact similarities, instead of using approximation, between queries and passages. Then search results are reranked according to the newly computed exact scores. In our evaluation results, <b>Exact Search</b> effectively enhances accuracy across all five tasks. On a cold start, embedding the results can be slow, so we've built an embedding cache to allow ~1000ms latency on similar queries in Exact Search.</p>

<p>Additionally, search results often suffer from information overlap, i.e. nearly identical text chunks. To address this problem we offer a Diverse Search option that penalizes redundant information. In our use cases, we find <b>Diverse Search</b> substantially improves user experience by eliminating redundant texts and improving overall coverage.</p>

<p>For queries that are less common or rely on very recent knowledge — for example “Jensen Huang” — the datastore may only contain a handful of truly relevant passages, due to its frozen state. In these situations, <b>Exact Search</b> performs well because it ranks those relevant results to the top. In contrast, <b>Diverse Search</b> doesn’t necessarily improve performance since it risks surfacing less accurate results.</p>

<p>During search, <b>DS Serve</b> initially oversamples a pool of candidates, and then easily finds top results among them. On the very rare occasion that retrieval fails, an alert message pops up to give improvement suggestions.</p> 
</details>
---
<br/>

## User guide

We provide two ways to use **DS Serve**: API calls and a web UI.

- API Call
  - **DS Serve** provides a free API for programmatic access via HTTP requests, enabling seamless integration into your applications and workflows. The API accepts configurable parameters and returns responses with retrieved passages and metadata. For detailed API documentation and usage examples, please refer to the [API Documentation]({{ 'API_DOCUMENTATION.html' | relative_url }}) page.

- Web Interface
  - <p align="left"><i>Figure 2: Control panel with tunable parameters and tooltips.</i></p>
  - <p align="center">
    <img src="{{ 'plots/parameter-panel.png' | relative_url }}" style="width: 90%;" />
    </p>
  - Use the control panel to adjust search behavior (Figure 2):
    - <b>nprobe</b>: Higher values increase accuracy but marginally add latency. Therefore a large value is generally recommended.
    - <b>k (max: 1000)</b>: number of top passages to display. 
    - <b>Min words</b>: filter out passages shorter than this before display to encourage more context-rich results.
    - <b>Exact Search</b>: improves accuracy at the cost of increased compute and overhead.
    - <b>Diverse Search</b>: reduces redundant results for better coverage. 
    - <b>λ (lambda)</b>: diversity weight used only for <b>Diverse Search</b>. Higher values favor diversity, and lower relevance.
    - <b>? icon</b>: click to reveal inline tooltips explaining each control parameter.
  - Quick Walkthrough
    - Type a query. Optionally enable <b>Exact Search</b> to prioritize accuracy and <b>Diverse Search</b> to prioritize diversity. Then press "Enter" or click the arrow icon to search with either IVF_PQ ANN or DiskANN backend. 
    - After results are shown, click the expand/collapse button to control the displayed chunk. Users can also vote <b>YES/NO</b> on the relevance of each result.

---
<br/>


## Technical design 

### Datastore
While prior work shows that retrieval over large pre‑training corpora can improve RAG accuracy (see <a href="https://arxiv.org/abs/2112.04426" target="_blank">RETRO</a>, <a href="https://arxiv.org/abs/2407.12854" target="_blank">MassiveDS</a>, <a href="https://arxiv.org/abs/2507.01297" target="_blank">CompactDS</a>,<a href="https://arxiv.org/abs/2005.11401" target="_blank">RAG</a>, <a href="https://arxiv.org/abs/2002.08909" target="_blank">REALM</a>), accessible frameworks for non‑experts to build and operate billion‑scale indexes have been lacking. Here, we demonstrate DS SERVE on CompactDS, a 380‑billion‑word corpus (~2B vectors) spanning web crawl data, Wikipedia, research papers, and more.

This represents a significantly larger datastore than most prior work, and to the best of our knowledge is the largest pretraining dataset that users can access in open source for free. Typical evaluations run at much smaller scales (often ≤ tens of millions of vectors), e.g., <a href="https://microsoft.github.io/msmarco/" target="_blank">MS&nbsp;MARCO</a>, <a href="https://ai.google.com/research/NaturalQuestions" target="_blank">Natural Questions</a>, and <a href="https://hotpotqa.github.io/" target="_blank">HotpotQA</a>, as well as consolidated leaderboards such as <a href="https://arxiv.org/abs/2104.08663" target="_blank">BEIR</a>. Even advanced commercial vector databases commonly impose per‑namespace/index limits well below the billion‑vector regime; see pricing/capacity notes for <a href="https://turbopuffer.com/pricing?namespaces=1&namespace=0&docs=1000000000&doc=7&writes=0&write=0" target="_blank">Turbopuffer</a>.

### Scalable and efficient search
<details>
<summary><b>How we integrates Approximate Nearest Neighbor (ANN) search</b></summary>
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

Key takeaways:

1.We find in real open‑source deployments that DiskANN offers the best balance of accuracy, latency, and RAM cost -- overall outperforming IVFPQ.

<p align="left"><i>Figure 3: IVFPQ QPS scaling with nprobe parameter. Higher nprobe values improve accuracy at the cost of increased latency.</i></p>
<p align="center">
  <img src="{{ 'plots/faiss_qps_vs_nprobe.png' | relative_url }}" style="width: 90%; margin: 5px;" />
</p>

<p align="left"><i>Figure 4: DiskANN end-to-end QPS scaling with L parameter. Shows how throughput scales with search list size.</i></p>
<p align="left"><i>Figure 5: DiskANN index-level QPS scaling with L parameter. DiskANN achieves >2000 QPS at the index level, enabling high-throughput deployments.</i></p>
<p align="center">
  <img src="{{ 'plots/diskann_qps_vs_L.png' | relative_url }}" style="width: 48%; margin: 5px;" />
  <img src="{{ 'plots/diskann_index_only_qps_vs_L.png' | relative_url }}" style="width: 48%; margin: 5px;" />
</p>

<p align="left"><i>Figure 6: Accuracy comparison on TriviaQA and NQ-Open datasets. DiskANN consistently outperforms IVFPQ across both Exact match and F1 scores on both datasets.</i></p>
<p align="center">
  <img src="{{ 'plots/nq_triviaqa_diskann_vs_ann_compact_no_recall.png' | relative_url }}" style="width: 90%; margin: 5px;" />
</p>

<p align="left"><i>Figure 7: DiskANN latency breakdown showing the relative contribution of different components — embedding, index searching, and post‑search mapping — to total latency across different L parameter values.</i></p>
<p align="center">
  <img src="{{ 'plots/diskann_latency_breakdown_vs_L.png' | relative_url }}" style="width: 90%; margin: 5px;" />
</p>


<table style="width:100%; border-collapse:collapse; text-align:center;">
  <thead>
    <tr>
      <th style="border:1px solid #ddd; padding:6px;">Method</th>
      <th style="border:1px solid #ddd; padding:6px;">Accuracy</th>
      <th style="border:1px solid #ddd; padding:6px;">Latency</th>
      <th style="border:1px solid #ddd; padding:6px;">RAM cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">DiskANN</td>
      <td style="border:1px solid #ddd; padding:6px;">Excellent</td>
      <td style="border:1px solid #ddd; padding:6px;">Excellent</td>
      <td style="border:1px solid #ddd; padding:6px;">Excellent</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">IVFPQ</td>
      <td style="border:1px solid #ddd; padding:6px;">Poor</td>
      <td style="border:1px solid #ddd; padding:6px;">Good</td>
      <td style="border:1px solid #ddd; padding:6px;">Excellent</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">Linear scan (disk)</td>
      <td style="border:1px solid #ddd; padding:6px;">Excellent</td>
      <td style="border:1px solid #ddd; padding:6px;">Poor</td>
      <td style="border:1px solid #ddd; padding:6px;">Excellent</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">IVF (no PQ)</td>
      <td style="border:1px solid #ddd; padding:6px;">Good</td>
      <td style="border:1px solid #ddd; padding:6px;">Good</td>
      <td style="border:1px solid #ddd; padding:6px;">Poor</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">HNSW</td>
      <td style="border:1px solid #ddd; padding:6px;">Excellent</td>
      <td style="border:1px solid #ddd; padding:6px;">Excellent</td>
      <td style="border:1px solid #ddd; padding:6px;">Poor</td>
    </tr>
  </tbody>
</table>



2.yichuan add more take away[TODO]

 


---
<br/>



## Acknowledgements

We thank the following open‑source projects and communities:

- <a href="https://github.com/RulinShao/massive-serve" target="_blank">Massive Serve</a> — for the serving infrastructure and deployment utilities that power DS Serve.
- <a href="https://github.com/facebookresearch/faiss" target="_blank">IVFPQ</a> and <a href="https://github.com/microsoft/DiskANN" target="_blank">DiskANN</a> — for enabling high‑performance ANN search at scale.

