---
layout: page
title: DS Serve
---

<style>
@import url('{{ "assets/infini-gram.css" | relative_url }}');
p { font-size: 18px; margin: 6px 0; }
.small-note { font-size: 14px; color: #666; margin-top: 2px; }
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

<p align="center">[<a href="https://tinyurl.com/compact-ds-dive">Web Interface</a>] [<a href="{{ 'API_DOCUMENTATION.html' | relative_url }}">API Endpoint</a>] [<a href="{{ 'VOTES_DOCUMENTATION.html' | relative_url }}">Voting System</a>] [<a href="https://github.com/Berkeley-Large-RAG/RAG-DS-Serve">Code</a>] [<a href="https://openreview.net/forum?id=nQBZKcF2bo">Paper</a>]</p>

<!-- **[✨NEW]** DiskANN integration: >2000 index-level QPS and ~200+ end-to-end QPS at 500B-token scale with ~200 GB RAM.

**[✨NEW]** Dual ANN backends: Choose between FAISS (~100 GB RAM, ~100 QPS) and DiskANN (+200 GB RAM, 200+ end-to-end QPS) based on your throughput and memory requirements.

**[✨NEW]** Exact + Diverse modes: tune accuracy–diversity–latency on demand. -->

---
<br/>

## Overview

We introduce **DS Serve**, a framework that transforms a large-scale text corpus into a high-performance neural retrieval system.

In one word: Use our framework to **retrieve** data from **near tillion-scale**, **high-quality pre-trained datasets** — **blazing fast** ⚡ and **absolutely free!** 🚀

**DS Serve** realizes the transformation of the **largest datastore** (~**500B tokens**, ~**2B vectors**, ~**5T vector embeddings**), into a public domain that provides **free and high-performance neural retrieval endpoints**.  

We support two high-performance ANN backends: **FAISS IVFPQ** for memory-efficient retrieval and **DiskANN** for more accurate, high-throughput search. With **FAISS**, we can offer <b>&lt;100&nbsp;ms latency</b> and handle <b>&gt;100&nbsp;end-to-end QPS</b> with a modest memory footprint of **~100GB**. With **DiskANN**, we can achieve **10000+ index-level QPS** and **200+ end-to-end QPS**, with an extra RAM usage of **~200 GB**. This enables state-of-the-art retrieval performance at scale while maintaining low memory overhead.

Additionally, our framework enables you to convert your **in-house large-scale data** into a **high-performance, controllable** neural retrieval endpoint that you can fully customize with different search options.


---
<br/>
## Contributions
<p align="left"><i>Figure 1: DS SERVE converts the largest pretraining dataset into an efficient neural retrieval system: a query q retrieves relevant text via ANN, optionally reranks with exact and/or diverse search, and returns the top-k chunks with voting options for user feedback.</i></p>

<p align="center">
  <img src="{{ 'Figure-1.png' | relative_url }}" style="width: 70%;" />
</p>

**Key Contributions**
- We present the **DS Serve** framework to convert any text corpus into a high-performance, fully controllable in-house neural datastore, with a web interface and API endpoints.  
- Through this framework, we enable access to and controlled experiemtation on the largest publicly-deployed datastore, featuring 500B tokens and achieving **>2000 index-level QPS** with **DiskANN** integration and **200+ end-to-end QPS**.
- We demonstrate **DiskANN** as a scalable and more accurate alternative to FAISS, achieving **>2000 QPS** at the index level while maintaining manageable memory footprint (~200 GB RAM).
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

## User guide

**DS Serve** provides a free API for programmatic access via HTTP requests, enabling seamless integration into your applications and workflows. The API accepts configurable parameters and returns responses with retrieved passages and metadata. For detailed API documentation and usage examples, please refer to the [API Documentation]({{ 'API_DOCUMENTATION.html' | relative_url }}) page.

Additionally, **DS Serve** offers a **Web Interface** for interactive exploration with a visual control panel, ideal for experimentation and visualization.

### Web Interface

<p align="left"><i>Figure 2: Control panel with tunable parameters and tooltips.</i></p>

<p align="center">
<img src="{{ 'parameter-panel.png' | relative_url }}" style="width: 90%;" />
</p>

- Use the control panel to adjust search behavior through the following parameters (Figure 2):
  - **nprobe**: Higher values increase accuracy but marginally add latency. Therefore a large value is generally recommended.
  - **k (max: 1000)**: number of top passages to display. 
  - **Min words**: filter out passages shorter than this before display to encourage more context-rich results.
  - **Exact Search**: improves accuracy at the cost of increased compute and overhead.
  - **Diverse Search**: reduces redundant results for better coverage. 
  - **λ (lambda)**: diversity weight used only for **Diverse Search**. Higher values favor diversity, and lower relevance.
  - **? icon**: click to reveal inline tooltips explaining each control parameter.

- Quick Walkthrough
  - Type a query. Optionally enable **Exact Search** to prioritize accuracy and **Diverse Search** to prioritize diversity. Then press "Enter" or click the arrow icon to search.
  - After results are shown, click the expand/collapse button to control the displayed chunk. Users can also vote **YES/NO** on the relevance of each result.



---
<br/>

## Examples

The **approximate** nature of the search backend inevitably sacrifices accuracy, thus we introduce **Exact Search** as an optional reranking mode. To do so, we compute **exact** similarities, instead of using **approximation**, between queries and passages. Then search results are reranked according to the newly computed **exact** scores. As shown in our evaluation results (Table 1), **Exact Search** effectively enhances accuracy across all five tasks. On a cold start, embedding the results can be slow, so we've built an embedding cache to allow ~1000ms latency on similar queries in Exact Search.

Additionally, search results often suffer from information overlap, i.e. nearly identical text chunks. To address this problem we offer a Diverse Search option that penalizes redundant information. In our use cases, we find **Diverse Search** substantially improves user experience by eliminating redundant texts and improving overall coverage. 

For queries that are less common or rely on very recent knowledge — for example “Jensen Huang” — the datastore may only contain a handful of truly relevant passages, due to its frozen state. In these situations, **Exact Search** performs well because it ranks those relevant results to the top. In contrast, **Diverse Search** doesn’t necessarily improve performance since it risks surfacing less accurate results.

During search, **DS Serve** initially oversamples a pool of candidates, and then easily finds top results among them. On the very rare occasion that retrieval fails, an alert message pops up to give improvement suggestions.

<p align="left"><i>Figure 3: Example failure mode and user guidance.</i></p>

<p align="center">
  <img src="{{ 'failure-mode.png' | relative_url }}" style="width: 70%;" />
</p>

<p align="left"><i>Table 1: Evaluation of DS SERVE on five established benchmarks. 'Acc' is accuracy (%), and t is end-to-end retrieval latency (s). For Exact Search, we report t without cache and tcache with cache. We use K = 1000, k = 10, and nprobe = 256 for all tasks.</i></p>

<p align="center">
  <img src="{{ 'table.png' | relative_url }}" style="width: 80%;" />
</p>

<p align="left"><i>Figure 3: Accuracy comparison between FAISS and DiskANN across MMLU, MMLU Pro, and MATH benchmarks. DiskANN beats FAISS ANN in accuracy across all tasks while providing significantly higher throughput.</i></p>
<p align="center">
  <img src="{{ 'accuracy_ann_diskann_full_table_bars.png' | relative_url }}" style="width: 48%; margin: 5px;" />
</p>

<p align="left"><i>Figure 4: Comprehensive accuracy comparison across ANN, Exact Search, and DiskANN modes. DiskANN achieves competitive accuracy compared to FAISS ANN and Exact Search.</i></p>
<p align="center">
  <img src="{{ 'accuracy_compact_ds_ann_exact_diskann.png' | relative_url }}" style="width: 48%; margin: 5px;" />
</p>

<p align="left"><i>Figure 5: Accuracy comparison on TriviaQA and NQ-Open datasets. DiskANN consistently outperforms FAISS across both Exact match and F1 scores on both datasets.</i></p>
<p align="center">
  <img src="{{ 'accuracy_faiss_vs_diskann_triviaqa_nq.png' | relative_url }}" style="width: 90%; margin: 5px;" />
</p>

<p align="left"><i>Figure 6: FAISS QPS scaling with nprobe parameter. Higher nprobe values improve accuracy at the cost of increased latency.</i></p>
<p align="left"><i>Figure 7: FAISS latency scaling with nprobe parameter. Shows the latency-accuracy tradeoff for different nprobe values.</i></p>
<p align="center">
  <img src="{{ 'faiss_qps_vs_nprobe.png' | relative_url }}" style="width: 48%; margin: 5px;" />
  <img src="{{ 'faiss_latency_vs_nprobe.png' | relative_url }}" style="width: 48%; margin: 5px;" />
</p>

<p align="left"><i>Figure 8: DiskANN end-to-end QPS scaling with L parameter (search list size). Shows how throughput scales with search list size.</i></p>
<p align="left"><i>Figure 9: DiskANN index-level QPS scaling with L parameter. DiskANN achieves >2000 QPS at the index level, enabling high-throughput deployments.</i></p>
<p align="center">
  <img src="{{ 'diskann_qps_vs_L.png' | relative_url }}" style="width: 48%; margin: 5px;" />
  <img src="{{ 'diskann_index_only_qps_vs_L.png' | relative_url }}" style="width: 48%; margin: 5px;" />
</p>

<p align="left"><i>Figure 10: DiskANN latency breakdown showing the relative contribution of different components :embedding, index searching, and post search mapping to total latency across different L parameter values.</i></p>
<p align="center">
  <img src="{{ 'diskann_latency_breakdown_vs_L.png' | relative_url }}" style="width: 90%; margin: 5px;" />
</p>

---
<br/>

## Technical design 

### Datastore
While prior work shows that retrieval over large pre‑training corpora can improve RAG accuracy (see <a href="https://arxiv.org/abs/2112.04426" target="_blank">RETRO</a>, <a href="https://arxiv.org/abs/2407.12854" target="_blank">MassiveDS</a>, <a href="https://arxiv.org/abs/2507.01297" target="_blank">CompactDS</a>,<a href="https://arxiv.org/abs/2005.11401" target="_blank">RAG</a>, <a href="https://arxiv.org/abs/2002.08909" target="_blank">REALM</a>), accessible frameworks for non‑experts to build and operate billion‑scale indexes have been lacking. Here, we demonstrate DS SERVE on CompactDS, a 380‑billion‑word corpus (~2B vectors) spanning web crawl data, Wikipedia, research papers, and more.

This represents a significantly larger datastore than most prior work, and to the best of our knowledge is the largest pretraining dataset that users can access in open source for free. Typical evaluations run at much smaller scales (often ≤ tens of millions of vectors), e.g., <a href="https://microsoft.github.io/msmarco/" target="_blank">MS&nbsp;MARCO</a>, <a href="https://ai.google.com/research/NaturalQuestions" target="_blank">Natural Questions</a>, and <a href="https://hotpotqa.github.io/" target="_blank">HotpotQA</a>, as well as consolidated leaderboards such as <a href="https://arxiv.org/abs/2104.08663" target="_blank">BEIR</a>. Even advanced commercial vector databases commonly impose per‑namespace/index limits well below the billion‑vector regime; see pricing/capacity notes for <a href="https://turbopuffer.com/pricing?namespaces=1&namespace=0&docs=1000000000&doc=7&writes=0&write=0" target="_blank">Turbopuffer</a>.

### Scalable and efficient search
<details>
<summary><b>Neural retrieval formulation</b></summary>
<p>Neural retrieval can be viewed as nearest‑neighbor search: select the top‑k chunks by cosine similarity <i>sim</i>(<i>q</i>, <i>d</i><sub>i</sub>), where <i>q</i>, <i>d</i><sub>i</sub> ∈ R<sup>h</sup> are the embedding vectors of the query and a candidate chunk. We use <a href="https://arxiv.org/abs/2112.09118" target="_blank">Contriever</a> as the encoder.</p>
</details>
<details>
<summary><b>Why use ANN?</b></summary>
<p>Exact nearest neighbor search over billions of vectors is prohibitively slow and memory intensive. ANN prunes comparisons via indexing and quantization, delivering near‑exact quality with orders‑of‑magnitude fewer distance computations—cutting latency and RAM while preserving accuracy.</p>
</details>


Real‑world vector datasets can contain billions of vectors and occupy terabytes. Keeping all vectors in DRAM is expensive. Two practical strategies reduce cost while preserving accuracy:

- Quantization with in‑memory ANN (e.g., FAISS IVFPQ)
- Disk‑based ANN that stores vectors on SSDs with a small RAM cache (~10–20% of dataset)

In DS Serve we support both backends:

1. **FAISS IVFPQ**  
   We use <a href="https://faiss.ai/" target="_blank">FAISS</a> with <a href="https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#ivfpq" target="_blank">IVFPQ</a> to reduce memory and latency by clustering and product quantization.  
   In our setting, FAISS supports inference within ~200 ms at ~100 GB RAM, achieving **~100 QPS** end‑to‑end.

2. **DiskANN**  
   For higher throughput, we integrate <a href="https://github.com/microsoft/DiskANN" target="_blank">DiskANN</a>, a disk‑based ANN system.  
   DiskANN achieves **>1000 index‑level QPS** and **~200+ end‑to‑end QPS** at ~200 GB RAM, making it suitable for high‑throughput deployments while maintaining competitive accuracy.  
   In our internal evaluations, DiskANN’s implicit reranking improved downstream accuracy compared to pure ANN and, on some tasks (e.g., MMLU), matched or exceeded Exact Search.

<details>
<summary><b>How DiskANN works</b></summary>
<p>DiskANN keeps a compressed copy of vectors in memory to compute approximate distances, while SSDs store full‑precision vectors and the proximity‑graph index. During search, the system fetches a node’s original vector (to refine distances) and its adjacency list (to continue traversal) from disk.</p>
<p>References: <a href="https://github.com/microsoft/DiskANN" target="_blank">GitHub</a>, <a href="https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/" target="_blank">MSR overview</a></p>
</details>

Key takeaways:
1.We find in real open‑source deployments that DiskANN offers the best balance of accuracy, latency, and RAM cost.

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
      <td style="border:1px solid #ddd; padding:6px;">IVFPQ (FAISS)</td>
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

2. @yichuan add more take away
### Exact Search

This mode boosts search accuracy by computing exact similarities between the query embedding and candidate passages (no approximation). It costs more compute, so we enable it on demand; combined with our embedding cache, latency remains practical for repeated or similar queries.

### Diversity Search

Search results often contain redundant passages (near‑duplicates). **Diverse Search** improves coverage by penalizing redundancy using maximal marginal relevance (MMR) <a href="https://dl.acm.org/doi/10.1145/290941.291025" target="_blank">MMR, diversity-based reranking</a> on the ANN candidates.

<p><b>MMR scoring</b> at step <i>t</i> with selected set <i>S</i>:</p>
<p align="center"><code>Score(i) = λ · sim(q, d<sub>i</sub>) − (1 − λ) · max<sub>j ∈ S</sub> sim(d<sub>i</sub>, d<sub>j</sub>)</code></p>
<p><small><code>sim(·,·)</code> is cosine similarity; <code>λ</code> (lambda) balances relevance and diversity.</small></p>

In our use cases, **Diverse Search** eliminates redundant texts and improves overall coverage.


---
<br/>



