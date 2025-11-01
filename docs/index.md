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
<p align="center" class="equal-note"><sup>*</sup>Equal contribution.</p>

<p align="center">[<a href="https://tinyurl.com/compact-ds-dive">Web Interface</a>] [<a href="{{ 'API_DOCUMENTATION.html' | relative_url }}">API Endpoint</a>] [<a href="{{ 'VOTES_DOCUMENTATION.html' | relative_url }}">Voting System</a>] [<a href="https://github.com/Berkeley-Large-RAG/RAG-DS-Serve">Code</a>] [<a href="https://openreview.net/forum?id=nQBZKcF2bo">Paper</a>]</p>

<!-- **[✨NEW]** DiskANN integration: ~1000 QPS at 500B-token scale with low RAM.

**[✨NEW]** Exact + Diverse modes: tune accuracy–diversity–latency on demand. -->

---
<br/>

## Overview

We introduce **DS Serve**, a framework that transforms a large-scale text corpus into a high-performance neural retrieval system.

In one word: Use our framework to **retrieve** data from **xx billion-scale**, **high-quality pre-trained datasets** — **blazing fast** ⚡ and **absolutely free!** 🚀

**DS Serve** realizes the transformation of the **largest datastore** (~**500B tokens**, ~**2B vectors**, ~**5T vector embeddings**), into a public domain that provides **free and high-performance neural retrieval endpoints**.  
We can offer <b>&lt;100&nbsp;ms latency</b> and handle <b>&gt;1000&nbsp;QPS</b>, enabling accurate search over a <b>pre‑trained‑scale datastore</b>.  
For deployment, everyone can clone our **codebase and index**, and run it with **~100 GB RAM** without any GPU for FAISS ANN, and **600 GB RAM** for additional DiskANN support.
Additionally, our framework enables you to convert your **in-house large-scale data** into a **high-performance, controllable** neural retrieval endpoint that you manage.

<p align="left"><i>Figure 1: DS SERVE converts a large dataset into a neural retrieval system: a query q retrieves relevant text via ANN, optionally reranks with exact and/or diverse search, and returns the top-k chunks with voting options for user feedback.</i></p>

<p align="center">
  <img src="{{ 'Figure-1.png' | relative_url }}" style="width: 70%;" />
</p>

**Key Contributions**
- We present the **DS Serve** framework to convert any text corpus into a high-performance, fully controllable in-house neural datastore, with a web interface and API endpoints.  
- Through this framework, we enable access to and controlled experiemtation on the largest publicly-deployed datastore, featuring 500B tokens and achieving ~1000 QPS. 
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

During search, **DS Serve** initially fetches a very large pool of candidates, and then easily finds top results among them. On the very rare occasion that retrieval fails, an alert message pops up to give improvement suggestions.

<p align="left"><i>Figure 3: Example failure mode and user guidance.</i></p>

<p align="center">
  <img src="{{ 'failure-mode.png' | relative_url }}" style="width: 70%;" />
</p>

<p align="left"><i>Table 1: Evaluation of DS SERVE on five established benchmarks. ‘Acc’ is accuracy (%), and t is end-to-end retrieval latency (s). For Exact Search, we report t without cache and tcache with cache. We use K = 1000, k = 10, and nprobe = 256 for all tasks.</i></p>

<p align="center">
  <img src="{{ 'table.png' | relative_url }}" style="width: 80%;" />
</p>

---
<br/>

## Technical design -- TO BE EXPANDED 

**Approximate Nearest Neighbor (ANN) Search**: ANN is the backbone of DS Serve. In particular, DS Serve incorporates FAISS IVFPQ, which reduces memory usage and latency by partitioning the vector space into clusters and avoiding full comparisons. In our setting, the ANN backbone supports inference within 200 ms at ~100GB memory overhead.

**Exact Search**: This mode is used to boost search accuracy. 

**Diversity Search**: Search results often suffer from information overlap, like nearly identical text chunks, so we offer a Diverse Search option to improve overall coverage of the results. To do this, we apply maximal marginal relevance (MMR) on candidates returned by ANN to penalize redundant information. In our use cases, we find Diverse Search substantially improves user experience by eliminating redundant texts.

---
<br/>

