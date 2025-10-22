---
layout: page
title: DS Serve
---

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
/* Use Infinigram base styles */
@import url('{{ "assets/infini-gram.css" | relative_url }}');
body {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: Arial, sans-serif;
  font-size: 18px;
  line-height: 1.6;
}
h2 { }
p { font-size: 18px; margin: 6px 0; }

/* Top navigation */
.top-nav {
  position: sticky;
  top: 0;
  z-index: 100;
  width: 100%;
  margin: 0 0 16px 0;
  background: #ffffff;
  border-bottom: 1px solid #eee;
}
.nav-inner {
  max-width: 800px;
  margin: 0 auto;
  padding: 10px 0;
  display: flex;
  align-items: center;
  justify-content: center; /* Center brand + links as a group */
  gap: 28px;
}
.brand {
  font-weight: 800;
  font-size: 26px; /* Bigger brand */
  letter-spacing: 0.2px;
  color: #111;
}
.nav-inner, .brand, .nav-link {
  font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', sans-serif;
}
.nav-link {
  text-decoration: none;
  color: #333;
  font-weight: 600;
  font-size: 16px;
  padding: 6px 0;
}
.nav-link:visited { color: #333; text-decoration: none; }
.nav-link:hover { color: #000; text-decoration: underline; text-underline-offset: 3px; }

/* Bracket links under title */
.bracket-links { text-align: center; word-spacing: 14px; margin: 6px 0 12px 0; }
.bracket-links a, .bracket-links a:visited { color: #1a73e8; text-decoration: none; word-spacing: normal; }
.bracket-links a:hover { text-decoration: none; }
.small-note { font-size: 14px; color: #666; margin-top: 2px; }

/* Author links like Infinigram: no underline */
.authors a, .authors a:visited { text-decoration: none; color: #1a73e8; }
.authors a:hover { text-decoration: none; }
</style>

<div class="top-nav">
  <div class="nav-inner">
    <span class="brand">DS Serve</span>
    <a class="nav-link" href="https://tinyurl.com/compact-ds-dive">Web Interface</a>
    <a class="nav-link" href="{{ 'API_DOCUMENTATION.html' | relative_url }}">API Endpoint</a>
    <a class="nav-link" href="{{ 'VOTES_DOCUMENTATION.html' | relative_url }}">Voting System</a>
    <a class="nav-link" href="https://github.com/Berkeley-Large-RAG/RAG-DS-Serve">Code</a>
    <a class="nav-link" href="https://openreview.net/forum?id=nQBZKcF2bo">Paper</a>
  </div>
</div>

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

<p class="bracket-links">[<a href="https://tinyurl.com/compact-ds-dive">Web Interface</a>] [<a href="{{ 'API_DOCUMENTATION.html' | relative_url }}">API Endpoint</a>] [<a href="{{ 'VOTES_DOCUMENTATION.html' | relative_url }}">Voting System</a>] [<a href="https://github.com/Berkeley-Large-RAG/RAG-DS-Serve">Code</a>] [<a href="https://openreview.net/forum?id=nQBZKcF2bo">Paper</a>]</p>

<!-- **[✨NEW]** DiskANN integration: ~1000 QPS at 500B-token scale with low RAM.

**[✨NEW]** Exact + Diverse modes: tune accuracy–diversity–latency on demand. -->

---
<br/>

## Overview

We introduce **DS Serve**, a framework that transforms a large-scale text corpus into a high-performance neural retrieval system.

In one word: Use our framework to **retrieve** data from **xx billion-scale**, **high-quality pre-trained datasets** — **blazing fast** ⚡ and **absolutely free!** 🚀

Our framework **DS Serve** realizes the transformation of the **largest datastore** (~**500B tokens**, ~**2B vectors**, ~**5T vector embeddings**), into a public domain that provides **free and high-performance neural retrieval endpoints**.  
We can offer **<100 ms latency** and handle **>1000 QPS**, enabling accurate search over a **pre-trained-scale datastore**.  
For deployment, everyone can clone our **codebase and index**, and run it using **under 200 GB RAM** without any GPU.
Additionally, our framework enables you to convert your **in-house large-scale data** into a **high-performance, controllable** neural retrieval endpoint that you manage.

<p align="left"><i>Figure 1: DS SERVE converts a large dataset into a neural retrieval system: a query q retrieves relevant text via ANN, optionally reranks with exact and/or diverse search, and returns the top-k chunks with voting options for user feedback.</i></p>

<p align="center">
  <img src="{{ 'Figure-1.png' | relative_url }}" style="width: 70%;" />
</p>

**Key Contributions**
- We present the **DS Serve** framework to convert any text corpus into a high-performance, fully controllable in-house neural datastore, with a web interface and API endpoints.  
- Through this framework, we enable access to and controlled experiemtation on the largest publicly-deployed datastore, featuring 500B tokens and achieving ~1000 QPS. 
- Furthermore, **DS Serve** contributes to practical applications including data attribution, training search agents, and pushing the frontier of search. For detail please refer to the Application section below. 

---
<br/>

## Application

We envision the use of DS Serve for fast, controllable retrieval in RAG and search applications.
1. **Data attribution & curation**: **DS Serve** can readily be used for training data attribution by indexing the entire pretraining corpus, as a complementary or improvement over OLMoTrace (). In addition, the framework enables improved data curation through semantic deduplication, decontamination, and customized filtering 
2. **Training search agents**: **DS Serve** addresses difficulties in search agent training by providing a fully controllable search backend, allowing developers to set their own latency-accuracy tradeoffs without incurring costs or rate limits.
3. **Pushing the frontier of search**: excels on longer, complex queries where keyword engines falter



---
<br/>

## User guide

- Use the parameters panel (see Figure 2) to control search behavior. The following parameters are all tunable:
  - **nprobe**: Higher values increase recall but marginally add latency. Therefore a large value is generally recommended.
  - **k (max: 1000)**: number of top passages to display. 
  - **Min words**: filter out passages shorter than this before display to encourage more context-rich results.
  - **Exact Search**: improves accuracy at the cost of increased compute and overhead.
  - **Diverse Search**: reduces redundant results for better coverage. 
  - **λ (lambda)**: diversity weight used only for **Diverse Search**. Higher leads to more diversity, and lower more relevance/accuracy.
  - **? icon**：click to reveal inline tooltips explaining each control parameter.

- Quick Walkthrough
  - Type a query. Optionally enable either or both of **Exact Search** to prioritize accuracy and **Diverse Search** to prioritize diversity. Then press "Enter" or click the arrow icon to search.
  - After results are shown. Click the expand/collapse button to control the displayed chunk. And the bottom-right of each passage features a voting option that allows users to vote **YES/NO** on the relevance of each result.


---
<br/>

## Examples

The **approximate** nature of the search backend inevitably sacrifices accuracy, thus we introduce Exact Search as an optional reranking mode. To do so, we compute **exact** similarities, instead of using **approximation**, between queries and passages. Then search results are reranked according to the newly computed **exact** scores. As shown in our evaluation results (Table 1), Exact Search effectively enhances accuracy across all five tasks. On a cold start, embedding the results can be slow, so we've built an embedding cache to allow ~1000ms latency on similar queries in Exact Search.

<p align="left"><i>Figure 2: Control panel with tunable parameters and tooltips.</i></p>

<p align="center">
  <img src="{{ 'parameter-panel.png' | relative_url }}" style="width: 60%;" />
</p>

Additionally, search results often suffer from information overlap, i.e. nearly identical text chunks. To address this problem we offer a Diverse Search option that penalizez redundant information. In our use cases, we find Diverse Search substantially improves user experience by eliminating redundant texts and improving overall coverage. 

For queries that are less common or rely on very recent knowledge — for example “Jensen Huang” — the datastore may only contain a handful of truly relevant passages, due to its frozen state. In these situations, Exact Search performs well because it prioritizes those relevant results at the top. By contrast, Diverse Search doesn’t necessarily improve performance since it risks surfacing less accurate passages when there aren’t enough strong candidates to begin with.

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

### Search Modes
**Approximate Nearest Neighbor (ANN) Search**: ANN is the backbone of DS Serve. In particular, DS Serve incorporates FAISS IVFPQ, which reduces memory usage and latency by partitioning the vector space into clusters and avoiding full comparisons. In our setting, the ANN backbone supports inference within 200 ms at ~100GB memory overhead.

**Exact Search**: This mode is used to boost search accuracy. 

**Diversity Search**: Search results often suffer from information overlap, like nearly identical text chunks, so we offer a Diverse Search option to improve overall coverage of the results. To do this, we apply maximal marginal relevance (MMR) on candidates returned by ANN to penalize redundant information. In our use cases, we find Diverse Search substantially improves user experience by eliminating redundant texts.

---
<br/>

