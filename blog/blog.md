# 🚀 DS Serve Documentation

## 🌟 Overview

**DS Serve** transforms large-scale text corpus into high-performance neural retrieval systems without daunting hardware requirements. Achieves low latency with modest memory overhead and supports inference time tradeoffs across accuracy, diversity, and latency.

<img src="Figure%201.png" alt="Figure 1: DS Serve Performance Overview" style="max-width: 100%; height: auto; width: 600px;">

*Figure 1: DS Serve Performance Overview*

**Key Performance**: DiskANN integration delivers ~1000 QPS at 500B token scale with low RAM footprint. Tunable controls enable smooth accuracy–latency tradeoffs, increasing QPS by hundreds of times over basic ANN.

## 🎯 Use Cases

### 📊 Applications

**🔍 Data Attribution & Curation**: Organize and search large data quantities with semantic understanding beyond keyword matching. Complements systems like OLMoTrace and enables smart dataset cleanup.

**🤖 Training Search Agents**: Provides controllable search backends for deep research applications, eliminating cost/throughput bottlenecks with customizable latency-accuracy tradeoffs.

**🚀 Advancing Search Frontiers**: Excels at longer, complex queries where traditional search engines fall short. Vector-based approach captures meaning better than keyword matching.

### 📈 Case Studies

**⚖️ Accuracy–Diversity Tradeoff**: For rare queries (e.g., "Jensen Huang"), Exact Search prioritizes relevant results. Diverse Search may surface less accurate passages when strong candidates are limited.

**🛡️ Failure Handling**: Fetches large candidate pools (K=1000) to ensure top-k results. Clear alerts provide helpful suggestions on rare retrieval failures.

## 📖 User Guide

### 🚀 Quick Walkthrough

**Control Panel**: Tunable parameters with helpful tooltips via question mark icon. Choose between Exact and Diverse Search modes for accuracy vs. diversity prioritization.

**Search Process**: Type query → press "Enter" → view scored results. Expand/collapse passages with one click. All searches logged in history panel.

### 🎥 Demo

[Video placeholder - insert your demonstration video here]

## ⚙️ Technical Design

### 🔍 Search Modes

**🎯 ANN Search**: FAISS IVFPQ backbone reduces memory usage and latency by partitioning vector space into clusters. Supports 200ms inference at ~100GB memory overhead.

**🎯 Exact Search**: Optional reranking mode using GritLM for exact query-passage similarity computation. Enhances accuracy across all tasks. Embedding cache enables ~1000ms latency on similar queries.

**🎯 Diversity Search**: Applies MMR to penalize redundant information and improve result coverage. Eliminates nearly identical text chunks for better user experience.

---

*This documentation follows the structure and style of the [infini-gram documentation](https://github.com/liujch1998/infini-gram/blob/master/docs/index.md) while presenting the unique capabilities and technical design of DS Serve.*
