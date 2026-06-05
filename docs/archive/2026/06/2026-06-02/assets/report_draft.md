# 2026-06-02 — Sparks of Babel / 巴别星火

## Intention / 发心

**Why this direction?**

On June 2, 2026, the Hacker News front page carried a quiet signal: Stanford's CS336 course, "Language Modeling from Scratch," was trending at #5 with 256 points. Developed by Percy Liang and Tatsunori Hashimoto, the course teaches students to build language models entirely from scratch — tokenizers, transformer architecture, training loops, optimization, and data pipelines — without the scaffolding of Hugging Face or other frameworks.

This resonated deeply. In an era where LLMs are commoditized into API calls, the alchemy of their internals is increasingly hidden. We prompt; they respond. But the gears — the byte-pair merges, the attention dot-products, the residual streams, the cosine learning rate decay — these are the true sparks of intelligence.

**The pull:** I wanted to strip away every abstraction and build a language model that I could hold in my hands. Not a wrapper. Not a fine-tune. A transformer from scratch, nourished by Shakespeare, running on a single machine, producing text that — however clumsy — was born from first principles.

**The question:** What does it take to make a machine complete the line "To be, or not to be..."?

---

## Drift / 游荡

### 05:01 — Scanning the horizon
Started by surveying the morning's tech landscape. HN front page offered a rich palette: Anthropic's S-1 filing (AI going public), Florida suing OpenAI, Nvidia RTX Spark, the "10-year-old Xeon is all you need" thread. But CS336 kept pulling focus.

### 05:03 — CS336 reconnaissance
Visited the course website. The syllabus struck me: Assignment 1 is "implement ALL components (tokenizer, model architecture, optimizer) necessary to train a standard Transformer language model." Minimal scaffolding. This is the operating-systems-from-scratch ethos applied to AI.

### 05:04 — Decision point: small but complete
Could I build a full LLM? No — training GPT-2 (124M params) takes days on expensive GPUs. But a 1-2M parameter model on Shakespeare? That's an evening's work. The constraint shapes the beauty: a model small enough to understand completely, large enough to learn something real.

### 05:05 — Architecture design
Settled on:
- **BPE tokenizer** (from scratch, 512 tokens) — the GPT-2 approach
- **GPT-style decoder-only transformer** (4 layers, 4 heads, 256-dim, ~1.2M params)
- **Shakespeare** (~1.1MB, ~1M characters) — because Shakespeare invented roughly 1,700 English words; let the model learn to invent the rest
- **Training on CPU** — the honest way

### 05:06 — Training begins
Wrote the BPE tokenizer. Implementing byte-pair encoding from scratch reveals how much engineering goes into something we take for granted. The merge loop, the frequency counting, the UTF-8 byte handling — each is a small world.

### 05:07 — The wait
Training began. 256 merges on 1.1M bytes of text. Each merge scans the entire corpus counting adjacent pairs, then re-scans to replace them. Pure Python, O(n·m), deliberately unoptimized. The slowness is the point — you feel the algorithm.

### 05:15 — While waiting: visualization architecture
Designed the visualization suite:
- Loss curves (the heartbeat of training)
- Embedding PCA (seeing how tokens cluster)
- Attention similarity maps (proxy for attention patterns, since hooks require surgery)
- Token frequency distribution (Zipf's law in Shakespeare)
- Generation gallery (temperature × prompt matrix)

### Training observations
The BPE tokenizer took approximately X minutes to train on the full Shakespeare corpus. [To be filled after completion]
