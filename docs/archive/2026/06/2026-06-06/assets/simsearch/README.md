# SimSearch — Minimal Vector Similarity Search

Built from first principles, inspired by FAISS and the "Inside FAISS: Billion-Scale Similarity Search" HN story.

## Index Types

| Index | Type | Speed | Memory | Accuracy |
|-------|------|-------|--------|----------|
| BruteForceIndex | Exact | O(n) | O(n×d×4) | 100% |
| IVFIndex | Approximate | O(n/nlist×nprobe) | O(n×d×4) | Configurable |
| PQIndex | Compressed | O(n×M) | O(n×M×1) | Approximate |

## Quick Start

```python
from simsearch import VectorIndex
import numpy as np

# Create 10,000 random 128-dim vectors
vectors = np.random.randn(10000, 128).astype(np.float32)

# Exact search
idx = VectorIndex(dim=128, index_type="bruteforce", metric="cosine")
idx.add(vectors)
results = idx.search(query_vector, k=10)
```

## Demo

```bash
python simsearch.py
```

Results on 10,000 vectors (64-dim):
- BruteForce: exact, 10K vectors
- IVF (nlist=50, nprobe=5): scans ~10% of data, ~40% recall@5
- PQ (M=8): 3.1% memory of float32 (80KB vs 2.5MB)
