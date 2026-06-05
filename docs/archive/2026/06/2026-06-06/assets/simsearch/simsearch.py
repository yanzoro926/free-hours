"""
SimSearch — A minimal vector similarity search engine from first principles.

Implements:
1. Brute-force cosine/L2 similarity (exact search)
2. IVF (Inverted File Index) — approximate search with clustering
3. PQ (Product Quantization) — memory-efficient compressed vectors

Inspired by FAISS but built from scratch to understand the algorithms.

Usage:
    from simsearch import VectorIndex
    idx = VectorIndex(dim=128)
    idx.add(vectors, ids)
    results = idx.search(query, k=10)
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray


class BruteForceIndex:
    """Exact nearest neighbor search using brute force."""

    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self.vectors: Optional[np.ndarray] = None
        self.ids: Optional[np.ndarray] = None

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """Add vectors to the index."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        assert vectors.shape[1] == self.dim

        if ids is None:
            ids = np.arange(len(vectors))

        if self.vectors is None:
            self.vectors = vectors
            self.ids = np.asarray(ids)
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.ids = np.concatenate([self.ids, np.asarray(ids)])

    def search(self, query: np.ndarray, k: int = 10) -> SearchResult:
        """Search for k nearest neighbors."""
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)

        if self.metric == "cosine":
            # Normalize for cosine similarity
            q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
            v_norm = self.vectors / (
                np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
            )
            scores = np.dot(q_norm, v_norm.T)[0]
            # Convert similarity to distance (1 - similarity)
            distances = 1.0 - scores
        elif self.metric == "l2":
            diff = self.vectors - query
            distances = np.linalg.norm(diff, axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        top_k = np.argsort(distances)[:k]
        return SearchResult(
            ids=self.ids[top_k], distances=distances[top_k]
        )

    def __len__(self) -> int:
        return len(self.vectors) if self.vectors is not None else 0


class IVFIndex:
    """
    Inverted File Index — approximate nearest neighbor search.

    1. Cluster vectors into nlist centroids (k-means)
    2. Assign each vector to nearest centroid
    3. Search: only scan vectors in the nprobe nearest clusters
    """

    def __init__(
        self, dim: int, nlist: int = 100, nprobe: int = 10, metric: str = "cosine"
    ):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric
        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: dict[int, list] = {}  # cluster_id → [(vec, id), ...]
        self._all_ids = []
        self._trained = False

    def train(self, vectors: np.ndarray):
        """Train k-means centroids on the data."""
        vectors = np.asarray(vectors, dtype=np.float32)
        n = len(vectors)

        # Simple k-means implementation
        rng = np.random.default_rng(42)
        # Initialize centroids with random vectors
        indices = rng.choice(n, min(self.nlist, n), replace=False)
        self.centroids = vectors[indices].copy().astype(np.float32)

        if n < self.nlist:
            self.nlist = n
            self.centroids = vectors.copy()
            self._trained = True
            return

        # Run k-means iterations
        for _ in range(20):
            # Assign to nearest centroid
            if self.metric == "cosine":
                c_norm = self.centroids / (
                    np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-10
                )
                v_norm = vectors / (
                    np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
                )
                similarities = np.dot(v_norm, c_norm.T)
                assignments = np.argmax(similarities, axis=1)
            else:
                diff = vectors[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=2)
                assignments = np.argmin(distances, axis=1)

            # Update centroids
            for i in range(self.nlist):
                mask = assignments == i
                if mask.sum() > 0:
                    self.centroids[i] = vectors[mask].mean(axis=0)

        self._trained = True

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """Add vectors to the inverted lists."""
        if not self._trained:
            raise RuntimeError("Must call train() before add()")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if ids is None:
            ids = np.arange(len(vectors)) + len(self._all_ids)

        # Assign each vector to nearest centroid
        if self.metric == "cosine":
            c_norm = self.centroids / (
                np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-10
            )
            v_norm = vectors / (
                np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
            )
            similarities = np.dot(v_norm, c_norm.T)
            assignments = np.argmax(similarities, axis=1)
        else:
            diff = vectors[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
            assignments = np.argmin(distances, axis=1)

        for i, (vec, idx, cluster) in enumerate(
            zip(vectors, ids, assignments)
        ):
            c = int(cluster)
            if c not in self.inverted_lists:
                self.inverted_lists[c] = []
            self.inverted_lists[c].append((vec, int(idx)))
            self._all_ids.append(int(idx))

    def search(self, query: np.ndarray, k: int = 10) -> SearchResult:
        """Search using IVF — only scan nprobe nearest clusters."""
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)

        # Find nprobe nearest centroids
        if self.metric == "cosine":
            q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
            c_norm = self.centroids / (
                np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-10
            )
            centroid_scores = np.dot(q_norm, c_norm.T)[0]
            probe_clusters = np.argsort(centroid_scores)[-self.nprobe :][::-1]
        else:
            diff = self.centroids - query
            centroid_distances = np.linalg.norm(diff, axis=1)
            probe_clusters = np.argsort(centroid_distances)[: self.nprobe]

        # Collect candidates from selected clusters
        candidates_vecs = []
        candidates_ids = []
        for c in probe_clusters:
            cluster_id = int(c)
            if cluster_id in self.inverted_lists:
                for vec, idx in self.inverted_lists[cluster_id]:
                    candidates_vecs.append(vec)
                    candidates_ids.append(idx)

        if not candidates_vecs:
            return SearchResult(ids=np.array([]), distances=np.array([]))

        candidates = np.array(candidates_vecs, dtype=np.float32)
        candidate_ids = np.array(candidates_ids)

        # Brute force over candidates
        if self.metric == "cosine":
            q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
            c_norm = candidates / (
                np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10
            )
            scores = np.dot(q_norm, c_norm.T)[0]
            distances = 1.0 - scores
        else:
            diff = candidates - query
            distances = np.linalg.norm(diff, axis=1)

        top_k_idx = np.argsort(distances)[:k]
        return SearchResult(
            ids=candidate_ids[top_k_idx], distances=distances[top_k_idx]
        )

    def __len__(self) -> int:
        return len(self._all_ids)


class PQIndex:
    """
    Product Quantization — memory-efficient compressed vectors.

    Splits each vector into M sub-vectors, quantizes each sub-vector
    independently using k-means. Storage: M × log2(kbits) bits per vector.

    For 128-dim vectors with M=8, kbits=256: 8 bytes per vector
    (vs 512 bytes for float32).
    """

    def __init__(self, dim: int, M: int = 8, kbits: int = 256):
        assert dim % M == 0, f"dim ({dim}) must be divisible by M ({M})"
        self.dim = dim
        self.M = M
        self.kbits = min(kbits, 256)
        self.dsub = dim // M
        self.codebooks: list[np.ndarray] = []  # M codebooks, each (kbits, dsub)
        self.codes: Optional[np.ndarray] = None  # (n, M) integer codes
        self.ids: Optional[np.ndarray] = None
        self._trained = False

    def train(self, vectors: np.ndarray):
        """Train M codebooks using k-means on sub-vectors."""
        vectors = np.asarray(vectors, dtype=np.float32)
        n = len(vectors)

        self.codebooks = []
        for m in range(self.M):
            sub_vectors = vectors[:, m * self.dsub : (m + 1) * self.dsub]
            # Simple k-means
            rng = np.random.default_rng(42 + m)
            indices = rng.choice(n, min(self.kbits, n), replace=False)
            centroids = sub_vectors[indices].copy()

            for _ in range(10):
                diff = (
                    sub_vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]
                )
                distances = np.linalg.norm(diff, axis=2)
                assignments = np.argmin(distances, axis=1)
                for k in range(min(self.kbits, n)):
                    mask = assignments == k
                    if mask.sum() > 0:
                        centroids[k] = sub_vectors[mask].mean(axis=0)

            self.codebooks.append(centroids)

        self._trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors into PQ codes."""
        if not self._trained:
            raise RuntimeError("Must call train() before encode()")

        vectors = np.asarray(vectors, dtype=np.float32)
        n = len(vectors)
        codes = np.zeros((n, self.M), dtype=np.uint8)

        for m in range(self.M):
            sub_vectors = vectors[:, m * self.dsub : (m + 1) * self.dsub]
            diff = (
                sub_vectors[:, np.newaxis, :]
                - self.codebooks[m][np.newaxis, :, :]
            )
            distances = np.linalg.norm(diff, axis=2)
            codes[:, m] = np.argmin(distances, axis=1).astype(np.uint8)

        return codes

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """Add vectors to the PQ index."""
        if not self._trained:
            raise RuntimeError("Must call train() before add()")

        new_codes = self.encode(vectors)
        if ids is None:
            ids = np.arange(
                len(vectors)
                if self.codes is None
                else len(self.codes) + len(vectors)
            )

        if self.codes is None:
            self.codes = new_codes
            self.ids = np.asarray(ids)
        else:
            self.codes = np.vstack([self.codes, new_codes])
            self.ids = np.concatenate([self.ids, np.asarray(ids)])

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode PQ codes back to approximate vectors."""
        vectors = np.zeros((len(codes), self.dim), dtype=np.float32)
        for m in range(self.M):
            vectors[:, m * self.dsub : (m + 1) * self.dsub] = self.codebooks[m][
                codes[:, m]
            ]
        return vectors

    def search(
        self, query: np.ndarray, k: int = 10, metric: str = "l2"
    ) -> SearchResult:
        """Search using ADC (Asymmetric Distance Computation)."""
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)

        # Precompute distance to each codeword for each sub-quantizer
        dist_tables = []
        for m in range(self.M):
            q_sub = query[:, m * self.dsub : (m + 1) * self.dsub]
            if metric == "cosine":
                q_norm = q_sub / (
                    np.linalg.norm(q_sub, axis=1, keepdims=True) + 1e-10
                )
                c_norm = self.codebooks[m] / (
                    np.linalg.norm(self.codebooks[m], axis=1, keepdims=True)
                    + 1e-10
                )
                dist_table = 1.0 - np.dot(q_norm, c_norm.T)[0]
            else:
                diff = self.codebooks[m] - q_sub
                dist_table = np.linalg.norm(diff, axis=1)
            dist_tables.append(dist_table)

        # Compute approximate distances
        distances = np.zeros(len(self.codes), dtype=np.float32)
        for m in range(self.M):
            distances += dist_tables[m][self.codes[:, m]]

        top_k = np.argsort(distances)[:k]
        return SearchResult(ids=self.ids[top_k], distances=distances[top_k])

    def memory_usage(self) -> str:
        """Report memory usage vs float32."""
        n = len(self.codes) if self.codes is not None else 0
        pq_bytes = n * self.M * 1  # uint8 codes
        float32_bytes = n * self.dim * 4
        ratio = pq_bytes / float32_bytes * 100 if float32_bytes > 0 else 0
        return (
            f"PQ: {pq_bytes:,} bytes, "
            f"float32: {float32_bytes:,} bytes, "
            f"compression: {ratio:.1f}%"
        )

    def __len__(self) -> int:
        return len(self.codes) if self.codes is not None else 0


# === Convenience factory ===

class VectorIndex:
    """Factory for different index types."""

    def __init__(
        self,
        dim: int,
        index_type: str = "bruteforce",
        metric: str = "cosine",
        **kwargs,
    ):
        if index_type == "bruteforce":
            self.index = BruteForceIndex(dim, metric)
        elif index_type == "ivf":
            self.index = IVFIndex(dim, metric=metric, **kwargs)
        elif index_type == "pq":
            self.index = PQIndex(dim, **kwargs)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def __getattr__(self, name):
        return getattr(self.index, name)


# === Demo ===

def demo():
    """Demonstrate all three index types."""
    print("=" * 60)
    print("  SimSearch — Vector Similarity Search Demo")
    print("=" * 60)

    # Generate synthetic data
    dim = 64
    n_vectors = 10000
    n_query = 5

    rng = np.random.default_rng(42)
    base_vectors = rng.normal(0, 1, (n_vectors, dim)).astype(np.float32)
    base_vectors = base_vectors / np.linalg.norm(
        base_vectors, axis=1, keepdims=True
    )
    query_vectors = rng.normal(0, 1, (n_query, dim)).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(
        query_vectors, axis=1, keepdims=True
    )

    # 1. Brute Force
    print("\n1. BruteForceIndex (exact)")
    bf = BruteForceIndex(dim, metric="cosine")
    bf.add(base_vectors)
    result_bf = bf.search(query_vectors[0], k=5)
    print(f"   Index size: {len(bf):,} vectors")
    print(f"   Top-5 distances: {result_bf.distances}")

    # 2. IVF
    print("\n2. IVFIndex (approximate, nlist=50, nprobe=5)")
    ivf = IVFIndex(dim, nlist=50, nprobe=5, metric="cosine")
    ivf.train(base_vectors[:5000])
    ivf.add(base_vectors, ids=np.arange(n_vectors))
    result_ivf = ivf.search(query_vectors[0], k=5)
    print(f"   Index size: {len(ivf):,} vectors")
    print(f"   Top-5 distances: {result_ivf.distances}")

    # Compare with exact
    bf_neighbors = set(result_bf.ids[:5])
    ivf_neighbors = set(result_ivf.ids[:5])
    recall = len(bf_neighbors & ivf_neighbors) / 5
    print(f"   Recall@5: {recall:.1%}")

    # 3. PQ
    print("\n3. PQIndex (compressed, M=8, kbits=256)")
    pq = PQIndex(dim, M=8, kbits=256)
    pq.train(base_vectors[:5000])
    pq.add(base_vectors)
    result_pq = pq.search(query_vectors[0], k=5)
    print(f"   Index size: {len(pq):,} vectors")
    print(f"   {pq.memory_usage()}")
    print(f"   Top-5 distances: {result_pq.distances}")


if __name__ == "__main__":
    demo()
