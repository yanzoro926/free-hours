"""Quick tests for SimSearch."""
import unittest
import numpy as np
from simsearch import BruteForceIndex, IVFIndex, PQIndex


class TestBruteForce(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.idx = BruteForceIndex(self.dim, metric="cosine")
        rng = np.random.default_rng(42)
        self.vectors = rng.normal(0, 1, (100, self.dim)).astype(np.float32)
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.idx.add(self.vectors)

    def test_len(self):
        self.assertEqual(len(self.idx), 100)

    def test_search(self):
        query = self.vectors[0].copy()
        result = self.idx.search(query, k=5)
        self.assertEqual(len(result.ids), 5)
        # The first result should be the query itself (distance ~0)
        self.assertAlmostEqual(result.distances[0], 0.0, places=3)

    def test_l2_metric(self):
        idx = BruteForceIndex(self.dim, metric="l2")
        idx.add(self.vectors)
        result = idx.search(self.vectors[0], k=3)
        self.assertAlmostEqual(result.distances[0], 0.0, places=3)


class TestIVF(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        rng = np.random.default_rng(42)
        self.vectors = rng.normal(0, 1, (500, self.dim)).astype(np.float32)
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def test_train_and_add(self):
        idx = IVFIndex(self.dim, nlist=10, nprobe=5, metric="cosine")
        idx.train(self.vectors)
        idx.add(self.vectors)
        self.assertEqual(len(idx), 500)

    def test_search(self):
        idx = IVFIndex(self.dim, nlist=10, nprobe=5, metric="cosine")
        idx.train(self.vectors)
        idx.add(self.vectors)
        result = idx.search(self.vectors[0], k=3)
        self.assertEqual(len(result.ids), 3)
        # Should find the query itself (or a very close neighbor)
        self.assertLess(result.distances[0], 0.5)


class TestPQ(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.M = 4
        rng = np.random.default_rng(42)
        self.vectors = rng.normal(0, 1, (500, self.dim)).astype(np.float32)

    def test_train_encode_add(self):
        pq = PQIndex(self.dim, M=self.M, kbits=64)
        pq.train(self.vectors)
        pq.add(self.vectors)
        self.assertEqual(len(pq), 500)

    def test_encode_decode(self):
        pq = PQIndex(self.dim, M=self.M, kbits=64)
        pq.train(self.vectors)
        codes = pq.encode(self.vectors[:10])
        self.assertEqual(codes.shape, (10, self.M))
        decoded = pq.decode(codes)
        self.assertEqual(decoded.shape, (10, self.dim))

    def test_search(self):
        pq = PQIndex(self.dim, M=self.M, kbits=64)
        pq.train(self.vectors)
        pq.add(self.vectors)
        result = pq.search(self.vectors[0], k=3)
        self.assertEqual(len(result.ids), 3)

    def test_memory_usage(self):
        pq = PQIndex(self.dim, M=self.M, kbits=64)
        pq.train(self.vectors)
        pq.add(self.vectors)
        usage = pq.memory_usage()
        self.assertIn("PQ:", usage)
        self.assertIn("float32:", usage)


if __name__ == "__main__":
    unittest.main()
