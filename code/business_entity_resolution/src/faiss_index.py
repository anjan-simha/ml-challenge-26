"""
ML Challenge 2026: Business Entity Resolution
FAISS Vector Indexing & Search Module

Wraps FAISS for building, querying, and serializing dense vector similarity indexes.
"""

import os
import numpy as np


class FaissSimilarityIndex:
    """
    Manages building and querying a FAISS vector index using Inner Product (cosine similarity).
    """

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = None
        self.id_map = []  # Maps FAISS integer row index -> entity_id

    def build(self, vectors, entity_ids):
        """
        Build an IndexFlatIP index over L2-normalized float32 vectors.
        """
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
            vectors_np = np.ascontiguousarray(vectors, dtype=np.float32)
            self.index.add(vectors_np)
            self.id_map = list(entity_ids)
        except ImportError:
            print("Notice: FAISS not available in environment; using brute-force numpy fallback.", flush=True)
            self.index = np.ascontiguousarray(vectors, dtype=np.float32)
            self.id_map = list(entity_ids)

    def search(self, query_vectors, top_k=10):
        """
        Search for top_k most similar vectors for each query vector.
        
        Returns:
            list of list of candidate entity_ids, one list per query vector
        """
        query_np = np.ascontiguousarray(query_vectors, dtype=np.float32)
        try:
            import faiss
            if isinstance(self.index, faiss.Index):
                _, indices = self.index.search(query_np, top_k)
                results = []
                for row in indices:
                    cands = [self.id_map[idx] for idx in row if 0 <= idx < len(self.id_map)]
                    results.append(cands)
                return results
        except ImportError:
            pass

        # Fallback numpy cosine search
        results = []
        # Matrix multiply: queries x candidates
        scores = np.dot(query_np, self.index.T)
        for i in range(len(query_np)):
            top_indices = np.argsort(-scores[i])[:top_k]
            cands = [self.id_map[idx] for idx in top_indices if 0 <= idx < len(self.id_map)]
            results.append(cands)
        return results

    def save(self, filepath):
        """Serialize FAISS index to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        try:
            import faiss
            if isinstance(self.index, faiss.Index):
                faiss.write_index(self.index, filepath)
                return
        except Exception:
            pass
        np.save(filepath + ".npy", self.index)

    def load(self, filepath):
        """Load FAISS index from disk."""
        try:
            import faiss
            if os.path.exists(filepath):
                self.index = faiss.read_index(filepath)
                return
        except Exception:
            pass
        if os.path.exists(filepath + ".npy"):
            self.index = np.load(filepath + ".npy")
