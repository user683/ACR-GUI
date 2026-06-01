from typing import Optional

import numpy as np


class FaissOrNumpyIndex:
    """Cosine/IP index using FAISS when available, numpy otherwise."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.index = None
        self.faiss = self._try_import_faiss()

    @staticmethod
    def _try_import_faiss():
        try:
            import faiss

            return faiss
        except Exception:
            return None

    @property
    def uses_faiss(self) -> bool:
        return self.faiss is not None

    def build(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {vectors.shape[1]}")

        self.vectors = vectors
        if self.faiss is None:
            self.index = None
            return

        self.index = self.faiss.IndexFlatIP(self.dim)
        if len(vectors) > 0:
            self.index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if top_k <= 0 or len(self.vectors) == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"expected query dim={self.dim}, got {query.shape[1]}")

        top_k = min(top_k, len(self.vectors))
        if self.index is not None:
            scores, indices = self.index.search(query, top_k)
            return scores[0], indices[0]

        scores = self.vectors @ query[0]
        indices = np.argsort(-scores)[:top_k]
        return scores[indices], indices.astype(np.int64)
