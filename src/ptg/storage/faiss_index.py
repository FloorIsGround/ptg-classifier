from __future__ import annotations

import faiss
import numpy as np


class FaissIPIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        X = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(X)
        self.index.add(X)

    def search(self, query: np.ndarray, k: int):
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        faiss.normalize_L2(q)
        return self.index.search(q, k)

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str) -> "FaissIPIndex":
        obj = cls(dim=1)
        obj.index = faiss.read_index(path)
        return obj
