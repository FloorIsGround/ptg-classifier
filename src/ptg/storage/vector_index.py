from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "VectorIndex": ...
