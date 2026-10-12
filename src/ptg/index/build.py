from __future__ import annotations

import uuid

import pandas as pd

from ptg.domain.models import IndexedMeta
from ptg.storage.faiss_index import FaissIPIndex


def write_metas(metas: list[IndexedMeta], path: str) -> None:
    pd.DataFrame([m.__dict__ for m in metas]).to_parquet(path)


def new_id() -> str:
    return str(uuid.uuid4())


def build_faiss_index(vectors, dim: int) -> FaissIPIndex:
    idx = FaissIPIndex(dim=dim)
    idx.add(vectors)
    return idx
