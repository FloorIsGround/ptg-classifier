from __future__ import annotations

import numpy as np
import pandas as pd

from ptg.index.embed import embed_text
from ptg.storage.faiss_index import FaissIPIndex


def retrieve_top_k(
    query_text: str,
    index: FaissIPIndex,
    metas: pd.DataFrame,
    embed_model: str,
    k: int,
) -> list[dict]:
    qv = embed_text(query_text, model=embed_model)
    D, I = index.search(np.asarray(qv, dtype=np.float32), k)

    out = []
    for dist, idx in zip(D[0], I[0]):
        row = metas.iloc[int(idx)].to_dict()
        row["score"] = float(dist)
        out.append(row)
    return out
