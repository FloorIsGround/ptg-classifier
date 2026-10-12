from __future__ import annotations

import unicodedata

import pandas as pd


def clean_text(x) -> str:
    """Normalize unicode, remove common zero-width and NBSP, strip."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "")
    s = s.replace("\ufeff", "")
    s = s.replace("\xa0", " ")
    return s.strip()
