from __future__ import annotations

import re
import unicodedata

import pandas as pd


def normalize_speaker_col(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .replace(r"^\s*$", pd.NA, regex=True)
        .ffill()
        .fillna("Unknown")
        .map(lambda x: unicodedata.normalize("NFKC", str(x)).strip())
    )


def normalize_transcript_col(series: pd.Series) -> pd.Series:
    def clean(x) -> str:
        if pd.isna(x):
            return ""
        s = str(x)
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u200b", "")
        s = s.replace("\xa0", " ")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    return series.map(clean)


def build_prev_lines(df: pd.DataFrame, idx: int, k: int) -> list[str]:
    start = max(0, idx - k)
    prev = df.loc[start : idx - 1, "Transcript"].astype(str)
    prev = [t.strip() for t in prev if str(t).strip()]
    return prev[-k:]
