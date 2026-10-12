import pandas as pd

from ptg.ingest.normalize import normalize_speaker_col, normalize_transcript_col


def test_normalize_speaker_ffill():
    s = pd.Series(["A", " ", None, "B"])
    out = normalize_speaker_col(s)
    assert out.iloc[1] == "A"
    assert out.iloc[2] == "A"
    assert out.iloc[3] == "B"


def test_normalize_transcript_strips():
    s = pd.Series(["  hi\u00a0there  ", None])
    out = normalize_transcript_col(s)
    assert out.iloc[0] == "hi there"
    assert out.iloc[1] == ""
