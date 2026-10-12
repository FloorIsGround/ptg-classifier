from __future__ import annotations

import pandas as pd
import pytest

from ptg.export.aggregate import aggregate_runs

LABELS = ["Emotional awareness", "Adaptability", "Empathy"]


def _make_raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# K=1 tests
# ---------------------------------------------------------------------------

def test_k1_yes():
    raw = _make_raw([
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi",
         "label": "Emotional awareness", "raw_response": "yes", "parsed": "yes", "run_id": 0},
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi",
         "label": "Adaptability", "raw_response": "no", "parsed": "no", "run_id": 0},
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi",
         "label": "Empathy", "raw_response": "no", "parsed": "no", "run_id": 0},
    ])
    df = aggregate_runs(raw, LABELS)
    assert len(df) == 1
    assert df.at[0, "Emotional awareness"] == "yes"
    assert df.at[0, "Adaptability"] == "no"
    assert df.at[0, "Flags"] == ""


def test_k1_failed_label_flagged():
    raw = _make_raw([
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi",
         "label": "Emotional awareness", "raw_response": "", "parsed": "", "run_id": 0},
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi",
         "label": "Adaptability", "raw_response": "no", "parsed": "no", "run_id": 0},
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi",
         "label": "Empathy", "raw_response": "yes", "parsed": "yes", "run_id": 0},
    ])
    df = aggregate_runs(raw, LABELS)
    assert df.at[0, "Emotional awareness"] == ""
    assert df.at[0, "Flags"] == "Emotional awareness"


def test_k1_multiple_rows():
    raw = _make_raw([
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hello",
         "label": "Empathy", "raw_response": "yes", "parsed": "yes", "run_id": 0},
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hello",
         "label": "Adaptability", "raw_response": "no", "parsed": "no", "run_id": 0},
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hello",
         "label": "Emotional awareness", "raw_response": "no", "parsed": "no", "run_id": 0},
        {"row_index": 1, "timing": "00:05", "speaker": "B", "transcript": "bye",
         "label": "Empathy", "raw_response": "no", "parsed": "no", "run_id": 0},
        {"row_index": 1, "timing": "00:05", "speaker": "B", "transcript": "bye",
         "label": "Adaptability", "raw_response": "yes", "parsed": "yes", "run_id": 0},
        {"row_index": 1, "timing": "00:05", "speaker": "B", "transcript": "bye",
         "label": "Emotional awareness", "raw_response": "no", "parsed": "no", "run_id": 0},
    ])
    df = aggregate_runs(raw, LABELS)
    assert len(df) == 2
    assert df.at[0, "Empathy"] == "yes"
    assert df.at[1, "Adaptability"] == "yes"
    assert df.at[0, "speaker"] == "A"
    assert df.at[1, "speaker"] == "B"


# ---------------------------------------------------------------------------
# K>1 tests
# ---------------------------------------------------------------------------

def _make_k2_raw(label: str, parsed_run0: str, parsed_run1: str) -> pd.DataFrame:
    base = {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi"}
    other_labels = [l for l in LABELS if l != label]
    rows = [
        {**base, "label": label, "raw_response": parsed_run0, "parsed": parsed_run0, "run_id": 0},
        {**base, "label": label, "raw_response": parsed_run1, "parsed": parsed_run1, "run_id": 1},
    ]
    for l in other_labels:
        rows += [
            {**base, "label": l, "raw_response": "no", "parsed": "no", "run_id": 0},
            {**base, "label": l, "raw_response": "no", "parsed": "no", "run_id": 1},
        ]
    return pd.DataFrame(rows)


def test_k2_both_yes():
    df = aggregate_runs(_make_k2_raw("Empathy", "yes", "yes"), LABELS)
    assert df.at[0, "Empathy"] == pytest.approx(1.0)


def test_k2_both_no():
    df = aggregate_runs(_make_k2_raw("Empathy", "no", "no"), LABELS)
    assert df.at[0, "Empathy"] == pytest.approx(0.0)


def test_k2_split():
    df = aggregate_runs(_make_k2_raw("Empathy", "yes", "no"), LABELS)
    assert df.at[0, "Empathy"] == pytest.approx(0.5)


def test_k2_one_failed_one_yes():
    df = aggregate_runs(_make_k2_raw("Empathy", "", "yes"), LABELS)
    # valid_count = 1 (only "yes"), yes_count = 1 → 1.0
    assert df.at[0, "Empathy"] == pytest.approx(1.0)
    assert df.at[0, "Flags"] == ""


def test_k2_all_failed_flagged():
    df = aggregate_runs(_make_k2_raw("Empathy", "", ""), LABELS)
    assert df.at[0, "Empathy"] == ""
    assert "Empathy" in df.at[0, "Flags"]


# ---------------------------------------------------------------------------
# Column structure tests
# ---------------------------------------------------------------------------

def test_output_columns_order():
    raw = _make_raw([
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi",
         "label": l, "raw_response": "no", "parsed": "no", "run_id": 0}
        for l in LABELS
    ])
    df = aggregate_runs(raw, LABELS)
    expected_cols = ["timing", "speaker", "transcript"] + LABELS + ["Flags"]
    assert list(df.columns) == expected_cols


def test_empty_input_returns_empty_with_correct_columns():
    df = aggregate_runs(pd.DataFrame(), LABELS)
    assert df.empty
    expected_cols = ["timing", "speaker", "transcript"] + LABELS + ["Flags"]
    assert list(df.columns) == expected_cols


def test_flags_codebook_order():
    """Multiple failed labels should appear in codebook order in Flags."""
    raw = _make_raw([
        {"row_index": 0, "timing": "00:01", "speaker": "A", "transcript": "hi",
         "label": l, "raw_response": "", "parsed": "", "run_id": 0}
        for l in LABELS
    ])
    df = aggregate_runs(raw, LABELS)
    assert df.at[0, "Flags"] == ", ".join(LABELS)
