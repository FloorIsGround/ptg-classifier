"""Local-only synthetic tests for per-label checkpoints and interrupted inference."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ptg.classify.codebook import DEFAULT_CODEBOOK_PATH
from ptg.pipeline import run
from ptg.storage.checkpoints import CheckpointStore
from ptg.storage.raw_results import RawResult, read_raw_results


def test_checkpoint_store_is_atomic_and_fingerprint_bound(tmp_path):
    path = tmp_path / "partial.sqlite3"
    item = RawResult(3, "00:01", "A", "synthetic line", "Emotional awareness", "yes", "yes")
    with CheckpointStore(path, fingerprint="data-A", run_sheet="Run 0") as store:
        store.append([item])
        assert store.completed_keys() == {(3, "Emotional awareness")}
    with CheckpointStore(path, fingerprint="data-A", run_sheet="Run 1") as store:
        assert store.run_sheet == "Run 0"
        assert store.all_results() == [item]
        with pytest.raises(sqlite3.IntegrityError):
            store.append([item])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CheckpointStore(path, fingerprint="data-B", run_sheet="Run 1")


def _fixture(tmp_path, monkeypatch):
    workbook = tmp_path / "synthetic.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({
            "Speaker": ["A"], "Transcript": ["indexed example"],
            "Reconciled Codes": ["Emotional awareness"],
        }).to_excel(writer, sheet_name="Team 1 Time 1 (F25)", index=False)
        pd.DataFrame({
            "Speaker": ["B"] * 9,
            "Transcript": [f"target line {i}" for i in range(9)],
            "Reconciled Codes": [None] * 9,
        }).to_excel(writer, sheet_name="Team 1 Time 1 (S26)", index=False)
    settings = SimpleNamespace(
        workbook_path=workbook,
        index_artifacts_dir=tmp_path / "index",
        artifacts_dir=tmp_path / "results",
        run_prefix="trial_",
        codebook_path=DEFAULT_CODEBOOK_PATH,
        active_labels=("Emotional awareness", "Questions to group"),
        sheet_template="Team {team} Time {meet} ({semester})",
        embed_model="synthetic-embed",
        prev_k=2,
        retrieval_top_k=5,
        lm_api_base="http://127.0.0.1:1/v1",
        lm_model="synthetic-lm",
        binary_max_retries=0,
    )
    monkeypatch.setattr(
        run, "embed_many",
        lambda texts, **kwargs: (
            np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts), 1)),
            list(range(len(texts))),
        ),
    )
    run.build_index(settings, semester="F25", teams=[1], meets=[1])
    return settings


def test_interrupted_infer_resumes_only_unfinished_jobs(tmp_path, monkeypatch):
    settings = _fixture(tmp_path, monkeypatch)
    calls = []

    async def classify(*, jobs, **kwargs):
        calls.append([job.row_index for job in jobs])
        if len(calls) == 2:
            raise RuntimeError("synthetic interruption")
        return {(job.row_index, job.label_name): ("yes", "yes") for job in jobs}

    monkeypatch.setattr(run, "classify_binary_jobs", classify)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run.infer_sheet(settings, team=1, meet=1, semester="S26")
    assert not (settings.artifacts_dir / "trial_raw_results.xlsx").exists()
    checkpoints = list(settings.artifacts_dir.glob("*_checkpoint.sqlite3"))
    assert len(checkpoints) == 1
    with sqlite3.connect(checkpoints[0]) as conn:
        assert conn.execute("SELECT count(*) FROM results").fetchone()[0] == 16
    run.infer_sheet(settings, team=1, meet=1, semester="S26")
    assert len(calls) == 3
    assert len(calls[-1]) == 2
    raw = read_raw_results(str(settings.artifacts_dir / "trial_raw_results.xlsx"),
                           "Team 1 Time 1 (S26)")
    assert len(raw) == 18
    assert set(raw["parsed"]) == {"yes"}
    assert not list(settings.artifacts_dir.glob("*_checkpoint.sqlite3"))
    assert len(list(settings.artifacts_dir.glob("*.done.sqlite3"))) == 1


def test_embed_failure_is_auditable_not_silent(tmp_path, monkeypatch):
    settings = _fixture(tmp_path, monkeypatch)

    def partial_embed(texts, **kwargs):
        return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts) - 1, 1)), list(range(len(texts) - 1))

    monkeypatch.setattr(run, "embed_many", partial_embed)

    async def classify(*, jobs, **kwargs):
        return {(job.row_index, job.label_name): ("no", "no") for job in jobs}

    monkeypatch.setattr(run, "classify_binary_jobs", classify)
    run.infer_sheet(settings, team=1, meet=1, semester="S26", embed_batch_size=9)
    raw = read_raw_results(str(settings.artifacts_dir / "trial_raw_results.xlsx"),
                           "Team 1 Time 1 (S26)")
    assert len(raw) == 18
    failed = raw[raw["parsed"].isna() | raw["parsed"].eq("")]
    assert len(failed) == 2
    assert failed["raw_response"].astype(str).str.contains("__EMBED_ERROR__").all()


def test_pilot_limit_isolated_and_fingerprinted(tmp_path, monkeypatch):
    settings = _fixture(tmp_path, monkeypatch)

    async def classify(*, jobs, **kwargs):
        return {(job.row_index, job.label_name): ("no", "no") for job in jobs}

    monkeypatch.setattr(run, "classify_binary_jobs", classify)
    with pytest.raises(ValueError, match="dedicated pilot"):
        run.infer_sheet(settings, team=1, meet=1, semester="S26", pilot_row_limit=2)
    settings.artifacts_dir = tmp_path / "study21_pilot"
    with pytest.raises(ValueError, match="positive"):
        run.infer_sheet(settings, team=1, meet=1, semester="S26", pilot_row_limit=0)
    run.infer_sheet(settings, team=1, meet=1, semester="S26", pilot_row_limit=2)
    raw = read_raw_results(str(settings.artifacts_dir / "trial_raw_results.xlsx"),
                           "Team 1 Time 1 (S26)")
    assert len(raw) == 4
    assert set(raw["row_index"]) == {0, 1}
    assert run._run_fingerprint(settings, "Team 1 Time 1 (S26)", 120.0, [0, 1]) != (
        run._run_fingerprint(settings, "Team 1 Time 1 (S26)", 120.0, [0, 1, 2]))
