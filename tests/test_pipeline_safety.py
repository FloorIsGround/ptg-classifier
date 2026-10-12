"""Synthetic regression checks for index provenance, retrieval, and embedding alignment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ptg.classify.codebook import DEFAULT_CODEBOOK_PATH
from ptg.index import embed
from ptg.pipeline import run


def test_retrieved_rows_ignores_faiss_padding():
    rows = run._retrieved_rows(
        np.array([0.9, 0.8, -1e30, -1e30]),
        np.array([1, 0, -1, -1]),
        [{"id": "a"}, {"id": "b"}],
    )
    assert [row[0]["id"] for row in rows] == ["b", "a"]


def test_indexed_target_rejected():
    metas = pd.DataFrame({"sheet": ["Team 1 Time 1 (F25)"]})
    with pytest.raises(ValueError, match="label leakage"):
        run._assert_target_not_indexed(metas, "Team 1 Time 1 (F25)")
    run._assert_target_not_indexed(metas, "Team 1 Time 1 (S26)")


def test_query_cache_fingerprint_tracks_text_and_model():
    baseline = run._query_cache_fingerprint(["first", "second"], "embed-a")
    assert baseline == run._query_cache_fingerprint(["first", "second"], "embed-a")
    assert baseline != run._query_cache_fingerprint(["first", "changed"], "embed-a")
    assert baseline != run._query_cache_fingerprint(["first", "second"], "embed-b")


def test_batch_embedding_length_mismatch_retries_positionally(monkeypatch):
    class Client:
        def embed(self, *, model, input):
            if isinstance(input, list):
                return {"embeddings": [[1.0, 0.0]]}  # invalid: two requested
            return {"embeddings": [[1.0, 0.0] if input == "a" else [0.0, 1.0]]}

    monkeypatch.setattr(embed, "get_ollama_client", lambda: Client())
    vectors, good = embed.embed_many(["a", "b"], "synthetic")
    assert good == [0, 1]
    assert vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="Empty embedding input"):
        embed.embed_many(["a", "\u200b"], "synthetic")


def _settings(tmp_path):
    workbook = tmp_path / "source.xlsx"
    pd.DataFrame(
        {"Speaker": ["A", "B"], "Transcript": ["a sentence", "b sentence"],
         "Reconciled Codes": ["Emotional awareness", "Communication"]}
    ).to_excel(workbook, sheet_name="Team 1 Time 1 (F25)", index=False)
    return SimpleNamespace(
        workbook_path=workbook,
        index_artifacts_dir=tmp_path / "index",
        run_prefix="trial_",
        codebook_path=DEFAULT_CODEBOOK_PATH,
        sheet_template="Team {team} Time {meet} ({semester})",
        embed_model="synthetic-embed",
        prev_k=2,
    )


def test_index_fails_closed_on_unknown_labels(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(run, "embed_many", lambda *a, **k: (np.array([[1.0, 0.0]]), [0]))
    with pytest.raises(ValueError, match="Communication"):
        run.build_index(settings, semester="F25", teams=[1], meets=[1])
    assert not run._index_manifest_path(settings).exists()


def test_explicit_communication_exclusion_never_enters_core_retrieval(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.excluded_source_labels = ("Communication", "Service Orientation")
    monkeypatch.setattr(run, "embed_many", lambda texts, **k: (
        np.array([[1.0, 0.0]]), [0]))
    run.build_index(settings, semester="F25", teams=[1], meets=[1])
    index, metas = run._load_artifacts(settings)
    assert index.index.ntotal == 1
    assert metas.row_index.tolist() == [0]
    assert metas.codes.tolist() == [["Emotional awareness"]]
    import json
    manifest = json.loads(run._index_manifest_path(settings).read_text())
    assert manifest["excluded_only_rows"] == 1
    assert manifest["excluded_label_instances"] == 1


def test_index_manifest_binds_workbook_and_model(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    frame = pd.read_excel(settings.workbook_path, sheet_name="Team 1 Time 1 (F25)")
    frame.loc[1, "Reconciled Codes"] = "Questions to group"
    frame.to_excel(settings.workbook_path, sheet_name="Team 1 Time 1 (F25)", index=False)
    monkeypatch.setattr(run, "embed_many", lambda texts, **k: (np.array([[1.0, 0.0], [0.0, 1.0]]), [0, 1]))
    run.build_index(settings, semester="F25", teams=[1], meets=[1])
    index, metas = run._load_artifacts(settings)
    assert index.index.ntotal == len(metas) == 2
    assert metas["row_index"].tolist() == [0, 1]
    frame.loc[0, "Transcript"] = "changed sentence"
    frame.to_excel(settings.workbook_path, sheet_name="Team 1 Time 1 (F25)", index=False)
    with pytest.raises(ValueError, match="provenance mismatch"):
        run._load_artifacts(settings)
    settings.embed_model = "different-model"
    with pytest.raises(ValueError, match="provenance mismatch"):
        run._load_artifacts(settings)
