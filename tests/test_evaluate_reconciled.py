"""Synthetic scoring checks: reviewed blanks, excluded Communication, and coverage."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ptg.classify.codebook import DEFAULT_CODEBOOK_PATH
from ptg.storage.checkpoints import CheckpointStore
from ptg.storage.raw_results import RawResult, read_raw_results, write_raw_results


def _load():
    spec = importlib.util.spec_from_file_location(
        "evaluate_reconciled", Path(__file__).resolve().parents[1] / "scripts/evaluate_reconciled.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    workbook = tmp_path / "private.xlsx"
    text = "PRIVATE_TRANSCRIPT_NOT_FOR_REPORT"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"Speaker": ["A", "B", "C"],
                      "Transcript": [text, text, text],
                      "Reconciled Codes": ["Initiative", None, "Communication"]}).to_excel(
            writer, sheet_name="Team 1 Time 1 (S26)", index=False)
        pd.DataFrame({"Speaker": ["A"], "Transcript": [text],
                      "Reconciled Codes": [None]}).to_excel(
            writer, sheet_name="Team 2 Time 1 (SP26)", index=False)
    results = tmp_path / "results"
    results.mkdir()
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "core_index_manifest.json").write_text('{"synthetic": true}')
    rows = [RawResult(i, "00:00", "A", text, label, value, value)
            for i, answers in enumerate((("yes", "yes"), ("yes", "no"), ("no", "no")))
            for label, value in zip(("Initiative", "Optimism"), answers)]
    write_raw_results(str(results / "core_raw_results.xlsx"),
                      "Team 1 Time 1 (S26) Run 0", rows)
    settings = SimpleNamespace(workbook_path=workbook, artifacts_dir=results,
                               index_artifacts_dir=index_dir,
                               run_prefix="core_", sheet_template="Team {team} Time {meet} ({semester})",
                               codebook_path=DEFAULT_CODEBOOK_PATH,
                               active_labels=("Initiative", "Optimism"),
                               excluded_source_labels=("Communication", "Service Orientation"),
                               embed_model="synthetic-embed", lm_model="synthetic-lm")
    return settings, text


def test_blank_reviewed_is_negative_but_communication_is_excluded(tmp_path):
    module = _load()
    settings, private_text = _fixture(tmp_path)
    report = module.score_one(settings, "S26", 1, 1)
    totals = report["totals"]
    assert totals["rows"] == 3 and totals["jobs"] == 6
    assert totals["reviewed_blank_rows"] == 1
    assert totals["excluded_only_coded_rows"] == 1
    assert (totals["tp"], totals["fp"], totals["fn"], totals["tn"]) == (1, 2, 0, 3)
    assert report["labels"]["Initiative"]["support"] == 1
    assert report["labels"]["Optimism"]["support"] == 0
    assert private_text not in json.dumps(report)


def test_reject_wholly_uncoded_sheet_and_unapproved_term(tmp_path):
    module = _load()
    settings, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="no approved code evidence"):
        module.score_one(settings, "SP26", 2, 1)
    with pytest.raises(ValueError, match="two explicitly reconciled"):
        module.score_one(settings, "FA26", 1, 1)
    with pytest.raises(ValueError, match="affirm"):
        module.evaluate("unused", [("S26", 1, 1)], reviewed_blanks=False)
    with pytest.raises(ValueError, match="two explicitly reconciled"):
        module.score_one(settings, "SP26", 3, 1)


def test_missing_prediction_fails_closed(tmp_path):
    module = _load()
    settings, private_text = _fixture(tmp_path)
    write_raw_results(str(settings.artifacts_dir / "core_raw_results.xlsx"),
                      "Team 1 Time 1 (S26) Run 0",
                      [RawResult(0, "00:00", "A", private_text, "Initiative", "yes", "yes")])
    with pytest.raises(ValueError, match="missing, extra, or duplicate"):
        module.score_one(settings, "S26", 1, 1)


def test_invalid_positive_prediction_penalizes_recall():
    module = _load()
    metrics = module._metrics(0, 0, 0, 1, support=1, abstained_positive=1)
    assert metrics["recall"] == 0 and metrics["f1"] == 0


@pytest.mark.parametrize("codes", [[" ", ",", None], ["Communication", None, None]])
def test_stray_or_retired_only_code_cannot_attest_reconciliation(tmp_path, codes):
    module = _load()
    settings, private_text = _fixture(tmp_path)
    with pd.ExcelWriter(settings.workbook_path, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        pd.DataFrame({"Speaker": ["A"] * 3, "Transcript": [private_text] * 3,
                      "Reconciled Codes": codes}).to_excel(
            writer, sheet_name="Team 1 Time 1 (S26)", index=False)
    with pytest.raises(ValueError, match="no approved code evidence"):
        module.score_one(settings, "S26", 1, 1)


def test_completed_checkpoint_binds_predictions_and_model(tmp_path, monkeypatch):
    module = _load()
    settings, private_text = _fixture(tmp_path)
    monkeypatch.setattr(module, "load_settings", lambda _: settings)
    monkeypatch.setattr(module, "_load_artifacts", lambda _: (None, None))
    monkeypatch.setattr(module, "_run_fingerprint", lambda *a: "synthetic-fingerprint")
    name = "Team 1 Time 1 (S26)"
    run_sheet = f"{name} Run 0"
    token = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    path = settings.artifacts_dir / f"core_{token}_checkpoint.sqlite3"
    responses = (("yes", "yes"), ("yes", "no"), ("no", "no"))
    rows = [RawResult(i, "00:00", "A", private_text, label, value, value)
            for i, answers in enumerate(responses)
            for label, value in zip(settings.active_labels, answers)]
    with CheckpointStore(path, fingerprint="synthetic-fingerprint", run_sheet=run_sheet) as store:
        store.append(rows)
    path.rename(path.with_name(f"{path.stem}.0.done.sqlite3"))
    report = module.evaluate("synthetic", [("S26", 1, 1)], reviewed_blanks=True)
    assert report["totals"]["rows"] == 3
    assert len(report["sheets"][0]["checkpoint_sha256"]) == 64
    assert "PRIVATE_TRANSCRIPT" not in json.dumps(report)
    assert "cohort mapping" in report["warning"]
    assert module._parse_specs("S26:1:1, SP26:2:1") == [("S26", 1, 1), ("SP26", 2, 1)]

    rows[0] = RawResult(0, "00:00", "A", private_text, "Initiative", "no", "no")
    write_raw_results(str(settings.artifacts_dir / "core_raw_results.xlsx"), run_sheet, rows)
    with pytest.raises(ValueError, match="differ from completed checkpoint"):
        module.evaluate("synthetic", [("S26", 1, 1)], reviewed_blanks=True)
    rows[0] = RawResult(0, "00:00", "A", private_text, "Initiative", "yes", "yes")
    write_raw_results(str(settings.artifacts_dir / "core_raw_results.xlsx"), run_sheet, rows)
    monkeypatch.setattr(module, "_run_fingerprint", lambda *a: "different-model")
    with pytest.raises(ValueError, match="provenance mismatch"):
        module.evaluate("synthetic", [("S26", 1, 1)], reviewed_blanks=True)
