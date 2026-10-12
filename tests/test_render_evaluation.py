"""Meeting report emits only aggregate whitelist fields."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_render_is_aggregate_only():
    path = Path(__file__).resolve().parents[1] / "scripts/render_evaluation.py"
    spec = importlib.util.spec_from_file_location("render_evaluation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = "PRIVATE_TRANSCRIPT_DO_NOT_EMIT"
    provenance = {"model": "synthetic-chat", "embedding_model": "synthetic-embed",
                  "workbook_sha256": "a" * 64, "codebook_sha256": "b" * 64,
                  "index_manifest_sha256": "c" * 64, "raw_results_sha256": "d" * 64,
                  "raw_transcripts": private}
    sheet = {"sheet": "Team 1 Time 1 (S26)", "totals": {"rows": 2, "tp": 1, "fp": 0,
             "fn": 0, "abstained_positive": 0, "micro_f1": 1.0,
             "macro_f1_supported_labels": 1.0, "prediction_coverage": 1.0}}
    total = {"rows": 2, "jobs": 42, "prediction_coverage": 1.0,
             "reviewed_blank_rows": 1, "excluded_only_coded_rows": 0,
             "tp": 1, "fp": 0, "fn": 0, "tn": 41,
             "abstained_positive": 0, "abstained_negative": 0,
             "micro_precision": 1.0, "micro_recall": 1.0, "micro_f1": 1.0,
             "macro_f1_supported_labels": 1.0, "supported_label_count": 1}
    label = {"support": 1, "tp": 1, "fp": 0, "fn": 0,
             "precision": 1.0, "recall": 1.0, "f1": 1.0,
             "private_example": private}
    text = module.render({"provenance": provenance, "sheets": [sheet], "totals": total,
                          "labels": {"Initiative": label}, "private_data": private})
    assert private not in text
    assert "FA26" in text and "Macro F1" in text and "S26" in text
    assert "development run (synthetic-chat)" in text
    assert "do not attribute them to another model" in text
    assert "All-no baseline" in text and "97.6%" in text
    assert "| Initiative | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |" in text
