"""Pilot diagnostics exercise only known-positive comparisons, never false positives."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

from ptg.classify.codebook import DEFAULT_CODEBOOK_PATH
from ptg.storage.raw_results import RawResult, write_raw_results


def test_pilot_report_is_aggregate_and_denominator_is_known_positives(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "scripts/pilot_diagnostics.py"
    spec = importlib.util.spec_from_file_location("pilot_diagnostics", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    workbook = tmp_path / "private.xlsx"
    private_text = "PRIVATE_TRANSCRIPT_DO_NOT_EMIT"
    pd.DataFrame({"Speaker": ["A", "B", "C"],
                  "Transcript": [private_text, private_text, private_text],
                  "Reconciled Codes": ["Initiative", None, "Communication"]}).to_excel(
        workbook, sheet_name="Team 1 Time 1 (S26)", index=False)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "trial_index_manifest.json").write_text('{"synthetic":true}')
    result_dir = tmp_path / "study21_pilot"
    result_dir.mkdir()
    rows = [RawResult(i, "00:00", "A", private_text, label, value, value)
            for i, answers in enumerate((("yes", "no"), ("no", "yes"), ("no", "no")))
            for label, value in zip(("Initiative", "Optimism"), answers)]
    write_raw_results(str(result_dir / "trial_raw_results.xlsx"),
                      "Team 1 Time 1 (S26) Run 0", rows)
    config = tmp_path / "study.yaml"
    config.write_text(
        f'io:\n  index_artifacts_dir: "{index_dir}"\n  artifacts_dir: "{tmp_path / "results"}"\n'
        f'excel:\n  workbook: "{workbook}"\n'
        '  sheet_template: "Team {team} Time {meet} ({semester})"\n'
        f'codebook:\n  path: "{DEFAULT_CODEBOOK_PATH.resolve()}"\n'
        '  active_labels: [Initiative, Optimism]\n'
        'retrieval:\n  run_prefix: trial_\n'
        'embedding:\n  model: synthetic-embedding\n'
        'llm:\n  model: synthetic-lm\n')
    report = module.diagnose(str(config), "S26", 1, 1, result_dir)
    total = report["totals"]
    assert total["jobs"] == 6 and total["selected_rows"] == 3
    assert total["known_human_positive_label_instances"] == 1
    assert total["model_yes_on_known_positive"] == 1
    assert total["rows_coded_only_outside_active_arm"] == 1
    assert total["valid_yes"] == 2  # Includes an unreviewed row, not a proven FP.
    assert private_text not in json.dumps(report)
    assert "precision" in report["warning"] and "NOT accuracy" in report["warning"]
