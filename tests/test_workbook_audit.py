import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_workbook", Path(__file__).resolve().parents[1] / "scripts/audit_workbook.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_blocks_unknown_index_label_without_exposing_transcript(tmp_path):
    module = load_audit_module()
    workbook = tmp_path / "private.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"Speaker": ["Private"], "Transcript": ["Never print this"],
                      "Reconciled Codes": ["Communication"]}).to_excel(
                          writer, sheet_name="Team 1 Time 1 (F25)", index=False)
        pd.DataFrame({"Speaker": ["Private"], "Transcript": ["Never print that"],
                      "Reconciled Codes": [None]}).to_excel(
                          writer, sheet_name="Team 3 Time 1 (FA26)", index=False)
    config = tmp_path / "latest.yaml"
    config.write_text(
        f'excel:\n  workbook: "{workbook}"\n  sheet_template: "Team {{team}} Time {{meet}} ({{semester}})"\n'
    )
    result = module.audit(config, "F25:1:1", "FA26:3:1")
    assert result["blockers"] == [
        "Team 1 Time 1 (F25): 1 unknown label instances would be silently dropped",
        "Team 1 Time 1 (F25): no recognized coded rows",
    ]
    assert "Never print" not in str(result)
    assert result["sheets"][1]["coded_cells"] == 0


def test_audit_refuses_overlapping_index_and_target_sheets(tmp_path):
    module = load_audit_module()
    config = tmp_path / "latest.yaml"
    config.write_text('excel:\n  sheet_template: "Team {team} Time {meet} ({semester})"\n')
    with pytest.raises(ValueError, match="disjoint"):
        module.audit(config, "FA26:1:1", "FA26:1:1")
