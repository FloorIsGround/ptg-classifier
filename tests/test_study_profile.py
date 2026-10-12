from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_profile_module():
    spec = importlib.util.spec_from_file_location(
        "study_profile", Path(__file__).resolve().parents[1] / "scripts/study_profile.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIVATE = "PRIVATE_TRANSCRIPT_XYZ"
SPEAKER = "SPEAKERID_XYZ"


def _row(speaker: object = SPEAKER, transcript: object = PRIVATE, codes: object = "Optimism"):
    return {"Speaker": speaker, "Transcript": transcript, "Reconciled Codes": codes}


def _write_synthetic_workbook(path: Path) -> None:
    """Workbook covering recognized/unknown labels, blank transcript, blank codes, S25."""
    with pd.ExcelWriter(path) as writer:
        # F25: one recognized row, one blank-transcript coded row, one unknown-only,
        # one alias, one blank-code (negative-looking) row.
        pd.DataFrame(
            [
                _row(codes="Optimism"),
                _row(transcript=None, codes="Empathy"),
                _row(codes="Communication"),  # unknown-only under clean codebook
                _row(codes="Question to Group"),  # alias -> Questions to group
                _row(transcript=PRIVATE, codes=None),
            ]
        ).to_excel(writer, sheet_name="Team 1 Time 1 (F25)", index=False)
        # S25 must be excluded entirely.
        pd.DataFrame([_row(codes="Optimism"), _row(codes="Empathy")]).to_excel(
            writer, sheet_name="Team 1 Time 1 (S25)", index=False
        )
        pd.DataFrame([_row(codes="Adaptability")]).to_excel(
            writer, sheet_name="Team 2 Time 1 (S26)", index=False
        )
        pd.DataFrame([_row(codes="Initiative"), _row(codes="Communication")]).to_excel(
            writer, sheet_name="Team 1 Time 1 (SP26)", index=False
        )
        # FA26: rows present, zero codes.
        pd.DataFrame(
            [_row(codes=None), _row(codes=None)]
        ).to_excel(writer, sheet_name="Team 1 Time 1 (FA26)", index=False)
        pd.DataFrame({"junk": [1, 2]}).to_excel(writer, sheet_name="Totals", index=False)


@pytest.fixture()
def workbook(tmp_path: Path) -> Path:
    path = tmp_path / "private.xlsx"
    _write_synthetic_workbook(path)
    return path


def test_blank_transcript_coded_rows_are_counted_separately(workbook):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    f25 = profile["periods"]["F25"]
    # 4 coded cells, one of which has no transcript text.
    assert f25["coded_cells"] == 4
    assert f25["coded_nonempty_transcript"] == 3


def test_unknown_only_rows_and_unknown_labels(workbook):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    f25 = profile["periods"]["F25"]
    assert f25["unknown_only_rows"] == 1
    assert f25["unknown_label_instances"] == 1
    assert "Communication" in profile["unknown_labels"]


def test_alias_canonicalized_to_clean_label(workbook):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    assert profile["labels"]["Questions to group"]["support"] == 1
    # Unknown code must not appear in per-label support.
    assert all(entry["support"] == 0 for label, entry in profile["labels"].items()
               if label == "Communication")


def test_s25_excluded_from_study_analysis(workbook):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    assert "S25" not in profile["provenance"]["study_periods"]
    assert "S25" not in profile["periods"]
    assert profile["provenance"]["excluded_sheet_count"] == 1
    # S25 rows (2 coded) must not be folded into totals: 4+1+2 = 7 study coded cells.
    assert profile["totals"]["coded_cells"] == 7
    assert not any("S25" in sheet["sheet"] for p in profile["periods"].values()
                   for sheet in p["sheets"])


def test_fa26_has_zero_ground_truth(workbook):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    fa26 = profile["periods"]["FA26"]
    assert fa26["rows"] == 2
    assert fa26["coded_cells"] == 0
    assert fa26["recognized_rows"] == 0
    assert fa26["unknown_only_rows"] == 0
    assert profile["fa26_ground_truth"] == {"present": False, "coded_cells": 0}


def test_blank_code_cells_reported_not_treated_as_negatives(workbook):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    f25 = profile["periods"]["F25"]
    assert f25["rows"] == 5
    assert f25["blank_code_cells"] == 1
    assert any("after reconciliation" in caveat and "FA26" in caveat
               for caveat in profile["caveats"])


def test_no_private_text_in_json_or_markdown(workbook):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    payload = module.render_json(profile) + module.render_markdown(profile)
    assert PRIVATE not in payload
    assert SPEAKER not in payload
    # Sanity: recognized aggregate labels are allowed through.
    assert "Optimism" in payload


def test_reporting_is_deterministic(workbook, tmp_path):
    module = load_profile_module()
    first = module.build_profile(workbook)
    second = module.build_profile(workbook)
    assert first == second
    assert module.render_json(first) == module.render_json(second)
    assert module.render_markdown(first) == module.render_markdown(second)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    module.write_outputs(first, out_a, None, None)
    module.write_outputs(second, out_b, None, None)
    assert (out_a / module.DEFAULT_MARKDOWN_NAME).read_bytes() == (
        out_b / module.DEFAULT_MARKDOWN_NAME
    ).read_bytes()
    assert (out_a / module.DEFAULT_JSON_NAME).read_bytes() == (
        out_b / module.DEFAULT_JSON_NAME
    ).read_bytes()


def test_json_and_markdown_share_same_aggregates(workbook, tmp_path):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    out = tmp_path / "out"
    module.write_outputs(profile, out, None, None)
    json_path = out / module.DEFAULT_JSON_NAME
    md_path = out / module.DEFAULT_MARKDOWN_NAME
    assert json_path.exists() and md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["totals"] == profile["totals"]
    md = md_path.read_text(encoding="utf-8")
    assert str(profile["periods"]["F25"]["coded_cells"]) in md
    assert PRIVATE not in md and SPEAKER not in md


def test_cli_writes_explicit_outputs(workbook, tmp_path):
    module = load_profile_module()
    json_out = tmp_path / "nested" / "profile.json"
    md_out = tmp_path / "nested" / "profile.md"
    rc = module.main([
        "--workbook", str(workbook),
        "--json-out", str(json_out),
        "--markdown-out", str(md_out),
    ])
    assert rc == 0
    assert json_out.exists() and md_out.exists()
    assert PRIVATE not in md_out.read_text(encoding="utf-8")


def test_cli_output_dir_writes_default_filenames(workbook, tmp_path):
    module = load_profile_module()
    out = tmp_path / "profiles"
    rc = module.main(["--workbook", str(workbook), "--output-dir", str(out)])
    assert rc == 0
    assert (out / module.DEFAULT_JSON_NAME).exists()
    assert (out / module.DEFAULT_MARKDOWN_NAME).exists()


def test_cli_stdout_when_no_outputs(workbook, capsys):
    module = load_profile_module()
    rc = module.main(["--workbook", str(workbook)])
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["totals"]["coded_cells"] == 7
    assert PRIVATE not in captured and SPEAKER not in captured


def test_provenance_records_workbook_hash_and_codebook(workbook):
    module = load_profile_module()
    profile = module.build_profile(workbook)
    prov = profile["provenance"]
    assert prov["workbook_sha256"] == module.sha256_file(workbook)
    assert len(prov["workbook_sha256"]) == 64
    assert prov["codebook_label_count"] == 21
    assert prov["codebook_version"]
    assert prov["study_periods"] == sorted(prov["study_periods"])
