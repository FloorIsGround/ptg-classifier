"""Summarize a local pilot without exposing transcript text or asserting accuracy.

Good: uv run python scripts/pilot_diagnostics.py --config config/study-21.yaml --team 1 --meet 1 --semester S26 --result-dir artifacts/study21_pilot50 --out artifacts/study21_pilot50/diagnostics.json
Bad: reporting this positive-only convenience sample as accuracy, precision, or F1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ptg.classify.codebook import load_codebook
from ptg.config import load_settings
from ptg.domain.models import SheetRef
from ptg.ingest.excel import load_sheet_df, normalize_sheet_df
from ptg.storage.raw_results import read_raw_results
from ptg.util.codes import parse_codes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnose(config: str, semester: str, team: int, meet: int, result_dir: Path) -> dict:
    """Compare *known human positives only* against one complete pilot run."""
    settings = load_settings(config)
    if "pilot" not in result_dir.name.casefold():
        raise ValueError("This diagnostic requires an isolated pilot result directory")
    sheet = SheetRef(team=team, meet=meet, semester=semester)
    name, source = load_sheet_df(str(settings.workbook_path), sheet, settings.sheet_template)
    source = normalize_sheet_df(source)
    raw_path = result_dir / f"{settings.run_prefix}raw_results.xlsx"
    raw = read_raw_results(str(raw_path), name)
    if raw.empty or raw.run_id.nunique() != 1:
        raise ValueError("Exactly one complete pilot run must exist for this sheet")
    required = {"row_index", "label", "parsed", "run_id"}
    if not required <= set(raw.columns):
        raise ValueError(f"Raw run lacks required columns: {sorted(required - set(raw.columns))}")
    selected = sorted(set(raw.row_index.astype(int)))
    if not selected or any(i < 0 or i >= len(source) for i in selected):
        raise ValueError("Pilot row indices are outside the source sheet")
    active = set(settings.active_labels)
    expected = {(i, label) for i in selected for label in settings.active_labels}
    keys = list(zip(raw.row_index.astype(int), raw.label.astype(str)))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Pilot has missing, duplicate, or unexpected per-label results")
    parsed = {(int(r.row_index), str(r.label)): str(r.parsed).strip().lower()
              for r in raw.itertuples(index=False)}
    codebook = load_codebook(settings.codebook_path)
    positive = Counter()
    detected = Counter()
    missed = Counter()
    unresolved = Counter()
    predicted = Counter()
    coded_rows = 0
    excluded_code_rows = 0
    for row_index in selected:
        codes = parse_codes(source.at[row_index, "Reconciled Codes"])
        coded_rows += bool(codes)
        canonical = {label for code in codes
                     if (label := codebook.canonicalize_label(code)) is not None}
        excluded_code_rows += bool(codes and not (canonical & active))
        for label in canonical & active:
            positive[label] += 1
            value = parsed[(row_index, label)]
            if value == "yes":
                detected[label] += 1
            elif value == "no":
                missed[label] += 1
            else:
                unresolved[label] += 1
        for label in active:
            if parsed[(row_index, label)] == "yes":
                predicted[label] += 1
    status = Counter(parsed.values())
    labels = {label: {"known_human_positive": positive[label],
                      "model_yes_on_known_positive": detected[label],
                      "model_no_on_known_positive": missed[label],
                      "unresolved_on_known_positive": unresolved[label],
                      "model_yes_all_pilot_rows": predicted[label]}
              for label in settings.active_labels}
    return {
        "warning": "Convenience sample; this diagnostic intentionally ignores blank-cell review status. "
                   "Use evaluate_reconciled.py for an explicitly reconciled complete sheet. "
                   "These are positive-support diagnostics, NOT accuracy, precision, F1, or final evaluation.",
        "provenance": {"sheet": name, "workbook_sha256": _sha256(settings.workbook_path),
                       "raw_results_sha256": _sha256(raw_path), "config": config,
                       "index_manifest_sha256": _sha256(settings.index_artifacts_dir /
                                                        f"{settings.run_prefix}index_manifest.json"),
                       "model": settings.lm_model, "embed_model": settings.embed_model,
                       "run_id": int(raw.run_id.iloc[0]), "selection": "first N nonempty transcript rows"},
        "totals": {"selected_rows": len(selected), "active_labels": len(active), "jobs": len(raw),
                   "valid_yes": status["yes"], "valid_no": status["no"],
                   "invalid_or_failed": len(raw) - status["yes"] - status["no"],
                   "rows_with_any_human_code": coded_rows,
                   "rows_coded_only_outside_active_arm": excluded_code_rows,
                   "known_human_positive_label_instances": sum(positive.values()),
                   "model_yes_on_known_positive": sum(detected.values()),
                   "model_no_on_known_positive": sum(missed.values()),
                   "unresolved_on_known_positive": sum(unresolved.values())},
        "labels": labels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--semester", required=True)
    parser.add_argument("--team", required=True, type=int)
    parser.add_argument("--meet", required=True, type=int)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = diagnose(args.config, args.semester, args.team, args.meet, args.result_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "totals": payload["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
