"""Score the 21-label arm on explicitly reconciled 2026 sheets.

Good: evaluate the S26/SP26 sheets that underwent reconciliation, with
--affirm-reviewed-blanks after the researcher confirms blank cells mean no code.
Bad: include FA26 or wholly uncoded sheets as all-negative ground truth.
No transcript, speaker, row ID, or example is emitted in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ptg.classify.codebook import load_codebook
from ptg.config import Settings, load_settings
from ptg.domain.models import SheetRef
from ptg.ingest.excel import load_sheet_df, normalize_sheet_df
from ptg.pipeline.run import _load_artifacts, _run_fingerprint
from ptg.storage.raw_results import read_raw_results
from ptg.util.codes import canonicalize_codes, parse_codes

RECONCILED_SHEETS = frozenset({("S26", 1, 1), ("SP26", 2, 1)})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(tp: int, fp: int, fn: int, tn: int, *, support: int,
             abstained_positive: int = 0) -> dict:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn + abstained_positive)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn + abstained_positive) if support else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": support,
            "precision": precision, "recall": recall, "f1": f1}


def _verify_completed_run(settings: Settings, name: str, raw: pd.DataFrame,
                          row_indices: list[int], lm_timeout: float) -> str:
    """Bind exported predictions to the model/index/prompt fingerprint in SQLite."""
    run_id = int(raw.run_id.iloc[0])
    token = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    done = settings.artifacts_dir / f"{settings.run_prefix}{token}_checkpoint.{run_id}.done.sqlite3"
    if not done.is_file():
        raise ValueError(f"{name}: completed checkpoint missing for Run {run_id}")
    with sqlite3.connect(f"{done.resolve().as_uri()}?mode=ro", uri=True) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        expected = _run_fingerprint(settings, name, lm_timeout, row_indices)
        if metadata.get("fingerprint") != expected or metadata.get("run_sheet") != f"{name} Run {run_id}":
            raise ValueError(f"{name}: model/index/prompt provenance mismatch")
        ledger = {(int(i), label): (text, response, parsed)
                  for i, label, text, response, parsed in connection.execute(
                      "SELECT row_index, label, transcript, raw_response, parsed FROM results")}

    def cell(value: object) -> str:
        return "" if pd.isna(value) else str(value)

    exported = {(int(row.row_index), str(row.label)):
                (str(row.transcript), cell(row.raw_response), cell(row.parsed))
                for row in raw.itertuples(index=False)}
    if exported != ledger:
        raise ValueError(f"{name}: exported predictions differ from completed checkpoint")
    return _sha256(done)


def score_one(settings: Settings, semester: str, team: int, meet: int, *,
              verify_ledger: bool = False, lm_timeout: float = 120.0) -> dict:
    """Return aggregate confusion counts on one declared reconciled sheet."""
    if (semester, team, meet) not in RECONCILED_SHEETS:
        raise ValueError("Only the two explicitly reconciled S26/SP26 development sheets can be scored")
    sheet_ref = SheetRef(team=team, meet=meet, semester=semester)
    name, source = load_sheet_df(str(settings.workbook_path), sheet_ref, settings.sheet_template)
    if "Reconciled Codes" not in source:
        raise ValueError(f"{name}: missing reconciled codes column")
    source = normalize_sheet_df(source)
    nonempty = {i for i, text in enumerate(source["Transcript"]) if str(text).strip()}
    if not nonempty:
        raise ValueError(f"{name}: no transcript rows")
    book = load_codebook(settings.codebook_path)
    excluded = {" ".join(x.casefold().split()) for x in settings.excluded_source_labels}
    if not any(canonicalize_codes(
            [code for code in parse_codes(cell)
             if " ".join(code.casefold().split()) not in excluded], codebook=book)
            for cell in source["Reconciled Codes"]):
        raise ValueError(f"{name}: no approved code evidence of reconciliation")
    raw_path = settings.artifacts_dir / f"{settings.run_prefix}raw_results.xlsx"
    raw = read_raw_results(str(raw_path), name)
    if raw.empty or raw.run_id.nunique() != 1:
        raise ValueError(f"{name}: require exactly one complete, isolated inference run")
    keys = list(zip(raw["row_index"].astype(int), raw["label"].astype(str)))
    expected = {(i, label) for i in nonempty for label in settings.active_labels}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"{name}: raw results have missing, extra, or duplicate row-label pairs")
    # A raw Excel sheet has no embedded source hash. Cross-check every row text
    # against the active workbook before accepting it as predictions for this run.
    for row in raw.itertuples(index=False):
        if str(row.transcript).strip() != str(source.at[int(row.row_index), "Transcript"]).strip():
            raise ValueError(f"{name}: raw result transcript differs from workbook")
    checkpoint_digest = (_verify_completed_run(settings, name, raw, sorted(nonempty), lm_timeout)
                         if verify_ledger else None)
    gold: dict[int, set[str]] = {}
    blank_reviewed = 0
    excluded_only = 0
    for idx in nonempty:
        raw_codes = parse_codes(source.at[idx, "Reconciled Codes"])
        blank_reviewed += not raw_codes
        retained = [code for code in raw_codes
                    if " ".join(code.casefold().split()) not in excluded]
        excluded_only += bool(raw_codes and not retained)
        gold[idx] = set(canonicalize_codes(retained, codebook=book))
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                      "abstained_positive": 0, "abstained_negative": 0}
              for label in settings.active_labels}
    for row in raw.itertuples(index=False):
        idx = int(row.row_index)
        label = str(row.label)
        actual = label in gold[idx]
        prediction = str(row.parsed).strip().casefold()
        cell = counts[label]
        if prediction not in ("yes", "no"):
            cell["abstained_positive" if actual else "abstained_negative"] += 1
        elif prediction == "yes":
            cell["tp" if actual else "fp"] += 1
        else:
            cell["fn" if actual else "tn"] += 1
    by_label = {}
    for label, cell in counts.items():
        support = cell["tp"] + cell["fn"] + cell["abstained_positive"]
        by_label[label] = {**cell, **_metrics(cell["tp"], cell["fp"], cell["fn"], cell["tn"],
                                           support=support,
                                           abstained_positive=cell["abstained_positive"])}
    total = {key: sum(cell[key] for cell in counts.values())
             for key in ("tp", "fp", "fn", "tn", "abstained_positive", "abstained_negative")}
    total["rows"] = len(nonempty)
    total["jobs"] = len(raw)
    total["reviewed_blank_rows"] = blank_reviewed
    total["excluded_only_coded_rows"] = excluded_only
    total["run_id"] = int(raw.run_id.iloc[0])
    summary: dict[str, int | float | None] = dict(total)
    summary["micro_f1"] = _metrics(total["tp"], total["fp"], total["fn"], total["tn"],
                                  support=total["tp"] + total["fn"] + total["abstained_positive"],
                                  abstained_positive=total["abstained_positive"])["f1"]
    supported_f1 = [v["f1"] for v in by_label.values() if v["support"]]
    summary["macro_f1_supported_labels"] = float(np.mean(supported_f1)) if supported_f1 else None
    summary["prediction_coverage"] = _ratio(total["tp"] + total["fp"] + total["fn"] + total["tn"],
                                          total["jobs"])
    return {"sheet": name, "totals": summary, "labels": by_label,
            "checkpoint_sha256": checkpoint_digest}


def evaluate(config: str, specs: list[tuple[str, int, int]], *, reviewed_blanks: bool,
             lm_timeout: float = 120.0) -> dict:
    if not reviewed_blanks:
        raise ValueError("Researcher must affirm blank cells were reviewed no-code after reconciliation")
    if not specs or len(specs) != len(set(specs)):
        raise ValueError("Select distinct reconciled development sheets")
    settings = load_settings(config)
    _load_artifacts(settings)  # verify workbook, codebook, model and source-index hashes
    sheets = [score_one(settings, *spec, verify_ledger=True, lm_timeout=lm_timeout)
              for spec in specs]
    fields = ("tp", "fp", "fn", "tn", "abstained_positive", "abstained_negative")
    totals = {key: sum(item["totals"][key] for item in sheets) for key in fields}
    totals["rows"] = sum(item["totals"]["rows"] for item in sheets)
    totals["jobs"] = sum(item["totals"]["jobs"] for item in sheets)
    totals["reviewed_blank_rows"] = sum(item["totals"]["reviewed_blank_rows"] for item in sheets)
    totals["excluded_only_coded_rows"] = sum(item["totals"]["excluded_only_coded_rows"] for item in sheets)
    overall = _metrics(totals["tp"], totals["fp"], totals["fn"], totals["tn"],
                       support=totals["tp"] + totals["fn"] + totals["abstained_positive"],
                       abstained_positive=totals["abstained_positive"])
    all_labels = {}
    for label in settings.active_labels:
        cell = {key: sum(item["labels"][label][key] for item in sheets) for key in fields}
        support = cell["tp"] + cell["fn"] + cell["abstained_positive"]
        all_labels[label] = {**cell, **_metrics(cell["tp"], cell["fp"], cell["fn"], cell["tn"],
                                               support=support,
                                               abstained_positive=cell["abstained_positive"])}
    supported = [v["f1"] for v in all_labels.values() if v["support"] and v["f1"] is not None]
    summary: dict[str, int | float | None] = dict(totals)
    summary["micro_precision"] = overall["precision"]
    summary["micro_recall"] = overall["recall"]
    summary["micro_f1"] = overall["f1"]
    summary["macro_f1_supported_labels"] = float(np.mean(supported)) if supported else None
    summary["supported_label_count"] = len(supported)
    summary["prediction_coverage"] = _ratio(totals["tp"] + totals["fp"] + totals["fn"] + totals["tn"],
                                            totals["jobs"])
    return {
        "provenance": {"config": config, "workbook_sha256": _sha256(settings.workbook_path),
                       "codebook_sha256": _sha256(settings.codebook_path),
                       "index_manifest_sha256": _sha256(settings.index_artifacts_dir /
                                                        f"{settings.run_prefix}index_manifest.json"),
                       "raw_results_sha256": _sha256(settings.artifacts_dir /
                                                     f"{settings.run_prefix}raw_results.xlsx"),
                       "model": settings.lm_model, "embedding_model": settings.embed_model,
                       "lm_timeout": lm_timeout,
                       "active_labels": list(settings.active_labels),
                       "excluded_source_labels": list(settings.excluded_source_labels),
                       "reviewed_blanks_attested": True,
                       "scope": "only explicitly named reconciled development sheets"},
        "warning": "S26/SP26 development scoring, not independent FA26 evaluation. "
                   "Macro F1 excludes labels with zero human-positive support; "
                   "invalid predictions are abstentions, not discarded. "
                   "Pooled scores are descriptive until S26/SP26 cohort mapping is confirmed. "
                   "The blank-as-negative policy applies only to selected reconciled sheets.",
        "sheets": sheets, "totals": summary, "labels": all_labels,
    }


def _parse_specs(text: str) -> list[tuple[str, int, int]]:
    out = []
    for part in text.split(","):
        semester, team, meet = part.strip().split(":")
        out.append((semester.strip(), int(team), int(meet)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sheets", required=True,
                        help="comma-separated semester:team:meet, e.g. S26:1:1,SP26:2:1")
    parser.add_argument("--affirm-reviewed-blanks", action="store_true")
    parser.add_argument("--lm-timeout", type=float, default=120.0,
                        help="Match --lm-timeout used during inference (fingerprint check)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = evaluate(args.config, _parse_specs(args.sheets),
                       reviewed_blanks=args.affirm_reviewed_blanks,
                       lm_timeout=args.lm_timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "totals": payload["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
