"""Audit source/target sheet schema and label compatibility without reading out transcripts.

Loads the codebook configured for the active study arm (settings.codebook_path) and
blocks unrecognized labels other than explicit excluded_source_labels. Exclusions
are counted separately rather than silently treated as unknown source data.

Good: uv run python scripts/audit_workbook.py --config config/study-21.yaml
Bad: uv run ptg build-index ...  # silently drops unknown labels before review
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ptg.classify.codebook import load_codebook
from ptg.config import load_settings
from ptg.util.codes import parse_codes

DEFAULT_INDEX = "F25:1:1,F25:2:1,F25:1:2,F25:2:2"
DEFAULT_TARGETS = "FA26:1:1,FA26:2:1,FA26:3:1"


def parse_sheet_specs(value: str) -> list[tuple[str, int, int]]:
    specs: list[tuple[str, int, int]] = []
    for part in value.split(","):
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError(f"Expected semester:team:meet, got {part!r}")
        semester, team, meet = fields
        specs.append((semester, int(team), int(meet)))
    if not specs or len(specs) != len(set(specs)):
        raise ValueError("Sheet list must be nonempty and unique")
    return specs


def audit(config: Path, index: str = DEFAULT_INDEX, targets: str = DEFAULT_TARGETS) -> dict:
    settings = load_settings(config)
    specs = [("index", s) for s in parse_sheet_specs(index)] + [
        ("target", s) for s in parse_sheet_specs(targets)
    ]
    names = [settings.sheet_template.format(team=t, meet=m, semester=sem) for _, (sem, t, m) in specs]
    if len(names) != len(set(names)):
        raise ValueError("Index and target sheets must be disjoint")
    codebook = load_codebook(settings.codebook_path)
    excluded = {" ".join(label.casefold().split()) for label in settings.excluded_source_labels}
    workbook = settings.workbook_path
    xls = pd.ExcelFile(workbook)
    rows = []
    blockers = []
    for (purpose, _), name in zip(specs, names):
        if name not in xls.sheet_names:
            blockers.append(f"Missing sheet: {name}")
            continue
        df = pd.read_excel(xls, sheet_name=name)
        missing = sorted({"Transcript", "Speaker", "Reconciled Codes"} - set(df.columns))
        if missing:
            blockers.append(f"{name}: missing columns {missing}")
            continue
        codes = df["Reconciled Codes"].dropna()
        unknown: Counter[str] = Counter()
        excluded_counts: Counter[str] = Counter()
        usable = 0
        indexable = 0
        excluded_only = 0
        text_present = df["Transcript"].fillna("").astype(str).str.strip().ne("")
        for row_idx, cell in codes.items():
            labels = parse_codes(cell)
            retained = [label for label in labels
                        if " ".join(label.casefold().split()) not in excluded]
            excluded_counts.update(label for label in labels if label not in retained)
            canonical = [codebook.canonicalize_label(label) for label in retained]
            unknown.update(
                label for label, mapped in zip(retained, canonical) if mapped is None
            )
            if any(mapped is not None for mapped in canonical):
                usable += 1
                indexable += bool(text_present.loc[row_idx])
            elif labels and not retained:
                excluded_only += 1
        transcript_rows = int(text_present.sum())
        row = {"sheet": name, "purpose": purpose, "rows": len(df),
               "transcript_nonempty": transcript_rows, "coded_cells": len(codes),
               "recognized_coded_cells": usable, "indexable_rows": indexable,
               "excluded_only_rows": excluded_only,
               "excluded_labels": dict(excluded_counts.most_common()),
               "unknown_labels": dict(unknown.most_common())}
        rows.append(row)
        if unknown:
            blockers.append(f"{name}: {sum(unknown.values())} unknown label instances would be silently dropped")
        if purpose == "index" and not indexable:
            blockers.append(f"{name}: no recognized coded rows")
        if purpose == "target" and not transcript_rows:
            blockers.append(f"{name}: no transcript rows")
    return {"workbook": str(workbook), "codebook_path": str(settings.codebook_path),
            "codebook_version": codebook.version, "sheets": rows, "blockers": blockers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/latest-local.yaml"))
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    args = parser.parse_args()
    result = audit(args.config, args.index, args.targets)
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result["blockers"] else 0)
