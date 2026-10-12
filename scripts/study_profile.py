"""Privacy-safe aggregate profile of the coded meeting workbook.

This tool reads the local workbook and emits only *aggregate* study statistics:
sheet/period inventory, per-label support under the clean codebook, counts of
coded-but-blank-transcript cells, unknown-only rows, and provenance (source
workbook hash + codebook version). It never prints or emits raw transcripts,
speakers, row ids, timestamps, or individual coded examples.

Study analysis excludes every ``(S25)`` sheet. The researcher confirms that
blanks after reconciliation mean reviewed no-code; wholly uncoded FA26 and
other sheets without confirmed reconciliation remain unannotated.

Usage (writes aggregate JSON + Markdown; source data stays local)::

    uv run python scripts/study_profile.py \
        --workbook data/MM_Team_Meeting_1_Coded.xlsx \
        --output-dir artifacts/study_profile

``--json-out`` / ``--markdown-out`` override the filenames inside
``--output-dir``. With no output flags the aggregate JSON is written to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ptg.classify.codebook import load_default_codebook
from ptg.domain.models import Codebook
from ptg.util.codes import parse_codes

# Periods excluded from every study aggregate (2025 spring sheets are out of scope).
EXCLUDED_PERIODS: tuple[str, ...] = ("S25",)

# Filenames used when only --output-dir is supplied.
DEFAULT_JSON_NAME = "meeting-data-profile.json"
DEFAULT_MARKDOWN_NAME = "meeting-data-profile.md"

CODE_COLUMN = "Reconciled Codes"
TRANSCRIPT_COLUMN = "Transcript"

PERIOD_RE = re.compile(r"\(([^()]+)\)\s*$")
TEAM_MEET_RE = re.compile(r"Team\s+(\d+)\s+Time\s+(\d+)", re.IGNORECASE)

_STAT_KEYS = (
    "rows",
    "coded_cells",
    "coded_nonempty_transcript",
    "recognized_rows",
    "unknown_only_rows",
    "blank_code_cells",
    "canonical_label_instances",
    "unknown_label_instances",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def period_of(sheet_name: str) -> str | None:
    match = PERIOD_RE.search(sheet_name)
    return match.group(1).strip() if match else None


def team_meet_of(sheet_name: str) -> tuple[int | None, int | None]:
    match = TEAM_MEET_RE.search(sheet_name)
    if not match:
        return (None, None)
    return (int(match.group(1)), int(match.group(2)))


def _nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _empty_stats() -> dict[str, int]:
    return {key: 0 for key in _STAT_KEYS}


def _profile_sheet(
    df: pd.DataFrame,
    *,
    codebook: Codebook,
    period: str,
    unknown_labels: Counter[str],
    label_support: dict[str, Counter[str]],
) -> tuple[dict[str, int], dict[str, bool]]:
    """Aggregate one sheet. Returns (counters, column-presence flags)."""
    columns = set(df.columns)
    has_codes = CODE_COLUMN in columns
    has_transcript = TRANSCRIPT_COLUMN in columns

    code_cells = df[CODE_COLUMN].tolist() if has_codes else [None] * len(df)
    transcript_nonempty = (
        _nonempty(df[TRANSCRIPT_COLUMN]).tolist() if has_transcript else [False] * len(df)
    )

    stats = _empty_stats()
    stats["rows"] = int(len(df))

    for cell, text_present in zip(code_cells, transcript_nonempty):
        labels = parse_codes(cell)
        if not labels:
            continue
        stats["coded_cells"] += 1
        canonical = [codebook.canonicalize_label(label) for label in labels]
        recognized = [c for c in canonical if c is not None]
        if recognized:
            stats["recognized_rows"] += 1
        else:
            stats["unknown_only_rows"] += 1
        if text_present:
            stats["coded_nonempty_transcript"] += 1
        stats["canonical_label_instances"] += len(recognized)
        for label, mapped in zip(labels, canonical):
            if mapped is None:
                stats["unknown_label_instances"] += 1
                unknown_labels[label] += 1
            else:
                label_support[mapped][period] += 1

    stats["blank_code_cells"] = stats["rows"] - stats["coded_cells"]
    flags = {"has_code_column": has_codes, "has_transcript_column": has_transcript}
    return stats, flags


def build_profile(workbook: str | Path, *, codebook: Codebook | None = None) -> dict[str, Any]:
    """Build a deterministic, aggregate-only profile of the coded workbook."""
    workbook = Path(workbook)
    if codebook is None:
        codebook = load_default_codebook()

    sheet_names = pd.ExcelFile(workbook).sheet_names
    study_sheets: list[tuple[str, str]] = []
    non_study_sheets: list[str] = []
    excluded_sheet_count = 0
    for name in sheet_names:
        period = period_of(name)
        if period is None:
            non_study_sheets.append(name)
            continue
        if period in EXCLUDED_PERIODS:
            excluded_sheet_count += 1
            continue
        study_sheets.append((name, period))

    unknown_labels: Counter[str] = Counter()
    label_support: dict[str, Counter[str]] = {entry.label: Counter() for entry in codebook.entries}

    period_stats: dict[str, dict[str, int]] = {}
    sheet_rows: dict[str, list[dict[str, Any]]] = {}
    totals = _empty_stats()
    sheets_missing_code_column = 0

    for name, period in study_sheets:
        df = pd.read_excel(workbook, sheet_name=name)
        sheet_stats, flags = _profile_sheet(
            df, codebook=codebook, period=period,
            unknown_labels=unknown_labels, label_support=label_support,
        )
        bucket = period_stats.setdefault(period, _empty_stats())
        for key in _STAT_KEYS:
            bucket[key] += sheet_stats[key]
            totals[key] += sheet_stats[key]
        if not flags["has_code_column"]:
            sheets_missing_code_column += 1
        team, meet = team_meet_of(name)
        sheet_rows.setdefault(period, []).append(
            {
                "sheet": name,
                "team": team,
                "meet": meet,
                "rows": sheet_stats["rows"],
                "coded_cells": sheet_stats["coded_cells"],
                "coded_nonempty_transcript": sheet_stats["coded_nonempty_transcript"],
                "recognized_rows": sheet_stats["recognized_rows"],
                "unknown_only_rows": sheet_stats["unknown_only_rows"],
                "blank_code_cells": sheet_stats["blank_code_cells"],
            }
        )

    ordered_periods = sorted(period_stats)
    labels: dict[str, Any] = {}
    for entry in codebook.entries:
        by_period = {p: int(label_support[entry.label].get(p, 0)) for p in ordered_periods}
        labels[entry.label] = {
            "id": entry.id,
            "category": entry.category,
            "support": sum(by_period.values()),
            "by_period": by_period,
        }

    fa26_coded = period_stats.get("FA26", _empty_stats())["coded_cells"]

    return {
        "provenance": {
            "generator": "scripts/study_profile.py",
            "workbook_filename": workbook.name,
            "workbook_sha256": sha256_file(workbook),
            "codebook_version": codebook.version,
            "codebook_label_count": len(codebook.entries),
            "study_periods": ordered_periods,
            "excluded_periods": list(EXCLUDED_PERIODS),
            "excluded_sheet_count": excluded_sheet_count,
            "non_study_sheets": sorted(non_study_sheets),
            "non_study_sheet_count": len(non_study_sheets),
            "sheets_missing_code_column": sheets_missing_code_column,
        },
        "caveats": [
            "Aggregate-only profile: no transcripts, speakers, row ids or individual "
            "examples are emitted.",
            "Researcher confirmed blanks after reconciliation mean reviewed no-code; "
            "this does not apply to wholly uncoded or unreconciled sheets such as FA26.",
            "Coded cells are rows with >=1 parsed code; a row may carry several codes.",
            "Unknown-only rows carry codes absent from the clean codebook and are excluded "
            "from per-label support.",
            "FA26 has no reconciled-code ground truth and cannot be scored until "
            "independently annotated.",
        ],
        "totals": totals,
        "fa26_ground_truth": {
            "present": bool(fa26_coded),
            "coded_cells": int(fa26_coded),
        },
        "periods": {
            period: {
                **period_stats[period],
                "sheet_count": len(sheet_rows.get(period, [])),
                "sheets": sheet_rows.get(period, []),
            }
            for period in ordered_periods
        },
        "labels": labels,
        "unknown_labels": dict(sorted(unknown_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def render_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, ensure_ascii=False) + "\n"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(profile: dict[str, Any]) -> str:
    prov = profile["provenance"]
    totals = profile["totals"]
    periods = profile["periods"]

    out: list[str] = ["# Meeting data profile", ""]
    out.append(
        "Aggregate-only inventory of reconciled codes in the coded meeting workbook, "
        "prepared for the F25 retrieval / S26 development / FA26 prospective-evaluation split. "
        "No transcripts, speakers, row ids or individual examples are included."
    )
    out += ["", "## Provenance", ""]
    out.append(f"- Source workbook: `{prov['workbook_filename']}`")
    out.append(f"- Workbook SHA-256: `{prov['workbook_sha256']}`")
    out.append(
        f"- Codebook version: `{prov['codebook_version']}` "
        f"({prov['codebook_label_count']} labels)"
    )
    out.append(f"- Study periods: {', '.join(prov['study_periods'])}")
    out.append(f"- Excluded periods: {', '.join(prov['excluded_periods']) or 'none'} "
               f"({prov['excluded_sheet_count']} sheets dropped)")
    out.append(f"- Non-study sheets skipped: {prov['non_study_sheet_count']}")
    out.append(f"- Sheets without a `{CODE_COLUMN}` column: {prov['sheets_missing_code_column']}")

    out += ["", "## Period summary", ""]
    headers = ["Period", "Sheets", "Rows", "Coded cells", "Coded w/ transcript",
               "Recognized rows", "Unknown-only rows", "Blank cells"]
    rows = [
        [p, periods[p]["sheet_count"], periods[p]["rows"], periods[p]["coded_cells"],
         periods[p]["coded_nonempty_transcript"], periods[p]["recognized_rows"],
         periods[p]["unknown_only_rows"], periods[p]["blank_code_cells"]]
        for p in prov["study_periods"]
    ]
    rows.append(["**Total**", sum(periods[p]["sheet_count"] for p in prov["study_periods"]),
                 totals["rows"], totals["coded_cells"], totals["coded_nonempty_transcript"],
                 totals["recognized_rows"], totals["unknown_only_rows"],
                 totals["blank_code_cells"]])
    out.append(_md_table(headers, rows))

    out += ["", "## Source sheet groupings", ""]
    for p in prov["study_periods"]:
        names = ", ".join(s["sheet"] for s in periods[p]["sheets"])
        out.append(f"- **{p}** ({periods[p]['sheet_count']} sheets): {names}")

    out += ["", "## Per-label support (study total)", ""]
    label_rows = [
        [label, info["category"], info["support"],
         " ".join(f"{k}:{v}" for k, v in info["by_period"].items() if v)]
        for label, info in profile["labels"].items()
    ]
    out.append(_md_table(["Label", "Category", "Support", "By period"], label_rows))

    out += ["", "## Unknown labels (excluded from support)", ""]
    if profile["unknown_labels"]:
        out.append(_md_table(["Label", "Instances"],
                             [[k, v] for k, v in profile["unknown_labels"].items()]))
    else:
        out.append("_None._")

    out += ["", "## FA26 ground truth", ""]
    fa26 = profile["fa26_ground_truth"]
    present = "yes" if fa26["present"] else "no"
    out.append(f"- Present: {present} ({fa26['coded_cells']} coded cells).")
    if not fa26["present"]:
        out.append("- FA26 rows are unannotated; no accuracy, agreement or F1 can be scored.")

    out += ["", "## Caveats", ""]
    out += [f"- {caveat}" for caveat in profile["caveats"]]
    return "\n".join(out) + "\n"


def write_outputs(
    profile: dict[str, Any],
    output_dir: Path | None,
    json_out: Path | None,
    markdown_out: Path | None,
) -> list[Path]:
    written: list[Path] = []
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if json_out is None and output_dir is not None:
        json_out = output_dir / DEFAULT_JSON_NAME
    if markdown_out is None and output_dir is not None:
        markdown_out = output_dir / DEFAULT_MARKDOWN_NAME
    if json_out is not None:
        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(render_json(profile), encoding="utf-8")
        written.append(json_out)
    if markdown_out is not None:
        markdown_out = Path(markdown_out)
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(profile), encoding="utf-8")
        written.append(markdown_out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True,
                        help="Coded workbook to profile (local, private).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Directory for {DEFAULT_JSON_NAME} and {DEFAULT_MARKDOWN_NAME}.")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Explicit JSON output path (overrides --output-dir name).")
    parser.add_argument("--markdown-out", type=Path, default=None,
                        help="Explicit Markdown output path (overrides --output-dir name).")
    args = parser.parse_args(argv)

    profile = build_profile(args.workbook)
    written = write_outputs(profile, args.output_dir, args.json_out, args.markdown_out)
    if not written:
        # No output paths: emit aggregate JSON to stdout (still privacy-safe).
        print(render_json(profile), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())