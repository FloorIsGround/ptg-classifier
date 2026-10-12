"""Render aggregate-only reconciled development metrics into meeting Markdown.

Good: render a verified evaluator JSON into docs/meeting-model-results.md.
Bad: copy row-level predictions, transcripts, or this report's pooled scores
into a claim of independent FA26 generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render(report: dict) -> str:
    """Use only aggregate whitelist fields; never reproduce arbitrary input text."""
    p = report["provenance"]
    t = report["totals"]
    lines = [
        f"# 21-label F25-index → S26/SP26 development run ({p['model']})",
        "",
        ("**Exploratory Qwen engineering baseline, not the planned Granite study arm or final evaluation.**"
         if p["model"] == "qwen3.5:4b"
         else "**Model-specific development run, not final evaluation.**"),
        "These scores belong only to the chat model named above; do not attribute them to another model.",
        "Retired Communication and Service Orientation are",
        "excluded from the 768-row F25 retrieval corpus and the 21 target labels. Blank codes",
        "count as reviewed no-code only on the two selected reconciled sheets. FA26 has no",
        "reconciled labels and is not scored.",
        "",
        "## Provenance",
        "",
        f"- Chat model: `{p['model']}`; embedding model: `{p['embedding_model']}`.",
        f"- Workbook SHA-256: `{p['workbook_sha256']}`.",
        f"- Codebook SHA-256: `{p['codebook_sha256']}`.",
        f"- F25 index manifest SHA-256: `{p['index_manifest_sha256']}`.",
        f"- Prediction workbook SHA-256: `{p['raw_results_sha256']}`.",
        "- Predictions are checked against each completed SQLite checkpoint's",
        "  model/index/prompt fingerprint and exact per-label results.",
        "",
        "## Coverage and confusion totals",
        "",
        f"- Transcript rows: **{t['rows']}**; binary label jobs: **{t['jobs']}**",
        f"  ({t['prediction_coverage']:.1%} resolved).",
        f"- Reviewed blank-code rows: **{t['reviewed_blank_rows']}**; rows coded only",
        f"  with retired labels: **{t['excluded_only_coded_rows']}** (kept in the negative denominator).",
        f"- TP {t['tp']}, FP {t['fp']}, FN {t['fn']}, TN {t['tn']};",
        f"  abstained positive {t['abstained_positive']}, negative {t['abstained_negative']}.",
        "",
        "## Summary metrics",
        "",
        "| Scope | Rows | Human-positive label instances | Micro P | Micro R | Micro F1 | Macro F1* | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["sheets"]:
        sheet = item["sheet"]
        s = item["totals"]
        support = s["tp"] + s["fn"] + s["abstained_positive"]
        precision = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else None
        recall = s["tp"] / support if support else None
        lines.append(f"| {sheet} | {s['rows']} | {support} | {_fmt(precision)} | {_fmt(recall)} | "
                     f"{_fmt(s['micro_f1'])} | {_fmt(s['macro_f1_supported_labels'])} | "
                     f"{s['prediction_coverage']:.1%} |")
    support = t["tp"] + t["fn"] + t["abstained_positive"]
    lines.append(f"| **Pooled descriptive** | **{t['rows']}** | **{support}** | "
                 f"**{_fmt(t['micro_precision'])}** | **{_fmt(t['micro_recall'])}** | "
                 f"**{_fmt(t['micro_f1'])}** | **{_fmt(t['macro_f1_supported_labels'])}** | "
                 f"**{t['prediction_coverage']:.1%}** |")
    all_no_accuracy = (t['jobs'] - support) / t['jobs']
    lines.extend([
        "",
        f"*Macro F1 averages the **{t['supported_label_count']}** labels with at least one",
        "human-positive example; zero-support labels are reported below but excluded from",
        "that macro mean. Invalid predictions are abstentions; positive abstentions",
        "penalize recall/F1. Pooled numbers are descriptive until the S26/SP26 cohort",
        "and participant mapping is confirmed.*",
        "",
        "**All-no baseline:** predicting no on every label-instance gives micro F1",
        f"**0.000** and label-instance accuracy **{all_no_accuracy:.1%}**. That high",
        "accuracy is a class-imbalance artifact, not a useful model-quality target.",
        "A no-retrieval model comparison has not been run.",
        "",
        "## Per-label aggregates (pooled development sheets)",
        "",
        "| Label | Support | TP | FP | FN | P | R | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, v in sorted(report["labels"].items(), key=lambda kv: (-kv[1]["support"], kv[0])):
        lines.append(f"| {label} | {v['support']} | {v['tp']} | {v['fp']} | {v['fn']} | "
                     f"{_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} |")
    lines.extend([
        "",
        "## Interpretation limits and next steps",
        "",
        "- These are development-sheet predictions. Do not present them as independent",
        "  out-of-year or FA26 accuracy. Error analysis and a controlled no-retrieval",
        "  comparison are needed before drawing stronger conclusions.",
        "- FA26's 2,039 transcript rows have no reconciled ground truth. Freeze the",
        "  model/index/codebook and obtain blinded coding before final scoring.",
        "- The old draft 23-label superset-index pilot used a different corpus and",
        "  cannot isolate a 21-vs-23 label effect.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.input_path.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(report), encoding="utf-8")
    print(f"Wrote aggregate-only report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
