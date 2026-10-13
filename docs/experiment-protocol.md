# Matched dynamic retrieval comparison

Branch: `research/rag-vs-none`. Shared implementation and all admission gaps:
[development experiment queue](research-experiments.md).

## Question and controls

Does dynamic retrieval help classification on a frozen development population?
Compare `rag_baseline` and `no_retrieval` with the same rows, all 21 labels,
static definitions/examples, context, pinned local model/runtime, decoding,
parser and scoring. Removing dynamic retrieval does not remove static examples.
Do not compare an original private codebook arm with the public fictional one.

Prepare both arms together and interleave them deterministically. Repetition
names identify fresh inference; resume names identify recovery only. Define
static-example overlap handling before predictions, with no arm-specific row
filtering. Report both meetings separately and pooled descriptive results,
coverage, false positives/negatives, precision/recall/F1, kappa and paired changes.
Treat positive abstentions as missed positives for recall/F1. Report matched
human-coder comparisons separately from engine repeatability.

## Executable fictional smoke

```bash
uv sync --frozen
uv run --frozen python scripts/classifier_suite_v2.py prepare --synthetic --recipes rag_baseline no_retrieval --out artifacts/synthetic-rag
uv run --frozen python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-rag --repeat first
uv run --frozen python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-rag --repeat first --resume
uv run --frozen python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-rag --repeat second
uv run --frozen python scripts/classifier_suite_v2.py report --out artifacts/synthetic-rag --repeats first second
```

Use a fresh output directory for a new smoke. Synthetic agreement and counts
validate plumbing, not scientific accuracy. No engine or model is installed by
these commands.

## Admission before real inference

Version/review private input extraction, admit the tokenizer/runtime on
fictional requests, implement verified scientific and meeting-specific metrics,
and replace the whole-JSON journal before full-development execution. The
native 512-job limit is binding. Use an explicitly frozen small pilot only for
admission, not as a substitute for the proposed full matched comparison.
Fall 2026 remains outside all selection/tuning. No paid compute or remote study
processing is authorized by this branch.
