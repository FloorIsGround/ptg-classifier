# Causal context-selection comparison

Branch: `research/context-falloff`. Shared implementation and all admission gaps:
[development experiment queue](research-experiments.md).

## Question and controls

How sensitive is target-turn coding to retained preceding context? Compare
`context_prev0`, `context_prev2`, `context_prev6` and
`context_recency_decay_budget` on identical target rows and all 21 labels.
All initial context arms hold dynamic retrieval absent, and hold the pinned
model/runtime, static examples, decoding and parser fixed. This is a separate
experiment from RAG/no-retrieval; combining interventions requires an explicit
factorial design and does not isolate one variable.

Retain only earlier turns in the same meeting, including non-target rows in the
frozen input. The initial decay rule is a byte-budget retention policy with a
fixed twofold cost multiplier per predecessor rank. It is not an internal
attention-decay control. Blank rows count toward rank; filtering rows changes
that rank, and position gaps are not elapsed time. Preparation must fail rather
than silently truncate or change the target population.

A subsequent tunable decay-rate sweep requires a reviewed recipe/validator
extension, fixed budget, new classifier identities and development-only
selection. Compare meetings separately: a pooled gain can hide opposite effects.
Do not choose per-label mixtures on evaluation rows and call their selection
score independent improvement.

## Executable fictional smoke

```bash
uv sync --frozen
uv run --frozen python scripts/classifier_suite_v2.py prepare --synthetic --recipes context_prev0 context_prev2 context_prev6 context_recency_decay_budget --out artifacts/synthetic-context
uv run --frozen python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-context --repeat first
uv run --frozen python scripts/classifier_suite_v2.py report --out artifacts/synthetic-context --repeats first
```

Use a fresh output directory for another smoke. A synthetic result establishes
runner/context behavior only. The initial policy is implemented; a configurable
rate sweep is queued, not implemented.

## Admission before real inference

Use reviewed context-complete private extraction and a pinned admitted local
engine; add independently verified meeting-specific scientific metrics and a
scalable journal for full-development execution. Preserve the native 512-job
gate. Freeze context rules and every effective classifier component before
inference. Reserve Fall 2026 outside development selection. No GPU service is
stopped, model downloaded or paid compute requested by this protocol.
