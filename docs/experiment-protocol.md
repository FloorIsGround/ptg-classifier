# Granite 4.2 matched confirmation protocol

Branch: `research/granite42-confirmation`. Shared implementation and all admission
gaps: [development experiment queue](research-experiments.md).

## Question and controls

Does a local Qwen retrieval finding persist with pinned Granite 4.2 8B? This
requires a new within-Granite matched retrieval/no-retrieval comparison, not
comparing a new no-retrieval arm against a historical Granite aggregate.
Use identical development rows/all 21 labels, static examples, context,
Granite artifact, quantization, backend, decoding, parser and scoring across
both new arms. Model/backend changes remain different classifier identities.

The registry currently contains only `granite42_confirmation`, a pinned
preparation recipe. The matching Granite no-retrieval recipe and reviewed
remote adapter remain to be implemented. Do not claim the two-arm Granite
experiment is already runnable. A later context confirmation is a separate
comparison with retrieval held fixed.

## Executable preparation check

```bash
uv sync --frozen
uv run --frozen python scripts/classifier_suite_v2.py dry-run --synthetic --recipes granite42_confirmation
```

This compiles fictional requests without inference. Even synthetic `run` of
the remote recipe is blocked by admission. No model download or cloud rental
occurs.

## Later compute admission

Prefer local development and fictional transport probes first. If remote GPU
compute is justified, obtain fresh privacy authorization and a finite rental
budget. Choose Vast.ai or Colab only after reviewing native backend/model
compatibility, private file transfer, credentials, networking, artifact return
and cleanup. Do not publish private prompts, results, hashes or rare cells.

Freeze both Granite arms, versioned private input conversion, tokenizer/runtime
admission and independently verified meeting-specific scientific analysis.
Replace the whole-JSON journal before full-development execution; the existing
512-job gate remains binding. Preserve all repeats and failures, score positive
abstentions explicitly and distinguish engine repeatability from human
agreement. Keep Fall 2026 outside all development/model selection and tuning.

Remote inference is currently hard-blocked. This branch is a queued protocol
and preparation recipe, not an authorization to rent hardware or transfer
transcripts.
