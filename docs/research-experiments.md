# Development experiment queue

This queue uses the separately versioned [classifier suite v2](classifier-suite-v2.md).
The September harness and historical artifacts remain unchanged. These are
research protocols and tested synthetic interfaces, not new accuracy results.

## Branches and order

- `research/reproducible-suite-v2`: shared contracts, frozen preparation,
  private attempt/decision artifacts, synthetic tests and guarded source export.
- `research/rag-vs-none`: first matched local comparison of dynamic retrieval
  versus no dynamic retrieval.
- `research/context-falloff`: a separate context-selection comparison with
  retrieval held fixed.
- `research/liquid-d1-native`: native decision probabilities, first with fictional
  inputs; then an admitted small development panel if feasible locally.
- `research/granite42-confirmation`: later matched confirmation with Granite 4.2,
  using separately authorized remote compute if local admission is impractical.

Each experiment branch starts from the same shared implementation. Its
`docs/experiment-protocol.md` states the branch-specific comparison, executable
synthetic/preparation commands and admission gaps. Branch existence is not proof
of live-model support or a completed study.

## Matched retrieval experiment

Use `rag_baseline` and `no_retrieval` on one frozen development population with
all 21 labels. The no-retrieval arm removes only dynamic retrieved examples.
Static definitions and positive/boundary examples remain identical, as do
preceding context, model artifact, runtime, decoding, parser and scoring.
Definitions-only prompts are a separate intervention, not the control arm.

Prepare all arms together, interleave requests deterministically, and record
fresh repetitions separately from resumption. A resumed or cached result is
not another inference repetition. Preserve failed and incomplete repetitions;
do not report only favorable completed ones.

The input interface currently rejects exact normalized target overlap with
static or retrieved examples. Do not silently remove overlap rows independently
by arm, or change static examples and claim to vary retrieval alone. Before a
real study, specify whether to use an overlap-free primary development population
or extend admission for an explicitly declared contaminated primary analysis
plus a matched overlap-exclusion sensitivity. Either choice changes the study
protocol and must be frozen before looking at predictions. Exact matching does
not prove participant, recording or semantic independence.

## Context selection, not attention decay

Compare the existing 0-, 2- and 6-predecessor windows with the current
`recency_decay_budget` policy on the same target rows and labels. All existing
context-only recipes use no dynamic retrieval. The decay recipe charges whole
turns increasingly with predecessor rank against a fixed byte budget; it does
not change the model's internal attention weights.

The initial decay multiplier is fixed at two per predecessor rank. A tunable
rate sweep needs a reviewed validator/recipe extension and new classifier
identities. Keep the budget and retrieval policy fixed while varying a rate.
Preserve non-target context rows, including blank rows: filtering them before
preparation changes predecessor rank. Integer position gaps are not elapsed-time
decay. Every context turn must precede the target in the same meeting.

## D1 native admission

Use the official pinned `LiquidAI/d1-3B` revision in the registry. Native
probabilities are not equivalent to generated yes/no strings. Single-label and
shared-state question packing are distinct classifier configurations. The
initial threshold is fixed at 0.5; scores are not presumed calibrated for this
study. Any threshold selection must use development data, not the final holdout.

The current implementation tests serialization, probability-vector validation
and synthetic execution. It does not implement a live D1 transport. Review the
custom model code and dependency versions before executing remote code. Verify
actual runtime memory and label order on fictional inputs without interrupting
existing GPU services. Weight size alone is not a memory-fit measurement.

## Before real study inference

1. Review and version the workbook/index-to-frozen-JSON converter. Preserve
   source bytes, selection rules, context rows and hashes privately. Blank
   individual-coder cells mean no category code, negative for every label;
   annotation independence is a separate provenance question.
2. Admit a pinned local engine, model artifact, tokenizer bound and template
   overhead. Probe actual request/response behavior on fictional inputs first.
3. Replace the whole-JSON ledger with a scalable append-only or transactional
   journal before full-development execution. Non-synthetic execution (currently
   local Ollama) is capped at 512 jobs per panel. Keep a study-level register of
   all preparations and panels, not just repetitions inside one directory. Do
   not bypass the gate or split a study into convenient panels without an
   explicit matched design and aggregate analysis. Before real inference,
   harden response preservation when post-call identity checks fail and the
   registry/first-ledger crash window; both remain audit/recovery limitations.
4. Extend and independently verify scientific reporting. Current private
   analysis supplies confusion counts, coverage and agreements. Scientific
   comparison still needs precision/recall/F1, Cohen's kappa, meeting-specific
   and pooled descriptive summaries, paired deltas, and declared treatment of
   positive abstentions. Human-human and LLM-human comparisons use identical
   eligible rows/labels; independent coder provenance is reported separately.
5. Freeze prompts, static examples, retrieval snapshots, context rules, parser,
   threshold, source/dependency versions, runtime identity and scoring before
   inference. Keep repeatability, human agreement and intervention sensitivity
   distinct. Reserve Fall 2026 outside all development selection and tuning.

## Public/private boundary

Publish source, synthetic fixtures, validated recipe definitions and methods.
Keep transcripts, original codebooks, private rankings/embeddings, prompts,
responses, row IDs, attempt ledgers, study fingerprints and detailed analysis
outside Git in restricted ignored storage. No original workbook or thesis text
belongs in an agent review or public branch.

The new source exporter accepts only explicitly selected reviewed committed
source bytes; it does not repair or approve the older filesystem-glob bundler.
Publication checks must inspect reachable blob contents as well as current
paths. Git ignore and successful tests alone do not establish privacy.

Vast.ai or Colab execution requires a later decision on privacy authorization,
GPU/backend compatibility, a finite budget and cleanup. No cloud rental,
remote transcript processing, model download or real study inference is
performed by this queue.
