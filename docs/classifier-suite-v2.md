# Versioned transcript classifier suite v2

`classifier-suite/2.0` is independent of `scripts/controlled_experiments.py`.
Historical fingerprints and outputs remain unchanged. A classifier includes its
model artifact, engine version, quantization, decoding, dependency environment,
source files, ordered labels, codebook, normalization, context and retrieval
rules, serializer, validation, retries and threshold. Changing any of these
creates a new classifier. Private panel identity additionally freezes the input
bundle, row/label keys, gold labels, ranking lists and design declaration.
Repetition/run/attempt IDs are separate from classifier identity.

## Four stages

1. **Preparation:** validate one frozen input bundle; compile all arms together;
   freeze prompts, source examples and row/label population. No inference occurs.
2. **Inference:** use a fresh repetition namespace or explicitly resume one;
   record every write-ahead attempt, raw response, failure and accepted decision.
   Native execution is limited to 512 jobs per panel in this phase; a scalable
   append-only/SQLite journal is required before admitting larger real studies.
3. **Analysis:** verify preparation, attempts and completion; retain abstentions;
   write detailed measurements only to the private output directory.
4. **Presentation:** emit only a typed public metadata allowlist. Study metrics,
   rare cells, hashes, paths, host details and raw inputs need a separate reviewed
   publication policy; they are not public outputs of this phase.

## Synthetic commands

Run from this checkout after `uv sync --frozen`. These commands use fictional
fixtures and a strict fake backend. They measure runner behavior, not model
accuracy or performance. Output directories must be new descendants of
`artifacts/`; directories/files are created as 0700/0600.

```bash
uv run python scripts/classifier_suite_v2.py list
uv run python scripts/classifier_suite_v2.py validate
uv run python scripts/classifier_suite_v2.py dry-run --synthetic --recipes rag_baseline no_retrieval
uv run python scripts/classifier_suite_v2.py prepare --synthetic --recipes rag_baseline no_retrieval --out artifacts/synthetic-rag
uv run python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-rag --repeat first
uv run python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-rag --repeat first --resume
uv run python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-rag --repeat second
uv run python scripts/classifier_suite_v2.py report --out artifacts/synthetic-rag --repeats first second
uv run python scripts/classifier_suite_v2.py export-public --out artifacts/synthetic-rag
uv run python scripts/classifier_suite_v2.py dry-run --synthetic --recipes context_prev0 context_prev2 context_prev6 context_recency_decay_budget
uv run python scripts/classifier_suite_v2.py dry-run --synthetic --factorial --recipes d1_single_label d1_shared_state
```

## Frozen input interface and local admission

Private inputs are supplied explicitly with `--input` and `--bindings`, replacing
`--synthetic`. The successor never discovers a workbook, private checkout,
credentials or services. The first phase consumes a separately reviewed JSON
bundle rather than silently selecting or regenerating a sample or retrieval
index. `synthetic_bundle()` documents an executable fictional input example.
All keys are required; unknown keys fail validation.

The `frozen-input/2` bundle has `synthetic`, 21 ordered `labels`, a matching ordered
`codebook` (`label`, `definition`, `positive`, `negative`), `rows`, `sources`,
`rankings`, `retrieval_identity`, and `reviewed_blanks: true`. Each row has `id`,
`semester`, `meeting`, integer `position`, `text`, boolean `target`, `gold` and
`coder_codes`. Non-target rows provide context only. Codes are lists of known
labels or the empty string. Empty individual-coder cells mean observed no-code
responses. `null`, missing cells and unknown codes fail; abstention exists only
in model outputs. Source examples have `id`, `semester`, `meeting`, `text`, and
`codes`. Rankings map every target ID to an ordered, duplicate-free source-ID
list. Retrieval identity has `algorithm`, `index_revision`, `embedding_revision`
and `ranking_revision`. Actual bundle contents are also fingerprinted, so
provenance declarations cannot replace a frozen source snapshot.

Only S26/SP26 target/context rows are admitted. Development and FA26 examples
are rejected from retrieval sources; exact normalized overlaps with target text
in retrieval/static examples are rejected. This boundary does not validate a
private workbook extraction or prove absence of semantic paraphrase leakage;
that preparation remains a separately reviewed prerequisite. No FA26 input may
be supplied. Freeze the selection before looking at results; never retune or
replace rows independently for an arm.

Bindings map recipe names to exact runtime objects: `backend`, `model`,
`artifact_digest` (SHA256), `engine_version`, `show_digest` (SHA256 of the explicit
Ollama show-field projection), `quantization`, `token_bound`,
`template_overhead_tokens`, and `hardware_identity` (a private declared runtime
identifier). Model tag alone is insufficient. The local adapter verifies the
exact installed digest, engine version, show identity and quantization before
and after each call, including completion. Hardware identity is an admission
declaration, not hardware attestation from the Ollama API.

`token_bound` must be `utf8_bytes_upper_bound_v1`: admission must establish that
the model tokenizer emits no more than one token per UTF8 input byte and that
the fixed template/system overhead stays within the recorded positive token
allowance. The complete serialized request plus overhead and output reservation
must strictly fit the pinned context capacity. This conservative bound can reject
requests that an exact tokenizer could fit; it never deliberately truncates
requests or removes rows. A runtime prompt-count check remains mandatory.

For a separately admitted native local Ollama endpoint:

```bash
uv run python scripts/classifier_suite_v2.py prepare --input research-inputs/panel.private.json --bindings research-inputs/runtime.private.json --recipes rag_baseline no_retrieval --out artifacts/admitted-local
uv run python scripts/classifier_suite_v2.py run --out artifacts/admitted-local --url http://127.0.0.1:11434 --repeat first
```

No service is started, model is pulled, or URL is guessed. Only numeric HTTP
loopback origins with an explicit port are accepted; redirects/proxy environment
are disabled. These commands require an existing admitted engine and exact
artifact. Synthetic and real adapters cannot share a prepared panel.

## Controlled recipes

The registry is `config/experiments/suite-v2.json`. `rag_baseline` uses frozen
global top-five source rankings, then filters for the current label.
`no_retrieval` empties only the retrieved-example section. Static definitions,
positive/boundary examples, context, gold, source snapshot, runtime and decoding
are identical. This phase does not build embeddings or issue retrieval queries.

Run the context-only recipes as a separate no-retrieval panel. Fixed windows
select the preceding 0, 2 or 6 positions within the same semester/meeting, ordered
by integer position, skipping blank text. Whole-turn byte cost is UTF8 bytes + 1;
fixed-window budget overflow fails preparation. The honestly named
`recency_decay_budget` selects at most 12 preceding positions, newest first,
charging `(UTF8 bytes + 1) * 2**distance` against 1024 weighted bytes and stopping
at the first overflow; output returns to chronological order. Blank positions
still count toward distance. It changes input selection, not transformer
attention or probability calibration. Context never contains the target or a
future turn, and prompts explicitly restrict evidence attribution to the target.
The current `no_retrieval` and `context_prev2` recipes have identical effective
settings apart from their registered names; they are controls in separate
comparisons, not distinct interventions. Real long-turn inputs can exceed the
fixed-window byte budget and fail the entire preparation. Admit feasibility
without silently dropping rows or truncating only one arm.

Combining context and retrieval interventions, or changing models/packing,
requires an explicit `--factorial` declaration during preparation. It is not a
claim that every cell of a complete factorial was provided. Execution supports
one exact runtime per panel; model comparisons use separate admitted panels and
require a separately matched analysis design. The fixed study-wide population
is never silently adjusted to fit an arm.

Granite confirmation pins `ibm-granite/granite-4.2-8b` revision
`f8de16cdcdbc6c779ca517604e050d82cc119e44`. Preparation is available; all remote
inference, including synthetic execution of the remote recipe, is blocked.
Fresh authorization and a reviewed remote adapter are required in a later phase.
No cloud/GPU activity is part of this implementation.

D1 pins `LiquidAI/d1-3B` revision
`051bcc464b01b9f92942b364d9586b0ef5912432`. The boundary serializes `state` and
ordered `questions`, and validates a bridge envelope `{labels, noul}` with native
P(yes), finite numeric values in [0,1], exact label order and vector length.
This envelope is a proposed adapter contract, not a verified live server format.
`d1_single_label` makes one isolated question per job; `d1_shared_state` makes
21 questions per job. They are distinct classifiers. Zero output tokens and
`P(yes) >= 0.5` are fixed initially. Native scores are not calibrated on this
study. Any future threshold tuning uses a development selection panel only.
No text-generated substitute, weight download, automatic `trust_remote_code`, or
unvetted native execution is implemented. Serializer/probability validation and
synthetic execution work; real D1 inference remains admission-gated.

## Replay, analysis and privacy

Fresh `--repeat` requires an unused namespace and new attempts. `--resume`
requires an existing namespace, reuses accepted decisions (including
abstentions), and marks unfinished attempts as interrupted. Transport failures
retry identical payloads up to the fixed per-recovery-session limit; exhausted
transport retries leave an incomplete run that must resume, never a model abstention; invalid content is an abstention
and is not retried. Interrupted/failed attempts remain in the ledger. A crash
after response persistence recovers that decision without pretending it was a
fresh call. Runtime drift quarantines the entire prepared panel and blocks new repeats.
Transport failure fetching identity stops execution without claiming identity drift. Completion must
verify before reports are accepted; every registered repetition must be named
and validated, so failed or unfinished repeats cannot be silently omitted. Seals detect accidental tampering; they are
not signatures against an adversary able to rewrite all local artifacts.

Private `analysis.json` reports confusion counts and coverage per label and
pooled, repeat agreement per label and for complete 21-label vectors, coverage
on both compared repeats, and separate sensitivity comparisons across
classifiers. Equality including abstention is distinct from agreement among
covered decisions. Vector coverage requires every label in both vectors to be
covered. Individual-coder agreement treats blanks as negative. Repeatability,
human agreement and intervention sensitivity are separate measurements; none
implies accuracy without an appropriate reference standard. A resume does not
add a repeatability observation. With a fixed seed and temperature zero these
repeats measure engine nondeterminism, not variation across sampling seeds.

Private prompts, IDs, raw responses, retrieval contents and fingerprints remain
in ignored artifacts with restrictive modes. CLI errors are generic to avoid
leaking private values. `export-public` reconstructs a strict allowlist of schema,
registered recipe names, evidence kind and phase. It exports no private hashes
or rare cells. `export-sources --include <path> --out artifacts/<fresh-name>`
packages only explicitly selected allowlisted committed source bytes unchanged
in the working tree, rejecting
untracked, ignored, symlink and private selections. It never globs a filesystem.
Git ignore is a convenience, not a publication guarantee. Source contents still
require human review before committing or publishing.
