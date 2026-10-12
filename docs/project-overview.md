# Project overview

## Purpose and status

This project studies whether an LLM can assist qualitative coding of team meeting transcripts under a researcher-supplied codebook. It currently uses 21 emotional-intelligence and relational-management labels. A turn can receive multiple labels or none.

The implementation runs end to end and has completed development experiments. Scientific validation remains incomplete: current performance is low, the human-human baseline has not been verified on matched independently coded rows, and the proposed final evaluation data have no reconciled labels in the saved workbook.

## Classification workflow

```text
Privately supplied workbook + approved codebook
              |
              +-- Selected coded F25 turns --> embeddings --> FAISS index
              |                                                  |
              +-- Development target turn --> query embedding ---+
                                                              |
                                      top 5 neighbors, label filter
                                                              |
                            21 independent binary classification jobs
                            - target-label definition
                            - static positive/boundary examples
                            - matching retrieved examples
                            - preceding 2 turns
                            - current target turn
                                                              |
                                  chat server --> parse yes/no --> retry
                                                              |
                                  SQLite checkpoint --> raw Excel results
                                                              |
                                wide export + aggregate evaluator
```

The model is asked to code only the current turn. The default prompt uses definitions, positive and negative boundary examples, two preceding turns and dynamically retrieved examples. The baseline retrieves five neighbors globally, then filters those neighbors to examples bearing the target label. This can leave some decisions without a retrieved example even when the index contains that label.

Inference is separate from export, so output formatting does not require another model run. Each row-label decision is checkpointed to SQLite; completed outputs are checked for coverage and consistency before scoring. Invalid responses after retries become abstentions rather than silent negative predictions.

## Main entry points

- `src/ptg/cli.py`: `build-index`, `infer`, `export`. There is no current `classify` subcommand.
- `src/ptg/ingest/`: Excel sheet loading, sparse speaker forward-fill and transcript normalization.
- `src/ptg/classify/codebook.py`: versioned YAML definitions, aliases and examples.
- `src/ptg/pipeline/run.py`: source-index manifests, retrieval, inference, caching, checkpoints and exports.
- `src/ptg/classify/binary_classifier.py`: prompt builder, concurrent HTTP requests, parsing and retry handling.
- `scripts/evaluate_reconciled.py`: coverage/provenance-gated scoring on the two explicitly approved development sheets.
- `scripts/controlled_experiments.py`: deterministic paired panel, retrieval/context ablations, runtime-identity checks and aggregate reporting.
- `scripts/retrieval_coverage_audit.py`: offline example availability, not classification quality.
- `scripts/synthetic_demo.py`: network-free integration demonstration.

## Data interface

The required columns are `Speaker` and `Transcript`. Index construction and human-reference scoring also use `Reconciled Codes`. Timestamp columns are retained for export when present. In individual coder columns, a blank cell means no category code and is a negative decision for each category in matched agreement analysis.

The study sheet naming pattern is `Team {team} Time {meet} ({semester})`, with semester values such as `F25`, `S26`, `SP26` and `FA26`. Teams with the same numeric identifier in different sheets are not necessarily the same participant cohort.

## Shareable snapshot versus the original archive

The snapshot derives from source commit `4157dc63e6986a9f2f694adb460eb9c1a92ccd4c`. `PORTING_MANIFEST.json` records the original selected-file hashes. The release replaces all codebook examples, sanitizes related test fixtures and removes machine-specific routing defaults. Original PDFs, ad-hoc notebooks, old Git objects, private datasets and run ledgers are excluded.

These changes are deliberate. A ported synthetic-codebook run must not be mislabeled as a reproduction of a frozen research run. The original research checkout and artifacts are preserved separately.

## Known limits

- The default loader expects the source-tree `config/codebooks/` directory. Run from a repository checkout; a standalone wheel without those configuration assets is not a verified distribution method.
- The production pipeline checks source/index hashes and rejects directly indexed target sheets. That does not prove participant-level independence or eliminate semantic overlap.
- The controlled harness has stronger runtime-identity/completion gates than the historical production path. Do not generalize its provenance guarantees to every old run.
- Some failure/debug logs can include transcript fragments or raw responses. Treat all runtime logs as private, even though they are ignored by Git.
- The current evaluator is deliberately restricted to two reconciled development sheets. A prospective final-test evaluator and human-human agreement workflow still need explicit protocol approval.
