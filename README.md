# Meeting transcript coding pipeline

This project aims to make a practical transcript-coding assistant that mimics recorded human coding of emotional intelligence and team interaction. The current research pipeline makes a binary decision for each of 21 labels on every nonempty turn, using codebook definitions, examples, preceding turns and optional retrieved examples.

**This is a sanitized collaborator repository, not the original research archive.** It has fresh Git history. It includes implementation, synthetic tests, aggregate development results and documentation. No participant transcript workbook, retrieval index, prompt ledger or prediction workbook is distributed.

**The bundled codebook examples are fictional replacements.** They do not reproduce the reported study results. Authorized study reproduction requires the frozen private workbook and codebook, plus the recorded model/runtime and original source version. See [reproduction](docs/reproducibility.md).

## Start here

- [Export saved results into a copied workbook with an AI column](docs/ai-workbook-export.md)
- [Current development results and their limits](docs/results.md)
- [Data split and evaluation methods](docs/methods.md)
- [Project overview and implementation](docs/project-overview.md)
- [Installation and reproduction](docs/reproducibility.md)
- [Data privacy and sharing boundaries](docs/data-governance.md)

## Quick start: no private data, no GPU, no network inference

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen python scripts/synthetic_demo.py
uv run --frozen ptg --help
uv run --frozen python scripts/check_repository.py
```

The synthetic demo exercises the real workbook ingestion, FAISS retrieval, binary prompting, checkpoints, export and evaluation. It replaces only embedding/model network calls with deterministic test doubles. Its scores test plumbing, **not LLM quality**. Generated files remain in ignored `artifacts/`.

## Real inference

The code needs an Ollama embedding service and an OpenAI-compatible chat service, supplied separately. `uv sync` does not install models or start a model server. The old variable name `LM_STUDIOS_HOST` is retained for compatibility; LM Studio is not mandatory.

```bash
cp .env.example .env
# Set approved local endpoints and exact model identifiers.
# Supply an authorized workbook privately, never commit it.
uv run --frozen ptg --config config/study-21.yaml build-index --semester F25 --teams 1,2 --meets 1,2
uv run --frozen ptg --config config/study-21.yaml infer --semester S26 --team 1 --meet 1 --pilot-row-limit 10 --lm-workers 2
uv run --frozen ptg --config config/study-21.yaml export --semester S26 --team 1 --meet 1 --pilot 10
```

These commands with the **synthetic template codebook** create a new exploratory run, not a historical reproduction. Follow the private-config instructions in the reproduction guide for the research codebook. Index and target sheets must differ.

## Current evidence

- September 24, 2026: Qwen and Granite were scored against reconciled human codes on 1,256 development turns and all 21 labels. Micro-F1 was **0.2398** for Qwen and **0.3628** for Granite. Their mean sheet-label Cohen's kappas were **0.1664** and **0.2372**, respectively; Granite had one invalid negative decision, excluded from its kappa summary.
- An offline replay of September Granite retrieval found normalized exact-text matches among the top five for **15 of 1,256** target turns. All were short; the overlap does not establish a cause for score differences.
- September 29: a 128-turn Qwen screen improved micro-F1 from **0.2011** to **0.2615** when dynamic retrieval was removed. Static codebook examples remained. This is a small development finding, not Granite evidence.
- October 14: a D1 packing and compute-path comparison produced micro-F1 **0.0821** for isolated single-label requests and **0.0848** for shared-state all-label packing. Three same-seed repeats had identical predictions within each arm.
- October 14: one complete Granite development repeat yielded micro-F1 **0.1650** with dynamic retrieval and **0.1613** without it. The meeting results split and macro-F1 favored no retrieval, so the earlier Qwen result does not transfer automatically.

These are development findings, not independent final-test performance or evidence of human-level coding. The reported model scores compare predictions with reconciled human codes. A human-human baseline requires original individual decisions or an explicitly documented reconstruction of accepted labels and objections. FA26 remains reserved from new inference; no independent final evaluation has been performed. See [results](docs/results.md) for experiment scope and limitations.

## Layout

```text
src/ptg/        Ingest, index, retrieve, classify, checkpoint, export
scripts/        Audits, scoring, controlled experiments, synthetic demo
config/         Model settings and clearly marked synthetic codebooks
notebooks/      Optional opt-in Colab staging, not required for local use
tests/          Synthetic regression tests, no participant fixtures
docs/           Collaborator-facing project and evaluation documentation
results/        Aggregate-only metrics and per-label confusion counts
```

`config/study-23.yaml` and its extended codebook are archived compatibility fixtures, not an approved evaluated arm. Optional rental/watchdog scripts are retained for code review but require separately configured approval, credentials and safety timers. No cloud processing is authorized by this repository.

## Repository access

[FloorIsGround/ptg-classifier](https://github.com/FloorIsGround/ptg-classifier) was confirmed **public** on October 29, 2026. Participant transcripts, private reproduction inputs and original research history are not included.

Public access does not establish which redistribution or new transcript processing is authorized. Review permissions and aggregate disclosure before further sharing. No redistribution license is asserted until the project owner or institution confirms it. See [data privacy and sharing boundaries](docs/data-governance.md).
