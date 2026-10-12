# Meeting transcript coding pipeline

A research pipeline for assigning emotional-intelligence and team-interaction codes to meeting transcript turns. Each nonempty turn is evaluated independently for each of 21 labels, using codebook definitions, examples, preceding turns and optional retrieved examples.

**This is a sanitized collaborator repository, not the original research archive.** It has fresh Git history. It includes implementation, synthetic tests, aggregate development results and documentation. No participant transcript workbook, retrieval index, prompt ledger or prediction workbook is distributed.

**The bundled codebook examples are fictional replacements.** They do not reproduce the reported study results. Authorized study reproduction requires the frozen private workbook and codebook, plus the recorded model/runtime and original source version. See [reproduction](docs/reproducibility.md).

## Start here

- [Project overview and implementation](docs/project-overview.md)
- [Data split and evaluation methods](docs/methods.md)
- [Tests run and actual results](docs/results.md)
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

- September 24, 2026: Qwen and Granite runs on 1,256 reconciled development turns, all 21 labels. Micro-F1: Qwen **0.2398**, Granite **0.3628**.
- Granite's descriptive mean of 42 sheet-label Cohen's kappas: **0.2372**, against reconciled codes, excluding its one invalid decision for kappa only.
- September 29: controlled local Qwen screen on 128 development turns. Removing dynamic retrieval improved micro-F1 from **0.2011** to **0.2615**.
- These are development findings. They do not demonstrate human-level coding or independent final-test performance. Human-human agreement on confirmed independently coded rows remains to be established.

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

[FloorIsGround/ptg-classifier](https://github.com/FloorIsGround/ptg-classifier) is a **private** repository. Participant transcripts, private reproduction inputs and original research history are not included.

Before wider sharing or public release, review construct-definition redistribution permissions, aggregate disclosure and collaborator access. No redistribution license is asserted until the project owner/institution confirms it. See [data privacy and sharing boundaries](docs/data-governance.md).
