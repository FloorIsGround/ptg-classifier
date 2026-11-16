# Quickstart and repository map

## What this program does

PTG reads an Excel sheet of conversation turns. For each nonempty turn, it asks a language model a separate yes/no question for each active codebook label. With the 21-label config, three target turns mean 63 label decisions before retries. Several labels, or no labels, may apply to a turn.

The Python program prepares questions and saves answers. A separately running inference server supplies the model answers. RAG means retrieving examples and adding them to a question; it does not train or change model weights. An embedding model converts text into numeric vectors used to find similar examples.

## First run: entirely fictional, without an inference service

Prerequisites: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). The commands below use the checked-in lockfile; see [uv's sync documentation](https://docs.astral.sh/uv/concepts/projects/sync/). They were exercised on Python 3.12.3, Linux and uv 0.13.4. Other operating systems were not tested for this draft.

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ptg --help
mkdir -p artifacts
uv run --frozen python scripts/maintainer_demo.py > artifacts/maintainer-demo.json
DEMO_DIR="$(uv run --frozen python -c 'import json; print(json.load(open("artifacts/maintainer-demo.json"))["output_directory"])')"
uv run --frozen python -m json.tool "$DEMO_DIR/walkthrough.json"
```

`DEMO_DIR` is a shell variable, used by the commands below in the same terminal. Each demo invocation creates a fresh `artifacts/synthetic-demo-*` directory. The summary is in `artifacts/maintainer-demo.json`; it points to that invocation's files. Do not assume the directory suffix stays the same.

The new [maintainer demo](../../scripts/maintainer_demo.py) wraps the existing [synthetic demo](../../scripts/synthetic_demo.py). It uses actual workbook ingestion, FAISS search, prompts, HTTP response parsing, checkpoints, raw/wide exports and scoring. Hash-based vectors replace model embeddings. An in-memory HTTP mock returns `yes` exactly when a fictional target contains `SYNTHETIC_POSITIVE[the target label]`. Socket connection attempts are refused during the demo. No actual model answers are measured.

The original demo remains available:

```bash
uv run --frozen python scripts/synthetic_demo.py
```

Its perfect oracle score is a plumbing check. It cannot show coding quality, model readiness, latency or a RAG benefit.

## Follow one fictional turn

Open the generated `fictional-workbook.xlsx`, sheet `Team 1 Time 1 (S26)`. Excel row 2 is DataFrame row index 0, with the text:

```text
Fictional target S26: SYNTHETIC_POSITIVE[Initiative]
```

1. [Excel ingestion](../../src/ptg/ingest/excel.py) requires `Speaker` and `Transcript`. [Normalization](../../src/ptg/ingest/normalize.py) fills missing speakers forward, normalizes text and preserves row positions. Empty transcript rows are not classification targets.
2. The demo loads [study-21.yaml](../../config/study-21.yaml) and overrides workbook, output paths, embedding/chat model identifiers and chat URL in memory. The wrapper writes those operational settings to `demo-config.yaml` and checks that loading it reproduces them. Its URL uses an intentionally unusable loopback port; it is not a live server configuration.
3. The demo builds the reference index from eight fictional coded turns in four F25 sheets. Only nonempty turns with retained reconciled codes enter that index. Human codes are metadata, not part of the text being embedded. Target sheets are separate from reference sheets.
4. The target vector retrieves the five nearest reference rows globally. For the `Initiative` question, only neighbors carrying that label become retrieved examples. This is not five neighbors per label. There are no preceding turns for row 0; later targets use up to the previous two physical rows, omitting empty text.
5. [The prompt builder](../../src/ptg/classify/binary_classifier.py) combines the label, category, definition, static examples, retrieved examples, previous text and current text. Inspect `prompt-with-retrieval.txt` and the captured `request-example.json`. The actual request has system/user messages and asks for one word, `yes` or `no`.
6. The mock reads the current target's marker and answers `yes` for Initiative. [The parser](../../src/ptg/classify/enforce.py) accepts yes/no with case/punctuation normalization; other text is invalid. The raw workbook records row index 0, label `Initiative`, raw response `yes`, parsed response `yes`.
7. The wide workbook's `Team 1 Time 1 (S26)` sheet has `yes` in the first target row's `Initiative` column. `walkthrough.json` links the input, prompt and output identities. It also names the raw sheet, `Team 1 Time 1 (S26) Run 0`.

`Reconciled Codes` supplies reference labels and scoring truth. The target's reconciled cell is not inserted into its classification prompt. The ordinary pipeline refuses an indexed target sheet, but does not automatically remove identical text occurring on another sheet, near duplicates or participant overlap. Static codebook examples need a separate overlap review.

## Files to know

| File or directory | Role and actual behavior |
| --- | --- |
| [pyproject.toml](../../pyproject.toml), [uv.lock](../../uv.lock) | Dependencies and `ptg = ptg.cli:main`. CPU FAISS and Excel packages are included; a model server is not. |
| [src/ptg/cli.py](../../src/ptg/cli.py) | Actual commands: `build-index`, `infer`, `export`. Global `--config` precedes the subcommand. No `classify` or `--no-rag` option exists. |
| [src/ptg/config.py](../../src/ptg/config.py) | YAML settings, label validation and whole-value `${ENV_VAR}` substitution. Relative paths resolve from the working directory, not the YAML directory. |
| [config/study-21.yaml](../../config/study-21.yaml) | 21 labels, previous 2 rows, top 5 retrieval, Qwen defaults through environment variables. |
| [config/study-21-granite-4.2.yaml](../../config/study-21-granite-4.2.yaml) | Model-specific Granite configuration, separate prediction namespace, shared reference index. It is not the November comparison configuration. |
| [config/codebooks/f25_codebook_2026_06_14.yaml](../../config/codebooks/f25_codebook_2026_06_14.yaml) | Definitions, categories, aliases, fictional positive/boundary examples. The original Word document is not distributed here. |
| [src/ptg/pipeline/run.py](../../src/ptg/pipeline/run.py) | Index creation/provenance, target exclusion, query cache, job creation, checkpoints, raw and wide export. This is the path called by `ptg`. |
| [src/ptg/index/embed.py](../../src/ptg/index/embed.py) | Ollama embedding client; batches, sanitization and retries. |
| [src/ptg/classify/binary_classifier.py](../../src/ptg/classify/binary_classifier.py) | OpenAI-compatible chat HTTP client, prompt and retry loop. |
| [scripts/evaluate_reconciled.py](../../scripts/evaluate_reconciled.py) | Scoring restricted to the two named development sheets, with provenance/coverage checks. |
| [scripts/controlled_experiments.py](../../scripts/controlled_experiments.py) | Separate local Ollama paired retrieval/context experiment, with prepare/run/report. |
| [scripts/retrieval_coverage_audit.py](../../scripts/retrieval_coverage_audit.py) | Offline availability of examples; it does not measure classifier accuracy. |
| [results/](../../results/) | Saved aggregate development evidence, not the omitted raw predictions or November paired package. |
| [deploy/granite-vllm-start.sh](../../deploy/granite-vllm-start.sh) | Historical model-serving launcher. Read it before any approved deployment; do not execute it for this walkthrough. |
| Rental scripts and [notebooks/colab_runner.ipynb](../../notebooks/colab_runner.ipynb) | Optional separate operational paths; not prerequisites for the demo or permission to rent/upload. |

`config/study-23.yaml` and its extended draft codebook are compatibility fixtures, not an approved evaluated arm. Additional modules such as `dspy_classifier.py` are not the classifier selected by the current `ptg infer` path. Follow imports and calls, not filenames alone.

## Inputs and outputs

The distributed checkout contains no research workbook or retrieval index. The configured study workbook filename is a placeholder for a separately authorized input. In the demo, use only the generated fictional workbook. Sheets follow `Team {team} Time {meet} ({semester})`. Reference building needs `Reconciled Codes`; scoring also needs confirmed annotation semantics. Optional named timestamp columns are used when available; the pipeline otherwise falls back to physical column B, so that fallback may not be a meaningful timestamp in a minimal fixture.

Using the demo's `f25_core_` prefix:

| Generated location under `DEMO_DIR` | Contents |
| --- | --- |
| `fictional-workbook.xlsx` | Eight fictional reference turns, six targets across two sheets. |
| `demo-config.yaml` | Effective synthetic settings, usable for offline export/scoring. Do not run `infer` or `build-index` with its fake model names. |
| `index/f25_core_index.faiss` | Numeric reference vectors. |
| `index/f25_core_metas.parquet` | Reference text, context, codes and source-row metadata. Treat as private with real data. |
| `index/f25_core_index_manifest.json` | Workbook/codebook/model/index provenance. |
| `predictions/f25_core_<sheet>_query_vectors.npz` | Cached target vectors. |
| `predictions/f25_core_<sheet-hash>_checkpoint.<run-number>.done.sqlite3` | Completed row-label ledger; unfinished checkpoints omit the `.done` suffix. |
| `predictions/f25_core_raw_results.xlsx` | Long form: one record per row-label decision; separate `Run N` sheets. |
| `predictions/f25_core_wide_results.xlsx` | One row per target, separate label columns and `Flags`. This is not an inserted `AI` column in the source workbook. |
| `walkthrough.json`, `request-example.json`, `prompt-*.txt` | Synthetic explanatory artifacts added by the wrapper. |

A single run yields yes/no label cells, with blank/flagged abstentions for unresolved answers. Multiple raw runs aggregate into fractions of valid answers that were yes, not calibrated probabilities. A new completed `infer` invocation creates a new run; `export` aggregates all matching runs. Use fresh output namespaces when changing models, endpoints or scientific settings.

Export and scoring can be demonstrated without running inference again:

```bash
uv run --frozen ptg --config "$DEMO_DIR/demo-config.yaml" export --semester S26 --team 1 --meet 1
uv run --frozen python scripts/evaluate_reconciled.py --config "$DEMO_DIR/demo-config.yaml" --sheets S26:1:1,SP26:2:1 --affirm-reviewed-blanks --out "$DEMO_DIR/evaluation.json"
```

That flag attests that blank reconciled cells in this fictional fixture mean no code. It does not establish that every real reviewer column is a complete independent annotation. Some reviewer columns contain sparse additions or objections; do not infer a human-human baseline from their blanks. Older wording in `docs/project-overview.md` and `docs/reproducibility.md` treats individual-coder blanks as dense no-code decisions. Confirm column-specific annotation semantics with the study owner before using those passages for agreement analysis.

## RAG, no retrieval and paired results

The main `infer_sheet` path always loads an index, embeds targets and calls FAISS. There is no first-class no-RAG CLI switch. Do not treat `retrieval.top_k: 0` as a verified service-free path.

The controlled harness has four paired arms: `baseline`, `no_retrieval`, `label_top5`, `context_prev6`. Its `no_retrieval` arm empties **dynamic retrieved examples**; label definitions, static codebook examples and the baseline previous-two-row context remain. Preparation still requires an index and cached query vectors. This differs from removing all example sources.

The demo writes a no-retrieval prompt using the same builder, with only `retrieved_examples=[]`. It does not classify or score that second prompt. Comparing those text files illustrates the mechanism, not performance.

For an existing controlled run, inspect its chosen directory under `artifacts/`: `manifest.json`, `prepared.sqlite3`, `responses.sqlite3`, then `aggregate.json` and `aggregate.md` if reporting completed. `manifest.json` records the arm specs and selected panel; SQLite files contain prompts/decisions. Reports live under `arms` in the JSON, with pooled/per-sheet/per-label results. The reporter refuses incomplete coverage or a missing final completion-validation marker. These files are generated, not shipped. `report --out` also checks source identities and writes reports; it is not a universal viewer for older runs made with other source versions.

The controlled `run` subcommand speaks native local Ollama, not the main client's `/v1/chat/completions` protocol. Its URL must be a bare numeric loopback HTTP origin without `/v1`; its workers are limited to 2 or 4. Verify its interface without dispatching:

```bash
uv run --frozen python scripts/controlled_experiments.py prepare --help
uv run --frozen python scripts/controlled_experiments.py run --help
uv run --frozen python scripts/controlled_experiments.py report --help
```

The later November Granite RAG/no-RAG experiment and copied workbooks discussed in the thread were frozen separately. Their runner, private inputs and results are not in this checkout. The thread records that both arms removed original fixed codebook examples and that the dynamic reference pool was filtered against evaluation text. Those are historical handoff facts, not safeguards implemented by this checkout's ordinary `build-index`. Request the approved manifest/protocol and saved results through the owner; do not reconstruct them from this template config. Small comparisons within the same development meetings do not establish whether RAG improves coding on new meetings.
