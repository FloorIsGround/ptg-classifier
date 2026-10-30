# Installation and reproduction

## Reading order

1. Read the [development results](results.md) and [methods](methods.md) to understand which findings are measured and what remains unvalidated.
2. Use the installation and synthetic demonstration below to exercise the shareable pipeline without participant data or network inference.
3. For authorized research reproduction, review the private-input and model-boundary sections before choosing the original source and runtime. FA26 remains reserved from new inference.

## Three distinct kinds of reproducibility

1. **Software demonstration:** a new clone runs tests and a synthetic integration demo without participant data or model downloads.
2. **Rescoring recorded study predictions:** authorized users obtain immutable private artifacts and use the original source/runtime contract to recompute aggregate results.
3. **Fresh inference replication:** rebuild retrieval and rerun an explicitly frozen workflow on the same private inputs and actual model runtime. GPU/server nondeterminism can still change predictions despite a seed.

A synthetic-template run is not any of the reported historical research runs. This repository cannot reproduce their numerical scores using only the distributed files.

The main snapshot preceding these documentation updates was last committed October 12, 2026. Five research branches published October 13 contain earlier protocol and synthetic-readiness descriptions; the October 14 experiments were completed later in a private execution environment. The commands below describe the shareable main snapshot and are not a claimed recipe for those later runs. Reproducing them requires the actual execution source and runtime as well as the authorized private artifacts.

## Install and verify the shareable source

Python 3.12+ and `uv` are required. From the repository root:

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen python scripts/synthetic_demo.py
uv run --frozen ptg --help
uv run --frozen python scripts/check_repository.py
```

No GPU, transcript workbook or inference endpoint is required for these commands. The demo generates fictional workbook rows and deterministic embedding/chat responses locally; it tests the actual pipeline with explicit test doubles. Generated workbooks, SQLite and indexes are ignored.

## Authorized private inputs

Obtain these through the research team's approved private channel, never GitHub:

- Original workbook with sheet names, reviewed-negative policy and row inclusion rules preserved.
- Frozen research codebook with the original examples, not the fictional distributed template.
- Immutable prediction workbooks, completed SQLite checkpoints and saved aggregate reports for rescoring.
- Original F25 FAISS index, metadata, manifest and query caches, or permission to rebuild them with the matching embedding model.
- Model/runtime details, original code version and any external-processing approval.

The September 24 input hashes are in `results/development-summary.json`. The original source commits are recorded separately from this new repository history. Source-file hashes in `PORTING_MANIFEST.json` do not embed the omitted original Git objects.

The September Granite overlap check replayed saved query vectors offline, without new inference. Reproducing that check requires the historical workbook, index and metadata, manifest, and saved query-vector caches tied to the recorded embedding model. The April/June F25-target index/target relationships remain unreconstructed.

## Configure a new approved study run

```bash
cp .env.example .env
cp config/study-21.yaml config/research.private.yaml
```

In the ignored `config/research.private.yaml`, change `codebook.path` to your privately supplied frozen codebook, for example `./data/research-codebook.yaml`. Supply the workbook under ignored `data/`. Set a **fresh** `io.artifacts_dir` and `io.index_artifacts_dir` for a new run; do not append to historical outputs or mix template/private-codebook hashes.

Configure `.env` with approved endpoints, exact embedding and chat model names. The main `ptg` CLI loads `.env`; standalone Python scripts do not automatically do so. Export the same four environment variables into your shell before invoking scoring/audit scripts:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
export EMBED_MODEL=qwen3-embedding:4b
export LM_STUDIOS_HOST=http://127.0.0.1:11434/v1
export LM_MODEL=qwen3.5:4b
```

Ollama and a compatible chat server are external dependencies. Downloading a model is a separate action. A code install does not prove the endpoint is serving the intended model, quantization or thinking mode.

```bash
uv run --frozen ptg --config config/research.private.yaml build-index --semester F25 --teams 1,2 --meets 1,2
uv run --frozen ptg --config config/research.private.yaml infer --semester S26 --team 1 --meet 1 --pilot-row-limit 10 --lm-workers 2
uv run --frozen ptg --config config/research.private.yaml export --semester S26 --team 1 --meet 1 --pilot 10
# After the pilot passes, run both complete development sheets:
uv run --frozen ptg --config config/research.private.yaml infer --semester S26 --team 1 --meet 1 --lm-workers 2
uv run --frozen ptg --config config/research.private.yaml export --semester S26 --team 1 --meet 1
uv run --frozen ptg --config config/research.private.yaml infer --semester SP26 --team 2 --meet 1 --lm-workers 2
uv run --frozen ptg --config config/research.private.yaml export --semester SP26 --team 2 --meet 1
uv run --frozen python scripts/evaluate_reconciled.py --config config/research.private.yaml --sheets S26:1:1,SP26:2:1 --affirm-reviewed-blanks --out artifacts/new-development/evaluation.json
```

`--affirm-reviewed-blanks` acknowledges the established no-code policy for blank reconciled cells on the two approved development sheets. It does not define blank coder/reviewer entries, which may represent no objection or implicit agreement during reconciliation. The evaluator intentionally refuses FA26 and other unapproved sheets, whose final-evaluation annotations remain a separate dataset-level issue.

For Granite use a private copy of `config/study-21-granite-4.2.yaml`, a separate output namespace and the recorded model-specific decoding. Do not merely switch a model name inside the Qwen configuration. The original BF16 vLLM run is not reproduced by substituting an unrelated quantized model.

The original model-only launch script is retained at `deploy/granite-vllm-start.sh`; it pins the model revision, BF16, 4,096-token server capacity and loopback binding. On an explicitly approved GPU machine with vLLM 0.29.0 (the original image was `vllm/vllm-openai:v0.29.0`):

```bash
# On the approved GPU host, not automatically on the collaborator's laptop:
bash deploy/granite-vllm-start.sh
# In a separate local terminal, if remote processing is approved:
ssh -N -L 18080:127.0.0.1:8000 APPROVED_GPU_HOST
# Set the private Granite configuration's chat endpoint through this variable:
export LM_STUDIOS_HOST=http://127.0.0.1:18080/v1
```

Replace `APPROVED_GPU_HOST` with the authorized SSH alias. This setup requires suitable GPU memory and driver/runtime support; no model server or rental was launched during shareable-repository preparation. The image tag was recorded historically, but an image digest was not captured, so container bit-identity is not established.

## Controlled local experiments

Use a private config with matching workbook, codebook, F25 index and cached query vectors. The harness does not produce embeddings during preparation.

```bash
export PTG_STUDY_ROOT=/absolute/path/to/approved-private-study
export PTG_STUDY_CONFIG=/absolute/path/to/config/research.private.yaml
uv run --frozen python scripts/controlled_experiments.py prepare --out artifacts/controlled-new --model qwen3.5:4b --affirm-reviewed-blanks
uv run --frozen python scripts/controlled_experiments.py run --out artifacts/controlled-new --model qwen3.5:4b --url http://127.0.0.1:11434 --workers 2
uv run --frozen python scripts/controlled_experiments.py report --out artifacts/controlled-new
```

These commands create a **new** frozen experiment. The port has different harness/config hashes and intentionally cannot silently resume the September 29 manifest. Re-render that historical screen with its original `6411fb4` source and private artifacts, or read the verified saved aggregate. Do not edit manifests to bypass hash or completion gates.

## Model and compute boundaries

- Qwen baseline and controlled screen: local workstation.
- September 24 Granite: approved remote vLLM GPU; prompt contents were processed remotely despite local artifact storage.
- Optional Colab notebook: opt-in staging only, not the recommended reproduction path. Drive authorization/storage and external inference require separate permission.
- Optional Vast wrapper: requires configured CLI/host, credentials, registered lease, active safety timers, budget and explicit transcript-processing approval. Host-specific service units are not bundled. It is not a turnkey rental command for collaborators.

## Exact saved-score reproduction limitations

Without the private input bundle and original source versions, collaborators can inspect implementation and aggregates and run synthetic verification, but cannot independently validate the reported scores. Arrange controlled access to those artifacts if an authorized reproduction is needed. Neither seeds nor a source-only repository alone guarantee identical LLM outputs.
