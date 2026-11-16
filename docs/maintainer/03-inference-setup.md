# Inference setup for a Python maintainer

This guide separates the interfaces supported by the checked-out code from infrastructure that would have to be supplied. No GPU was rented, no model server was launched and no live model requests were made for this draft. The offline quickstart is in [the repository map](01-quickstart-and-map.md).

## Client, server, model and host

The **client** is the Python process running `ptg`. It loads Excel, builds prompts and sends HTTP requests. A **server** is a separate, long-running process listening on a port. It loads a model's weights, runs that model for incoming requests and returns responses. **Inference** means executing the trained model; this repository does not train one during classification.

The **host** is the computer running the server. It may be the same workstation as the client, another approved machine or a rented GPU machine. A GPU accelerates model execution; renting one does not by itself make a compatible endpoint available. vLLM and Ollama are serving software, not model names. Granite and Qwen are model families. The exact served model identifier is a string agreed between client and server.

The ordinary pipeline has two independent service interfaces:

```text
Workbook -> Python embedding client -> Ollama embedding server -> vectors
                                                     |
                                 local FAISS index <-+
Target -> query vector -> local retrieval -> prompt -> chat server -> yes/no
                                                        |
                                    Python checkpoint/export <-+
```

FAISS runs inside the Python client; it is not a third HTTP service. With local embedding and remote chat, target text goes to the local embedding service, then the target, preceding context, static examples and selected reference text travel in the remote chat prompt. A tunnel protects the connection; it does not change where the model processes the text.

## What has to be ready before `infer`

1. Python dependencies are installed (`uv sync --frozen`). Run from the checkout root so relative config paths resolve correctly.
2. The embedding service is running and has the configured embedding model available. The model must match the reference index and target-vector cache. Changing it requires a fresh matching index/cache.
3. The chat service has loaded the intended model, its tokenizer/chat template and a compatible runtime. It accepts the fields described below and has enough memory/context capacity. A listening port alone does not establish readiness.
4. The configured URL is reachable from the client. For a remote loopback-bound server, an approved SSH tunnel must already be running.
5. A synthetic workbook, the intended codebook and separate output directories are selected. An index has been built from separate reference sheets; `infer` does not build it for you.

`uv sync` installs the client dependencies, not Ollama, vLLM, model weights, CUDA drivers or rented infrastructure. The CPU synthetic demonstration skips service/model readiness through explicit test doubles; it does not verify those prerequisites.

## The configuration this checkout actually reads

See [config.py](../../src/ptg/config.py), [the CLI](../../src/ptg/cli.py), [.env.example](../../.env.example) and the two study YAML files.

| Setting | Consumer and meaning |
| --- | --- |
| `OLLAMA_HOST` | Environment-only host for `ollama.Client` in `index/embed.py`; normally `http://127.0.0.1:11434`, without `/v1`. |
| `embedding.model` / `${EMBED_MODEL}` | Model identifier passed to Ollama, e.g. the template's `qwen3-embedding:4b`. It is not the chat model. |
| `llm.api_base` / `${LM_STUDIOS_HOST}` | Main chat base, e.g. `http://127.0.0.1:18080/v1`. The client appends `/chat/completions`. The historical environment name does not require LM Studio. |
| `llm.model` / `${LM_MODEL}` | Served chat model identifier. `study-21.yaml` uses the environment; the Granite config hardcodes `granite-4.2-8b`, so changing `LM_MODEL` alone does not change that file's model. |
| `codebook.path`, `codebook.active_labels` | Definitions, categories, static examples, aliases and label order. |
| `excel.workbook`, `excel.sheet_template` | Input file and sheet selection pattern. |
| `io.index_artifacts_dir` | Reference index, metadata and provenance manifest. |
| `io.artifacts_dir` | Per-run query caches, checkpoints and predictions. Use a fresh directory for a changed endpoint/model/arm. |
| `retrieval.top_k`, `context.prev_k` | Global neighbor count and preceding physical-row window. |

The CLI calls `load_dotenv()` before loading settings. Standalone scripts do not automatically load `.env`. For those scripts, supply required environment variables explicitly or use literal values in a synthetic YAML. YAML substitution only recognizes an entire string of the form `${NAME}`, not embedded substitutions such as `${HOST}/v1`. Missing variables are not comprehensively validated at config-load time; check required fields before running.

### Models and decoding are configuration-specific

| Path | Defaults in this checkout |
| --- | --- |
| `study-21.yaml` main client | Environment chat model; `.env.example` suggests `qwen3.5:4b`. Temperature 0, max tokens 10, `reasoning_effort: none`; top-p and seed omitted. |
| `study-21-granite-4.2.yaml` main client | `granite-4.2-8b`, temperature 1.0, top-p 0.95, max tokens 64, seed 42, `chat_template_kwargs.enable_thinking: false`. |
| `controlled_experiments.py` | Separate native Ollama requests: temperature 0, seed 42, `num_ctx` 8192 by default, `num_predict` 16, `think: false`, `stream: false`. |

These are not interchangeable decoding recipes. The November matched comparison in the thread used a different frozen configuration, including temperature 0 and removal of fixed examples. This checkout's Granite YAML does not reproduce it. Changing runtime, model revision, quantization, prompt or decoding creates a different experimental setup even if the served name is unchanged.

## Main chat HTTP contract

[classify_binary_jobs](../../src/ptg/classify/binary_classifier.py) uses `httpx.AsyncClient`, not an OpenAI SDK. It posts one request per row-label question to:

```text
llm.api_base.rstrip('/') + '/chat/completions'
```

The synthetic demo captures an actual request in `request-example.json`. The shape is:

```json
{
  "model": "APPROVED_SERVED_MODEL_ID",
  "messages": [
    {"role": "system", "content": "A coding instruction"},
    {"role": "user", "content": "A synthetic per-label question"}
  ],
  "temperature": 0.0,
  "max_tokens": 10
}
```

Optional configured fields are `reasoning_effort`, `top_p`, `chat_template_kwargs` and `seed`. Requests are non-streaming: the client expects a complete JSON response, and does not implement SSE/chunk parsing. It reads:

```json
{"choices": [{"message": {"content": "yes"}}]}
```

`content` must be a string resolving to `yes` or `no`. Case and punctuation are tolerated; an explanation, tool call, reasoning-only answer or `maybe` is invalid. HTTP errors and invalid responses are retried up to `binary_max_retries + 1` total attempts, default 3. Exhaustion records an abstention, not an ordinary no. A command can finish with unresolved decisions, so inspect coverage and `Flags` before interpreting output.

The client does not validate returned model identity, finish reason or token usage. It assumes the base URL serves the intended model and that the server respects the configured controls. The separate controlled harness has stronger runtime/completion checks; do not attribute those checks to the main client.

vLLM documents an [OpenAI-compatible chat server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/). Compatibility must still be checked for the selected version/model and optional fields; supporting the URL alone is insufficient.

### Authentication and connection limits

The main chat client has **no API-key config field and sends no Authorization header**. Setting `OPENAI_API_KEY` will not add one. An endpoint requiring bearer auth needs a deliberately implemented client change or an approved local proxy that injects credentials. Neither is supplied here. Do not put credentials in URLs, YAML examples or commands shown during screen sharing.

The embedding client is constructed with its host, with no repository-configured auth headers. Prefer approved loopback/tunneled service access. Review proxy environment settings locally without dumping them into logs; the main HTTP client uses httpx's default environment handling. The controlled harness explicitly disables environment-proxy use and restricts its run URL to loopback.

`--lm-workers` controls concurrency, not the number of models or GPUs. Start small for a future synthetic live check; reduce it if the server runs out of memory or times out. The main `--lm-timeout` default is 120 seconds. Longer timeouts do not fix an incorrect route, missing model or unsupported request field.

## Embedding HTTP contract

The repository uses Ollama's Python client, calling `client.embed(model=..., input=...)` with one string or a batch of strings. It consumes numeric `embeddings` arrays and has a legacy singular `embedding` fallback. Vectors must be finite and align one-to-one with input positions; FAISS reference/query dimensions must agree. The service interface is [Ollama's `/api/embed`](https://docs.ollama.com/api/embed), not an OpenAI `/v1/embeddings` client path. Replacing it with an OpenAI embedding endpoint requires an adapter or a code change.

Failed batch embedding falls back to individual retries. Index construction refuses incomplete embedding coverage. Target embedding failures can become abstentions for all labels on affected rows. The default service may have its own input truncation behavior; this client does not request a tokenizer-based no-truncation guarantee.

## Local GPU versus Vast.ai

Local serving means installing a suitable runtime and loading weights on an approved local machine. Renting compute means doing those server-side tasks on a rented machine instead. The Python client, workbook and FAISS index can remain on the workstation in either case.

Vast.ai provides GPU-hosted compute; its [Docker instance documentation](https://docs.vast.ai/guides/instances/docker-environment) explains the container environment, resource allocation and networking. The host/container still needs a suitable image, model weights and a running model server. A Vast account/API key controls rentals; it is distinct from SSH access and any chat API authentication. No Vast CLI is needed to run PTG once an endpoint is supplied.

This checkout retains optional rental/lease scripts, but the walkthrough does not invoke them. A future rental requires a separate decision about budget, data processing, model/runtime, duration, automatic teardown and retained results. An old experiment's spending cap is not a standing authorization for a new rental. No new price or GPU-fit estimate was measured here.

The existing [Granite launch script](../../deploy/granite-vllm-start.sh) specifies:

- Model `ibm-granite/granite-4.2-8b`, revision `f8de16cdcdbc6c779ca517604e050d82cc119e44`, served as `granite-4.2-8b`.
- BF16 weights, 4,096-token capacity, up to 8 sequences and GPU memory utilization 0.88.
- Host `127.0.0.1`, port 8000 and request logging disabled by the launch flag.

[Earlier reproduction notes](../reproducibility.md) associate that launcher with vLLM 0.29.0. The November thread records a later frozen runtime. The launcher was inspected and shell-parsed for this draft; its flags, model availability, GPU fit and actual runtime behavior were not exercised. Do not install the latest runtime and assume historical equivalence.

For a future approved remote setup, an operator could forward a workstation port to an already-running remote loopback server:

```bash
# Template only: not executed. Replace the SSH alias with an approved one.
ssh -N -L 18080:127.0.0.1:8000 APPROVED_GPU_HOST
```

Then the client base would be `http://127.0.0.1:18080/v1`. The first `127.0.0.1` is local to the client; the forwarding destination is interpreted on the remote host. With a direct local server on port 8000, the base could instead be `http://127.0.0.1:8000/v1`. Neither URL causes a server to start.

## Substituting a compatible endpoint

For the main pipeline, change `llm.api_base`, `llm.model` and model-specific decoding in a new config. Keep embedding settings and index identity consistent if only chat serving changes. Allocate a fresh prediction directory to prevent mixing runs; changing a URL in an unfinished checkpoint can cause a fingerprint refusal. Do not bypass that check.

The following is an **optional future synthetic live sequence**. It requires both services to be ready. It uses the generated fictional workbook and distributed fictional-example codebook. `APPROVED_SERVED_MODEL_ID` is a placeholder. The generation command was checked locally; the build/infer commands below were help-checked only and were not dispatched to live services.

```bash
# Run the offline quickstart first; DEMO_DIR is set there.
uv run --frozen python - "$DEMO_DIR" <<'PY'
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load((Path(sys.argv[1]) / 'demo-config.yaml').read_text())
cfg['io']['artifacts_dir'] = 'artifacts/endpoint-demo-a'
cfg['io']['index_artifacts_dir'] = 'artifacts/endpoint-demo-index'
cfg['embedding']['model'] = 'qwen3-embedding:4b'
cfg['llm'] = {
    'api_base': 'http://127.0.0.1:18080/v1',
    'model': 'APPROVED_SERVED_MODEL_ID',
    'temperature': 0.0, 'max_tokens': 10, 'binary_max_retries': 2,
}
Path('config/endpoint-demo.private.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

# Future only: these commands call the services. Not run for this draft.
export OLLAMA_HOST=http://127.0.0.1:11434
uv run --frozen ptg --config config/endpoint-demo.private.yaml build-index --semester F25 --teams 1,2 --meets 1,2
uv run --frozen ptg --config config/endpoint-demo.private.yaml infer --semester S26 --team 1 --meet 1 --pilot-row-limit 3 --lm-workers 2
uv run --frozen ptg --config config/endpoint-demo.private.yaml export --semester S26 --team 1 --meet 1 --pilot 3
```

Add the selected model's validated thinking controls/token allowance before live use; the minimal generic settings above are not a Granite reproduction recipe. The CLI places that pilot's predictions under `artifacts/endpoint-demo-a_pilot3`. For endpoint B, create another config, change base/model/decoding and use `artifacts/endpoint-demo-b`; the same synthetic input and a still-compatible embedding index can be retained. Merely changing `LM_STUDIOS_HOST` will have no effect on a config containing a literal `llm.api_base`.

## Proposed adapter around PyTorch or another runtime

**Design only; not implemented or tested here.** PyTorch tensors/model calls alone are not an HTTP service. A wrapper would expose `POST /v1/chat/completions` and translate the checked-out client's JSON into a runtime call:

```text
receive and validate JSON/model ID
    -> apply the model's exact tokenizer chat template to system/user messages
    -> validate input + output token budget; reject overflow explicitly
    -> run model inference under a bounded request queue
    -> decode only the newly generated assistant text
    -> return {"choices": [{"message": {"content": generated_text}}]}
```

The adapter must map `max_tokens`, temperature, optional top-p, seed and supported thinking controls deliberately. Reject unsupported controls instead of silently ignoring them. Return non-success HTTP status for invalid inputs/runtime failures; do not fabricate `no` on error. Keep model identity stable, bound concurrency, avoid logging prompts, and provide an operator readiness check. Authentication must match the client's limitations or be added explicitly.

Model/tokenizer loading, device placement, memory fit, revision/quantization, generation stopping and framework dependencies remain implementation work. This draft adds no server package, runtime or production adapter. A chat wrapper also does not satisfy the independent Ollama embedding contract. Validate a proposed adapter with mocked contract tests, then a separately authorized fictional live request before any research workload.

## Troubleshooting order

| Symptom | First thing to check |
| --- | --- |
| Missing workbook/sheet or required column | Root working directory, `excel.workbook`, sheet template, `Speaker`/`Transcript`. Use the fictional fixture for troubleshooting. |
| Embedding connection refused | `OLLAMA_HOST`, service process, listening port, model availability. A chat endpoint alone is insufficient. |
| Index provenance/dimension mismatch | Workbook, codebook, embedding model and index/cache generation. Create consistent fresh artifacts; do not edit manifest hashes. |
| Chat 404 | Base ends at `/v1`, not `/v1/chat/completions`; served model name also matches. |
| Chat 401/403 | Endpoint requires auth that this client does not currently supply. |
| Chat 400 or unsupported field | Model-specific `reasoning_effort`, `chat_template_kwargs` or decoding incompatibility. Inspect sanitized errors and the actual payload. |
| Timeout or GPU memory error | Server readiness, model size/context, then concurrency. Do not repeatedly rent larger machines without measuring the actual failure. |
| Blank prediction/Flags | Embedding or parse failures, retries and coverage. Completion is not synonymous with valid decisions. |
| Endpoint switch appears ineffective | Literal YAML overrides env expectations, or old outputs/checkpoints are being viewed. Use a fresh namespace and record effective settings. |
