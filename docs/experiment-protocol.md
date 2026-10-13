# Native D1 decision-model feasibility and packing comparison

Branch: `research/liquid-d1-native`. Shared implementation and all admission gaps:
[development experiment queue](research-experiments.md).

## Question and controls

Can the pinned official Liquid D1-3B native decision path run within the local
resource budget, and does shared-state question packing change decisions relative
to isolated questions? Use `d1_single_label` and `d1_shared_state` with the same
frozen target rows and all 21 labels. Compare separate panels bound to the same
admitted native artifact, runtime and input population. The existing synthetic
factorial panel exercises both packing shapes but does not prove native batching
or local GPU fit.

Use native P(yes), not generated yes/no text or chat-token probability proxies.
Verify exact label order, vector length and finite values in [0,1]. Keep the
initial 0.5 threshold fixed. Domain calibration is not established. Threshold
selection belongs on a declared development-selection panel, never the final
holdout. Packing changes the effective classifier, even if weights stay fixed.

## Executable fictional boundary checks

```bash
uv sync --frozen
uv run --frozen python scripts/classifier_suite_v2.py prepare --synthetic --recipes d1_single_label --out artifacts/synthetic-d1-single
uv run --frozen python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-d1-single --repeat first
uv run --frozen python scripts/classifier_suite_v2.py report --out artifacts/synthetic-d1-single --repeats first
uv run --frozen python scripts/classifier_suite_v2.py prepare --synthetic --recipes d1_shared_state --out artifacts/synthetic-d1-shared
uv run --frozen python scripts/classifier_suite_v2.py run --synthetic --out artifacts/synthetic-d1-shared --repeat first
uv run --frozen python scripts/classifier_suite_v2.py report --out artifacts/synthetic-d1-shared --repeats first
```

Each output directory must be fresh. These commands use no model weights and
produce no real probabilities. The proposed `{labels, noul}` bridge envelope is
an internal contract, not an asserted live-server API.

## Native admission sequence

1. Review the pinned custom model code and exact dependency compatibility before
   any remote-code execution. Build a dedicated native probability adapter.
2. Establish tokenizer/template bounds and measure load/runtime memory using
   fictional inputs. Preserve existing GPU services; do not infer fit from
   checkpoint size. A chat/GGUF artifact is not automatically equivalent to the
   native probability implementation.
3. Verify actual probability shape/order, strict contract failures and repeated
   requests on fictional inputs. Record native engine/artifact identity.
4. Only then admit a frozen small development panel under the native job cap,
   with reviewed extraction and scientific reporting. Full-development work
   still needs a scalable journal. Fall 2026 remains reserved.

Real D1 inference is currently hard-blocked and the live adapter is
unimplemented. No model download, service deployment, GPU interruption or
remote study processing is performed by this branch.
