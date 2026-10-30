# Completed experiments and results

These results compare LLM predictions with reconciled human codes on development data. They are not independent final validation or evidence of parity with human coders. September aggregate source artifacts were checked against their workbook, codebook, retrieval-manifest and prediction hashes while preparing this repository. The October results below come from sanitized aggregates of completed private runs. The restricted underlying artifacts remain private.

October's frozen bundle metadata treated blank coder/reviewer cells as negatives. The researcher has clarified that a blank may instead represent no objection or implicit agreement during reconciliation. Coder-to-gold diagnostics based on the earlier interpretation are not valid evidence of human agreement. This does not change model scores against reconciled codes.

## October 14, 2026: Granite retrieval and context panel

One complete whole-panel repeat, `r1`, covered the same 1,256 development turns and all 21 labels across eight arms: 26,376 requests per arm and 211,008 total, with no abstentions. Its final runtime identity check passed. Earlier partial run namespaces were excluded.

The run used the same pinned Granite model as the September arm, in BF16 with vLLM 0.31 and newer canonical state/questions serialization. It is not a faithful reproduction of the September classifier, so the score difference cannot be attributed to any one change.

| Matched retrieval condition | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Dynamic retrieval (RAG) | 0.1058394161 | 0.3741935484 | 0.1650071124 | 0.1322033819 |
| No dynamic retrieval | 0.1073205402 | 0.3247311828 | 0.1613247863 | 0.1500582366 |

Meeting-level micro-F1 was **0.1669626998** with RAG versus **0.1532258065** without it for the first meeting, and **0.1627670397** versus **0.1704545455** for the second. Removing RAG did not produce a consistent micro-F1 win; macro-F1 favored no retrieval in the pooled comparison. The earlier Qwen screen therefore does not automatically transfer to this Granite workflow.

The context family yielded micro-F1 **0.1851590106** with zero preceding turns, **0.1603420631** with two, and **0.1547064306** with six. Recency ratios of one quarter, one half and three quarters yielded **0.1544554455**, **0.1506080449** and **0.1556776557**, respectively. Zero preceding turns was highest in this observed developmental run, not a validated general rule. The retrieval and context families used separate realized calls, so stochastic outputs can differ even for seemingly comparable settings.

## October 14, 2026: D1 packing and compute-path comparison

`LiquidAI/d1-3B` used its native SystemOne probability path in BF16. The two arms compared isolated single-label requests with all-label shared-state packing on the same 1,256 development turns and all 21 labels. Three fully completed same-seed repeats produced 82,896 saved responses representing 158,256 individual label decisions, with no abstentions.

| Packing arm | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Isolated single-label | 0.0642 | 0.1140 | 0.0821 | 0.0635 |
| All-label shared state | 0.0667 | 0.1161 | 0.0848 | 0.0647 |

Predicted labels were identical within each arm across the same-seed repeats. This demonstrates runtime reproducibility for these runs, not independent samples. This is a packing and compute-path comparison, not a retrieval ablation or a test isolating only question co-presence. Task quality was low on this development data; the result is not a broad verdict on the model's capabilities.

## September 24, 2026: complete development runs

Both runs use the same two reconciled development sheets, 768-row F25 retrieval corpus, original private June codebook and `qwen3-embedding:4b` embeddings. Each covers 1,256 turns and 26,376 label jobs. The index sources were four F25 sheets, not the evaluated S26/SP26 sheets; a later replay confirmed cross-semester exact-text overlap.

### Qwen engineering baseline

- Model: `qwen3.5:4b`, local Ollama/OpenAI-compatible path.
- Micro precision **0.153940**, recall **0.541935**, F1 **0.239772**.
- Supported-label macro-F1 **0.213099**.
- TP **252**, FP **1,385**, FN **213**, TN **24,526**; no abstentions.
- Coverage **26,376/26,376**.
- Mean defined sheet-label Cohen's kappa **0.166447** across 42 values.
- Per-sheet mean kappa: S26 **0.206362**; SP26 **0.126532**.

### Granite study-model development arm

- Model: `ibm-granite/granite-4.2-8b`, revision `f8de16cdcdbc6c779ca517604e050d82cc119e44`, BF16, vLLM 0.29.0.
- Temperature 1.0, top-p 0.95, max output tokens 64, seed 42, thinking disabled.
- Micro precision **0.324830**, recall **0.410753**, F1 **0.362773**.
- Supported-label macro-F1 **0.304018**.
- TP **191**, FP **397**, FN **274**, TN **25,513**; **one negative abstention**.
- Exact coverage **26,375/26,376**, approximately **99.9962%**, not literally 100%.
- Mean defined sheet-label Cohen's kappa **0.237196** across 42 values. Its one invalid decision is excluded from this kappa summary and reported separately as an abstention.
- Per-sheet mean kappa: S26 **0.322656**; SP26 **0.151737**.
- Per-sheet micro-F1: S26 approximately **0.396**; SP26 approximately **0.329**.

Granite made fewer false-positive decisions than Qwen, but detected fewer human-coded positives. Different models and serving/decoding configurations mean this is a comparison of implemented workflows, not an isolated architecture effect. Neither run uses independently reserved FA26 ground truth.

The Granite chat prompts were processed on an explicitly approved rented GPU via an SSH tunnel. Source workbook, embeddings/index, checkpoints and outputs stayed on the workstation. This is not an entirely local-inference result: prompts still left the workstation. The saved run report records teardown verification. Any new remote processing needs fresh approval.

### September 24 retrieval overlap

An offline replay used saved Granite query vectors whose fingerprints were checked against the ordered workbook transcripts and recorded embedding model. The index, metadata and workbook were checked against the historical manifest. No inference was rerun.

For **15 of 1,256** target turns, the retrieved top five contained target text matching after whitespace and case normalization: 10 turns on the first sheet and five on the second. These matches exposed a label-positive example in **22 distinct row-label decisions**. All 15 target turns were shorter than eight whitespace-delimited words. The frozen October bundle separately records **21 source-target overlap pairs** across those 15 turns and **five static-example/target pairs** across five turns.

Common short repeats may explain some exact matches. The replay establishes overlap, not that it caused a score difference or materially changed task performance. The older April/June F25-target index/target relationships have not been fully reconstructed.

## September 29, 2026: controlled local retrieval/context screen

Frozen harness commit: `6411fb4`. Native Ollama 0.34.4, local Qwen 3.5 4B Q4_K_M, temperature 0, seed 42, context 8,192, output budget 16, thinking off. The deterministic panel selects 64 turns from each of the same two development sheets, 128 total, with all 21 labels and 43 positive label instances.

Four arms each completed 2,688 decisions, 10,752 total:

- Baseline global top-five then label filter, two preceding turns: micro-F1 **0.2011**, TP 19, FP 127, FN 24.
- Remove dynamic retrieval, retain static definitions/examples: **0.2615**, TP 17, FP 70, FN 26.
- Up to five label-positive neighbors: **0.1721**, TP 21, FP 180, FN 22.
- Six preceding turns, otherwise baseline: **0.2121**, TP 21, FP 134, FN 22.

No dynamic retrieval improved F1 on both meetings, mainly by reducing overcoding. The ordering persisted after excluding the one selected row matching a static codebook example. The saved within-meeting row-bootstrap F1-difference interval was approximately +0.0068 to +0.1093, descriptive only. It is not a participant-level significance/equivalence result.

The saved confusion counts were independently recomputed and the aggregate/response artifacts checked. This screen is not a Granite improvement and must not be compared numerically with the full-sheet scores as though the populations and serving settings matched.

## Earlier results familiar to collaborators

- April archive: 619 matched F25 turns, 24 labels; frequency-weighted mean kappa **0.269943**, or **0.317820** excluding Communication.
- June archive: 2,781 matched turns across six F25 meeting sheets, 21-label framework; mean of **125 reported** sheet-label kappas **0.208094**. One sheet reports only 20 labels.
- Saved June configuration names Granite 4.1 8B Instruct. This is configuration evidence, not model provenance embedded in the prediction file.

Different populations, retrieval settings, label counts and aggregation methods prevent a controlled improvement claim over April/June. Index/target overlap in older F25-target runs has not been fully reconstructed. Static-codebook and cross-semester exact-text overlaps have also been observed. These are development results, and the overlaps do not by themselves establish the cause of a score difference.

## Evidence files and software verification

- `results/development-summary.json`: exact aggregate counts, scores and source hashes for September 24.
- `results/development-per-label.csv`: 84 sheet-label rows with both models' confusion counts and valid-decision kappas; no participant turn IDs or text.
- `PORTING_MANIFEST.json`: source commit, selected-file hashes and deliberate export omissions.
- `docs/reproducibility.md`: installation, synthetic integration smoke and authorized research reproduction instructions.

A human-human baseline requires the original individual decision records or an explicitly documented reconstruction of accepted labels and objections. The provenance and independence of those decisions also need to be established.
