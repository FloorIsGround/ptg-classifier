# Completed experiments and results

These are measured development results, not final validation or evidence that LLM coding matches human coding. Aggregate source artifacts were checked against their workbook, codebook, retrieval-manifest and prediction hashes while preparing this repository. The restricted underlying artifacts remain private.

## September 24, 2026: complete development runs

Both runs use the same two reconciled development sheets, 768-row F25 retrieval corpus, original private June codebook and `qwen3-embedding:4b` embeddings. Each covers 1,256 turns and 26,376 label jobs.

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
- Mean defined sheet-label Cohen's kappa **0.237196** across 42 values. Its one invalid decision is excluded only for this kappa summary.
- Per-sheet mean kappa: S26 **0.322656**; SP26 **0.151737**.
- Per-sheet micro-F1: S26 approximately **0.396**; SP26 approximately **0.329**.

Granite made fewer false-positive decisions than Qwen, but detected fewer human-coded positives. Different models and serving/decoding configurations mean this is a comparison of implemented workflows, not an isolated architecture effect. Neither run uses independently reserved FA26 ground truth.

The Granite chat prompts were processed on an explicitly approved rented GPU via an SSH tunnel. Source workbook, embeddings/index, checkpoints and outputs stayed on the workstation. This is not an entirely local-inference result: prompts still left the workstation. The saved run report records teardown verification. Any new remote processing needs fresh approval.

## September 29, 2026: controlled local retrieval/context screen

Frozen harness commit: `6411fb4`. Native Ollama 0.34.4, local Qwen 3.5 4B Q4_K_M, temperature 0, seed 42, context 8,192, output budget 16, thinking off. The deterministic panel selects 64 turns from each of the same two development sheets, 128 total, with all 21 labels and 43 positive label instances.

Four arms each completed 2,688 decisions, 10,752 total:

- Baseline global top-five then label filter, two preceding turns: micro-F1 **0.2011**, TP 19, FP 127, FN 24.
- Remove dynamic retrieval, retain static definitions/examples: **0.2615**, TP 17, FP 70, FN 26.
- Up to five label-positive neighbors: **0.1721**, TP 21, FP 180, FN 22.
- Six preceding turns, otherwise baseline: **0.2121**, TP 21, FP 134, FN 22.

No dynamic retrieval improved F1 on both meetings, mainly by reducing overcoding. The ordering persisted after excluding the one selected row matching a static codebook example. The saved within-meeting row-bootstrap F1-difference interval was approximately +0.0068 to +0.1093, descriptive only. It is not a participant-level significance/equivalence result.

The parent independently recomputed confusion counts and verified the saved aggregate/response artifacts. This screen is not a Granite improvement and must not be compared numerically with the full-sheet scores as though the populations and serving settings matched.

## Earlier results familiar to collaborators

- April archive: 619 matched F25 turns, 24 labels; frequency-weighted mean kappa **0.269943**, or **0.317820** excluding Communication.
- June archive: 2,781 matched turns across six F25 meeting sheets, 21-label framework; mean of **125 reported** sheet-label kappas **0.208094**. One sheet reports only 20 labels.
- Saved June configuration names Granite 4.1 8B Instruct. This is configuration evidence, not model provenance embedded in the prediction file.

Different populations, retrieval settings, label counts and aggregation methods prevent a controlled improvement claim over April/June. Historical F25 index/target overlap has not been fully reconstructed and must be audited before treating those results as independent evaluation.

## Evidence files and software verification

- `results/development-summary.json`: exact aggregate counts, scores and source hashes for September 24.
- `results/development-per-label.csv`: 84 sheet-label rows with both models' confusion counts and valid-decision kappas; no participant turn IDs or text.
- `PORTING_MANIFEST.json`: source commit, selected-file hashes and deliberate export omissions.
- `docs/reproducibility.md`: installation, synthetic integration smoke and authorized research reproduction instructions.

Human-human agreement and a matched statistical LLM-human comparison have **not** been established here. The existing individual-coder columns require confirmation of independent coding and completeness before they can be used for that comparison.
