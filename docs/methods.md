# Methods and data separation

## Target task

Multi-label coding is implemented as one yes/no decision per nonempty transcript turn per active label. The clean study codebook has 21 labels. Historical 24-label or provisional 23-label runs are not interchangeable with the current study arm.

## Reference examples, development and final evaluation

### Dynamic reference corpus: F25

The approved retrieval corpus contains 768 eligible coded, nonempty turns from:

- `Team 1 Time 1 (F25)`
- `Team 2 Time 1 (F25)`
- `Team 1 Time 2 (F25)`
- `Team 2 Time 2 (F25)`

The September 24 index manifest and index contain examples from these four F25 sheets only; neither evaluated S26/SP26 sheet is an index source. A completed offline replay found normalized exact target-text matches in the retrieved top five for 15 of 1,256 turns. Every matched target was shorter than eight whitespace-delimited words. See [results](results.md) for the counts and limits.

Only rows with at least one retained current label are indexed. Retired `Communication` and `Service Orientation` codes are excluded from metadata; excluded-only rows do not enter retrieval. F25 Time 3 is not in the current index and can be used only as an explicitly labeled temporal development check. It is not an independent final test.

### Static reference examples: the research codebook

The original June codebook contains fixed positive and boundary examples. These are also reference examples, even in the no-dynamic-retrieval arm. They must be audited separately from the F25 retrieval corpus.

An audit found exact long-example matches in S25, F25 and S26 sheets, including two matched example entries on `Team 1 Time 1 (S26)`. Therefore it is not correct to say **all prompt examples** came exclusively from F25 or that the full S26 development population was completely example-unseen. S25 is excluded from dynamic indexing and scoring, but some static research-codebook examples match S25 text. The 128-row controlled panel contained one row matching a static example; the paired sensitivity analysis excluded it from all arms without replacing the panel.

Index/target overlap in earlier April/June F25-target runs has not been fully reconstructed, unlike the completed September retrieval replay. The observed overlaps do not by themselves establish the cause of any score gap.

All bundled examples in this repository are newly written fictional replacements. Historical aggregate scores used the original private codebook, not these replacements.

### Development data: S26/SP26

The two explicitly reconciled sheets are:

- `Team 1 Time 1 (S26)`: 627 nonempty turns, 242 retained positive label instances.
- `Team 2 Time 1 (SP26)`: 629 nonempty turns, 223 retained positive label instances.

Total: 1,256 turns, 26,376 row-label decisions and 465 positive instances. Reviewed blank reconciled cells count as no current code on these approved sheets only. Rows with only retired labels remain negative for the 21-label denominator. Empty transcript rows are not scored.

These data have already informed inspection and workflow selection. They are development data, not an untouched final evaluation. Confirm the actual cohort/recording mapping behind `S26` and `SP26` before making population-level claims.

### Proposed final holdout: FA26

Three Time 1 sheets contain 2,039 transcript turns and no reconciled labels in the saved workbook. They are not all-negative ground truth. FA26 remains reserved from prompt/example/model selection and new inference; independent final evaluation has not been performed. Confirm the period/cohorts, recording completeness, participant independence and blinded annotation plan before final testing.

Freeze the workflow and inclusion rules before receiving the final labels. Independent coders should be blinded to LLM outputs and each other's annotations; preserve their original labels before reconciliation.

## Metrics

For each label and sheet, report support and TP/FP/FN/TN, plus abstentions. Micro precision, recall and F1 pool label decisions. In the September 24 report, supported-label macro-F1 excludes labels with zero human-positive support; the pooled development set has 19 supported labels. Report zero-support false positives separately.

The no-code baseline has micro-F1 0 and approximately 98.2% label-instance accuracy on this imbalanced development set. High raw accuracy therefore does not establish useful coding.

Cohen's kappa measures chance-corrected binary agreement. The current descriptive summary is the arithmetic mean of defined per-label, per-sheet kappas, not a single pooled kappa and not a frequency-weighted kappa. Granite's one invalid negative decision is excluded for this kappa calculation and reported separately as an abstention. F1 and coverage retain the abstention policy recorded in the aggregate evaluator.

## Human comparison and baseline

The reported F1 and Cohen's kappa already compare LLM predictions with reconciled human codes. On the approved development sheets, a blank reconciled cell means no category code. This policy does not apply to coder/reviewer entries.

The researcher reports that each coder labeled all transcript rows before reconciliation and that all rows were reviewed. During reconciliation, a coder or reviewer could leave an entry blank when agreeing, while other labels or objections could be noted or discussed. A blank coder/reviewer entry may therefore mean no objection or implicit agreement; it does not by itself establish a negative decision or indicate that a row was unreviewed. These columns are not dense independent label vectors.

A human-human baseline requires original individual decision records or an explicitly documented reconstruction of accepted labels and objections. Assess the provenance and independence of those decisions before interpreting a baseline. Reconciliation is not independent of its contributing coders, so coder-versus-reconciliation scores need that caveat.

## Statistical interpretation

The September controlled screen uses paired row-level comparisons within two observed meetings. Its row-bootstrap intervals are descriptive and conditional on those meetings; they do not establish meeting/participant-level generalization. More turns from the same two meetings do not create more independent meetings. The three same-seed October D1 repeats yielded identical predictions within each arm and are not independent samples; the October Granite panel has one complete repeat.

Prespecify a primary endpoint, matched comparison population, clustering unit and acceptable performance/equivalence margin with the research team. Use meeting/team-level resampling only when enough independent groups exist; otherwise report descriptive results and the limited scope. Do not claim equivalence because a difference is nonsignificant.
