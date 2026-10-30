# Export a workbook with an AI coding column

Keep a reusable JSON profile in a private directory outside Git. Copy
[`config/examples/ai-workbook-profile.json`](../config/examples/ai-workbook-profile.json)
there, replace its fictional paths, and set the profile's permissions to `0600`
and its directory's permissions to `0700`. Existing `*.private.json` ignore rules
provide an additional safeguard. The example contains no real study data.

From this checkout, ordinary export is one command:

```sh
uv run --frozen python scripts/export_ai_workbook.py --profile /absolute/private/review.private.json
```

The profile keeps the workbook path, output path, saved journal, exact arm, and
repeat. Every relative path, including `--output`, resolves from the profile's
directory. The selected sheet and physical column mappings come from the frozen
conversion inside the journal. No codebook file or model connection is needed.
An optional `source.plan_sha256` pins a profile to a previously inspected plan.

The command makes a new `.xlsx` and a sibling `<filename>.xlsx.ai.json` sidecar.
It inserts a header named **AI** immediately after the two adjacent human coder
columns. For human columns 5 and 6, AI is column 7 and the existing column 7 moves
to column 8. Duplicate header names do not affect the physical mapping.

Each originally selected, nonempty transcript row receives one comma-separated
list of predicted category names in the frozen active label order. Numeric turns
are supported. An evaluated AI row with no positive decisions is blank. Original
physical rows are retained; blank transcript rows are not compacted or assigned
predictions.

Human coder/reviewer and reconciled cells are copied exactly. A blank reviewer
entry may mean implicit agreement with accepted labels. These reviewer fields
must not be interpreted as dense independent label sets or their blanks as
negative decisions. The exporter does not reconstruct, union, or score human
labels, and frozen annotation metadata does not verify independent coding.
Reconciled labels remain ground truth; empty reconciled cells still mean no
category in the existing model scoring workflow. This exporter performs no
scoring.

The exporter only reads an existing `granite-live-openai/3` SQLite journal, in
immutable read-only mode. It validates plan, bundle, mapping, population, request,
and selected result hashes and identities. It requires the full planned
row-by-label panel for one explicit arm and repeat. It accepts only frozen
S26/SP26 development target mappings, never holdout mappings. It makes **zero
inference requests** and performs no arm selection, threshold tuning, uploads,
or service changes.

Outputs never overwrite an existing workbook or sidecar. For another copy, use
a new filename:

```sh
uv run --frozen python scripts/export_ai_workbook.py --profile /absolute/private/review.private.json --output review-02.xlsx
```

Missing results and saved abstentions cause refusal by default. For an intentional
partial review, add `--allow-partial`. Every incomplete row then displays
`[AI INCOMPLETE: ...; see sidecar]`; it cannot be mistaken for a negative row.
The sidecar records unavailable labels and their statuses. Corrupt hashes,
duplicate events, malformed predictions, and mismatched row identities still
cause refusal. The CLI prints only status and aggregate counts. Errors do not
include source cells or model replies.

The private sidecar records source/output workbook hashes, plan and result
snapshot identities, model revision/runtime, chosen arm/repeat, active labels,
coverage counts, original physical row IDs/positions, and modified package parts.
It contains no transcript text, participants, or raw replies. Both output files
are `0600` in a `0700` directory outside Git. Symlink paths and hard-linked inputs
are refused. Atomic publication without replacement currently requires Linux
`renameat2`. The workbook is staged, reopened, and checked before publication;
failed validation leaves no final workbook.

Preservation uses direct ZIP/XML edits, without an openpyxl load/save round trip.
Untouched members retain their exact compressed bytes, local headers, and XML,
including unselected sheets and shared strings. New AI cells use inline strings
and copy the neighboring human cell style and explicit column width. The exporter shifts existing cells,
column formats, validation/filter ranges, frozen panes, ordinary A1 formulas and
defined names, comment references, threaded comment references, drawing column
anchors, and legacy comment anchors. Formulas on other sheets referring to an
inserted column are updated. Unaffected formulas and strings remain unchanged.

This is deliberately a narrow insertion tool, not a general Excel engine. It
fails safely on selected-sheet merges, tables, unknown extensions/objects,
protection, populated document tasks, affected shared/array formulas, charts
referring to selected sheets, dynamic text references such as `INDIRECT`, external
and 3D references, unscoped relative global names, calculation chains, pivot or
slicer packages, signatures, macros, encryption, ZIP64, and Excel column overflow.
Unrelated opaque package members survive unchanged. Verification checks package
integrity and the physical cells/references; it does not claim a manual Excel UI
review of the result.

Synthetic checks:

```sh
uv run --frozen pytest -q tests/test_ai_workbook.py tests/test_ai_workbook_structure.py
```
