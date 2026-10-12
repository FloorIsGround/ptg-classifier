# Data privacy and sharing boundaries

## Included material

Source code, configuration schemas, fictional codebook examples and regression fixtures, unexecuted optional notebook code, technical documentation, source-file hashes and aggregate-only results.

Label names, construct definitions and aliases are retained for readability. All 145 original codebook examples were replaced with fictional examples. The fictional codebook is labeled `shareable-synthetic-v1`; its historical filename is retained only for compatibility.

## Excluded material

- Participant transcript workbooks, row-level annotations and prediction exports.
- FAISS vectors/metadata, cached embeddings, checkpoints and prompt/response ledgers.
- Original codebook examples, research PDFs and example-bearing ad-hoc notebooks.
- Real environment files, credentials, personal host paths and original research Git history.
- Runtime logs, temporary files and host-specific systemd deployments.
- Email drafts, scheduling notes and internal handoff records.

The original research history contains participant data. Never copy its `.git`, push its branches/tags or add it as a remote to this repository. Deleting private files in a later commit does not remove them from earlier history.

## Local checks

`.gitignore` protects common private directories and generated binary extensions. It is not a security boundary: forced staging and copying content into text files can bypass it.

```bash
uv run --frozen python scripts/check_repository.py
# Optional local comparison against an authorized private workbook:
uv run --frozen python scripts/check_repository.py --reference-workbook /absolute/path/to/private-workbook.xlsx
```

The scanner checks tracked paths/file types, notebook outputs, credential patterns and historical private paths. The optional workbook comparison checks normalized long transcript strings in current tracked files. It emits paths/counts only. Historical blob content needs a separate audit; exact matching does not exclude paraphrases or every identifying detail.

Generated demonstration artifacts remain ignored even when fictional. `results/` must contain only aggregate summaries, never row IDs, speaker names or example quotations.

## Runtime privacy

Real inference sends transcript content to configured embedding/chat endpoints. A loopback URL may forward to a remote machine. Local workbook storage does not establish local processing.

Some debug/error paths include snippets and raw responses. Treat logs as restricted research material. External processing, provider retention and remote storage require explicit approval under the study's policy; a previous one-run approval does not authorize another run.

## Sharing and reproduction

Private repository access does not authorize public redistribution, collaborator access or new transcript processing. Confirm code ownership, construct-definition permissions, aggregate disclosure and licensing before wider sharing. No redistribution license is asserted here.

Historical score reproduction requires the frozen private inputs and original source/runtime contract described in [reproduction](reproducibility.md). Supply those only through an approved private channel, not this repository.
