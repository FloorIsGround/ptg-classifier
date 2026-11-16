# Using an AI coding agent on this project

A coding agent is a development assistant that can inspect files, explain code and, when allowed, run tools or propose edits. Here it can help a new maintainer find `ptg.cli:main`, trace configuration into `infer_sheet`, check dependency versions, explain an error and identify which service is missing. It can also compare documentation with CLI help and propose a small fix with an appropriate synthetic check.

The inference model has a different job. It receives a prepared coding question and returns a label decision. In this pipeline an embedding model finds related text; a chat model answers yes/no. Neither is the coding agent. Installing or opening a coding agent does not download those models, start their servers, create an index or establish scientific validity.

An agent's explanation is a hypothesis until checked against source or an executed test. Ask for exact file/function references and evidence levels. For example, a mocked HTTP test can prove payload formatting, but cannot prove that a rented GPU has enough memory or that a particular server honors a thinking control.

A useful troubleshooting request includes the command, sanitized exception, Python version and config **keys** involved. Do not paste `.env`, request logs, workbook rows, participant names, private service addresses or full environment dumps. Some pipeline errors can echo input fragments or raw replies. Review excerpts before submitting them to any coding assistant. Even a read-only agent may send the source context it reads to its provider; use only this sanitized checkout under the team's approved tooling policy.

## Example prompt: read-only, no private data

```text
Walk me through this sanitized PTG checkout as a new Python maintainer.
Do not edit, install, execute Python, run tests, start services, call endpoints,
download models, use cloud tools or publish anything.
Read only tracked source/config/docs/tests needed for the walkthrough.
Do not read .env, ignored or untracked files, data/, artifacts/, private/,
research-inputs/, credentials, local session history or files outside this repo.

Identify the actual CLI entry point and trace build-index, infer and export.
Explain the config fields and workbook columns those calls consume.
Distinguish embedding calls, FAISS retrieval and chat-model calls.
Explain the difference between the main HTTP client and the controlled harness.
Use only synthetic fixture text already tracked in tests or scripts.
Give relative file/function references, label every inference as an inference,
and list any uncertain claims for me to verify. Suggest commands as text only.
If evidence is missing or contradictory, stop that claim and report the gap.
```

After this walkthrough, a maintainer can separately authorize a specific synthetic check or bounded edit. Keep that authorization narrower than permission to read a whole workstation or process research data.
