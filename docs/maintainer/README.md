# Maintainer guide

Verified 2026-11-16 against classifier revision `ef169f6e8c577dc5743a4ec5e21ce86bafd000f2`, with the accompanying maintainer demo. These guides describe the client interfaces at that revision.

Read in this order:

1. [Quickstart and repository map](01-quickstart-and-map.md): run the offline demo, follow one fictional row, understand the actual entry points and locate comparison artifacts.
2. [Using a coding agent](02-coding-agent.md): what it can help maintain, how it differs from the classifier, and a read-only walkthrough prompt.
3. [Inference setup](03-inference-setup.md): clients, servers, GPU hosting, the actual HTTP contracts, endpoint substitution and a proposed adapter.

All runnable examples here use fictional data or loopback placeholders. Run commands from the repository root. `artifacts/` holds generated files and is ignored by Git. The demo needs Python packages but no model downloads, GPU or running inference service. Dependency installation may use the network; the synthetic demo does not.

Validation used `uv sync --frozen`, `uv run --frozen pytest -q` (140 passed), the synthetic demo, config-driven export and scoring, and CLI help checks. The maintainer demo resolved 126 fictional decisions with no model-network calls. Live model serving, GPU fit and remote inference were not tested.

The current checkout and the earlier private November comparison are different artifacts. These guides distinguish them without including private artifact locations or study excerpts. The small development meeting comparisons do not establish whether RAG helps on new meetings.
