"""Synthetic Granite request and isolated study configuration checks."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from ptg.classify import binary_classifier as binary
from ptg.config import load_settings
from ptg.domain.models import CodebookEntry
from ptg.pipeline.run import _run_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_granite_configuration_isolated_and_model_specific(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "qwen3-embedding:4b")
    monkeypatch.setenv("LM_STUDIOS_HOST", "http://127.0.0.1:18080/v1")
    granite = load_settings(ROOT / "config/study-21-granite-4.2.yaml")
    qwen = load_settings(ROOT / "config/study-21.yaml")
    assert granite.artifacts_dir != qwen.artifacts_dir
    assert granite.index_artifacts_dir == qwen.index_artifacts_dir
    assert granite.run_prefix == qwen.run_prefix
    assert granite.codebook_path == qwen.codebook_path
    assert granite.active_labels == qwen.active_labels
    assert granite.excluded_source_labels == qwen.excluded_source_labels
    assert granite.lm_model == "granite-4.2-8b"
    assert granite.lm_temperature == 1.0
    assert granite.lm_top_p == 0.95
    assert granite.lm_max_tokens == 64
    assert granite.lm_seed == 42
    assert granite.lm_chat_template_kwargs == {"enable_thinking": False}
    assert granite.lm_reasoning_effort is None


def test_granite_request_uses_non_thinking_fields(monkeypatch):
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "no"}}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(binary.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(respond), **kwargs))
    entry = CodebookEntry("test", "Test label", "Test", "Synthetic definition")
    job = binary.BinaryClassificationJob(0, entry, [], "Synthetic line", [])
    result = asyncio.run(binary.classify_binary_jobs(
        jobs=[job], api_base="http://127.0.0.1:18080/v1", model="granite-4.2-8b",
        temperature=1.0, top_p=0.95, max_tokens=64,
        seed=42,
        chat_template_kwargs={"enable_thinking": False}, max_retries=0,
    ))
    assert result[(0, "Test label")] == ("no", "no")
    assert sent[0]["temperature"] == 1.0
    assert sent[0]["top_p"] == 0.95
    assert sent[0]["max_tokens"] == 64
    assert sent[0]["seed"] == 42
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in sent[0]
    assert sent[0]["model"] == "granite-4.2-8b"


def test_decoding_changes_checkpoint_fingerprint_without_invalidating_qwen(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBED_MODEL", "qwen3-embedding:4b")
    monkeypatch.setenv("LM_STUDIOS_HOST", "http://127.0.0.1:11435/v1")
    monkeypatch.setenv("LM_MODEL", "qwen3.5:4b")
    qwen = load_settings(ROOT / "config/study-21.yaml")
    granite = load_settings(ROOT / "config/study-21-granite-4.2.yaml")
    # Fingerprinting hashes bytes only. Do not depend on ignored private workbooks
    # or index artifacts being present in a fresh checkout/worktree.
    from dataclasses import replace
    workbook = tmp_path / "synthetic-workbook.xlsx"
    workbook.write_bytes(b"synthetic workbook identity")
    (tmp_path / f"{qwen.run_prefix}index_manifest.json").write_text('{"synthetic": true}')
    qwen = replace(qwen, workbook_path=workbook, index_artifacts_dir=tmp_path)
    granite = replace(granite, workbook_path=workbook, index_artifacts_dir=tmp_path)
    name = "Team 1 Time 1 (S26)"
    rows = list(range(2))
    first = _run_fingerprint(qwen, name, 120.0, rows)
    assert first == _run_fingerprint(qwen, name, 120.0, rows)
    assert first != _run_fingerprint(granite, name, 120.0, rows)


@pytest.mark.parametrize("extra", [
    "  temperature: -1.0\n", "  top_p: 2.0\n", "  max_tokens: 0\n",
    "  chat_template_kwargs:\n    enable_thinking: maybe\n",
    "  chat_template_kwargs:\n    unsafe: true\n",
    "  seed: -1\n",
])
def test_invalid_decoding_config_rejected(tmp_path, extra):
    path = tmp_path / "bad.yaml"
    path.write_text("llm:\n" + extra)
    with pytest.raises(ValueError):
        load_settings(path)
