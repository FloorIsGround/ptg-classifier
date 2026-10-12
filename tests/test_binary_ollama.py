"""Synthetic compatibility check for a non-thinking Ollama chat endpoint."""

from __future__ import annotations

import asyncio

import httpx

from ptg.classify import binary_classifier as binary
from ptg.domain.models import CodebookEntry


def test_reasoning_disabled_only_when_configured(monkeypatch):
    sent: list[dict] = []

    def response(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": "yes"}}]})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(binary.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(response), **kwargs))
    entry = CodebookEntry("test", "Test label", "Test", "Synthetic definition")
    job = binary.BinaryClassificationJob(0, entry, [], "Synthetic line", [])

    async def exercise(**kwargs):
        return await binary.classify_binary_jobs(
            jobs=[job], api_base="http://127.0.0.1:11435/v1", model="synthetic",
            max_retries=0, **kwargs,
        )

    assert asyncio.run(exercise())[(0, "Test label")][0] == "yes"
    assert "reasoning_effort" not in sent[-1]
    assert asyncio.run(exercise(reasoning_effort="none"))[(0, "Test label")][0] == "yes"
    assert sent[-1]["reasoning_effort"] == "none"
    assert sent[-1]["max_tokens"] == 10
