from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
from ollama import Client

import re

log = logging.getLogger(__name__)

# Regex to strip non-printable control characters (keeps tabs/newlines)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\u2060\uffff]")


def _sanitize(text: str) -> str:
    """Remove characters that cause Ollama JSON serialization errors."""
    return _CONTROL_CHARS.sub("", text).strip()


def get_ollama_client() -> Client:
    host = os.getenv("OLLAMA_HOST")
    if not host:
        raise RuntimeError("OLLAMA_HOST is not set")
    return Client(host=host)


def _parse_vec(resp) -> np.ndarray:
    if "embeddings" in resp:
        return np.asarray(resp["embeddings"][0], dtype=np.float32)
    if "embedding" in resp:
        return np.asarray(resp["embedding"], dtype=np.float32)
    raise RuntimeError(f"Unexpected embed response keys: {list(resp.keys())}")


def embed_text(text: str, model: str) -> np.ndarray:
    text = _sanitize(text)
    if not text:
        raise ValueError("Empty input after sanitization")
    client = get_ollama_client()
    resp = client.embed(model=model, input=text)
    return _parse_vec(resp)


def embed_many(
    texts: Sequence[str],
    model: str,
    sheet: str = "",
    payloads: list[dict] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Embed a list of texts. Returns (vectors, good_indices).

    Attempts a single batch request first. If that fails (e.g. one text
    produces a NaN embedding), falls back to one-at-a-time embedding,
    skipping any text whose vector contains NaN and logging a warning.

    good_indices is the subset of input positions that produced valid vectors,
    so callers can align payloads to the returned matrix.
    """
    texts = [_sanitize(t) for t in texts]
    if not texts or any(not text for text in texts):
        raise ValueError("Empty embedding input after sanitization; input positions must remain stable")
    client = get_ollama_client()

    # --- fast path: single batch request ---
    try:
        resp = client.embed(model=model, input=texts)
        if "embeddings" in resp:
            vecs = np.asarray(resp["embeddings"], dtype=np.float32)
        elif "embedding" in resp:
            vecs = np.asarray([resp["embedding"]], dtype=np.float32)
        else:
            raise RuntimeError(f"Unexpected embed response keys: {list(resp.keys())}")

        if vecs.ndim != 2 or len(vecs) != len(texts):
            raise ValueError(f"Embedding response length/shape mismatch: {len(vecs)}/{len(texts)}")
        if np.isfinite(vecs).all():
            return vecs, list(range(len(texts)))

        log.warning("Batch embed returned NaN values for sheet %r; falling back to one-at-a-time", sheet)
    except Exception as e:
        log.warning("Batch embed failed for sheet %r (%s); falling back to one-at-a-time", sheet, e)

    # --- slow path: embed individually with retries, skip persistently bad rows ---
    _MAX_RETRIES = 3
    vecs_out: list[np.ndarray] = []
    good: list[int] = []
    for i, text in enumerate(texts):
        text = _sanitize(text)
        if not text:
            context = ""
            if payloads and i < len(payloads):
                p = payloads[i]
                context = f" | speaker={str(p.get('speaker','?'))} transcript={str(p.get('transcript',''))[:80]!r}"
            log.debug("Empty after sanitization — skipping index=%d%s", i, context)
            continue

        context = ""
        if payloads and i < len(payloads):
            p = payloads[i]
            context = f" | speaker={str(p.get('speaker','?'))} transcript={str(p.get('transcript',''))[:80]!r}"

        vec = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = client.embed(model=model, input=text)
                v = _parse_vec(resp)
                if not np.isfinite(v).all():
                    log.debug("NaN on attempt %d — sheet=%r index=%d%s", attempt, sheet, i, context)
                    continue
                vec = v
                break
            except Exception as e:
                log.debug("Embed error attempt %d — sheet=%r index=%d (%s)", attempt, sheet, i, e)

        if vec is None:
            log.warning("Skipping after %d attempts — sheet=%r index=%d%s", _MAX_RETRIES, sheet, i, context)
            continue

        vecs_out.append(vec)
        good.append(i)

    if not vecs_out:
        raise RuntimeError(f"All texts in batch failed to embed (sheet={sheet!r})")

    return np.vstack(vecs_out).astype("float32"), good
