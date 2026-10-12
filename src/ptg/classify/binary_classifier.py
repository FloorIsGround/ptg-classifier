from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import httpx
from tqdm.asyncio import tqdm

from ptg.classify.enforce import parse_yes_no
from ptg.domain.models import CodebookEntry

log = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a careful assistant that labels transcript lines with existing codes."


@dataclass(frozen=True)
class BinaryClassificationJob:
    row_index: int
    entry: CodebookEntry
    retrieved_examples: list[str]
    line_to_analyze: str
    prev_lines: list[str]

    @property
    def label_name(self) -> str:
        return self.entry.label


def build_binary_prompt(job: BinaryClassificationJob) -> tuple[str, str]:
    """Build the per-label binary prompt used for one transcript-line decision."""

    entry = job.entry
    sections = [
        "You are assisting with transcript coding.",
        "",
        f"Target label: {entry.label}",
        f"Category: {entry.category}",
    ]

    if entry.definition.strip():
        sections.append(f"Definition: {entry.definition}")

    if entry.positive_examples:
        sections.extend(
            [
                "",
                "Codebook examples where this IS the label:",
                *[f"- {item}" for item in entry.positive_examples],
            ]
        )

    if entry.negative_examples:
        sections.extend(
            [
                "",
                "Codebook boundary examples where this is NOT the label:",
                *[
                    f"- {example.text} ({example.reason})" if example.reason else f"- {example.text}"
                    for example in entry.negative_examples
                ],
            ]
        )

    sections.extend(
        [
            "",
            "Retrieved previously-coded transcript examples for this label:",
            _format_block(job.retrieved_examples, empty="(none found)"),
            "",
            "Previous lines:",
            _format_block(job.prev_lines, empty="(none)", bullets=True),
            "",
            "Current line to analyze:",
            job.line_to_analyze,
            "",
            f'Does the current line exhibit "{entry.label}"?',
            'Answer ONLY with the single word "yes" or "no". No explanation.',
            f'CRITICAL: Only answer "yes" if the line CLEARLY and UNAMBIGUOUSLY demonstrates "{entry.label}".',
        ]
    )

    return SYSTEM_PROMPT, "\n".join(sections)


def _format_block(items: list[str], *, empty: str, bullets: bool = False) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return empty
    if bullets:
        return "\n".join(f"- {item}" for item in cleaned)
    return "\n\n".join(cleaned)


async def classify_binary_jobs(
    *,
    jobs: list[BinaryClassificationJob],
    api_base: str,
    model: str,
    max_concurrency: int = 4,
    max_retries: int = 2,
    timeout_s: float = 120.0,
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    max_tokens: int = 10,
    chat_template_kwargs: dict[str, bool] | None = None,
    seed: int | None = None,
) -> dict[tuple[int, str], tuple[Literal["yes", "no"] | None, str]]:
    """Run per-label binary classification concurrently via OpenAI-compatible HTTP.

    Returns a mapping of (row_index, label_name) -> (parsed, raw_response).
    parsed is "yes" | "no" | None — None means retries were exhausted.
    raw_response is the last raw string returned by the LM (or error sentinel).
    """
    api_base = api_base.rstrip("/")
    url = f"{api_base}/chat/completions"
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    out: dict[tuple[int, str], tuple[Literal["yes", "no"] | None, str]] = {}

    log.info("classify_binary_jobs: %d jobs, max_concurrency=%d", len(jobs), max_concurrency)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        limits=httpx.Limits(
            max_connections=max_concurrency + 16,
            max_keepalive_connections=max_concurrency + 4,
        ),
    ) as client:

        async def one(job: BinaryClassificationJob) -> None:
            system_prompt, user_prompt = build_binary_prompt(job)
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort
            if top_p is not None:
                payload["top_p"] = top_p
            if chat_template_kwargs is not None:
                payload["chat_template_kwargs"] = chat_template_kwargs
            if seed is not None:
                payload["seed"] = seed

            raw = ""
            parsed: Literal["yes", "no"] | None = None

            for attempt in range(max_retries + 1):
                async with sem:
                    try:
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        raw = data["choices"][0]["message"].get("content") or ""
                    except Exception as exc:
                        raw = f"__HTTP_ERROR__: {exc}"
                        if attempt < max_retries:
                            log.warning(
                                "row=%d label=%r attempt=%d HTTP error, retrying: %s",
                                job.row_index, job.label_name, attempt, exc,
                            )
                            continue
                        log.error(
                            "row=%d label=%r retries exhausted after HTTP error: %s",
                            job.row_index, job.label_name, exc,
                        )
                        break

                parsed = parse_yes_no(raw)
                if parsed is not None:
                    break

                if attempt < max_retries:
                    log.warning(
                        "row=%d label=%r attempt=%d invalid response %r, retrying",
                        job.row_index, job.label_name, attempt, raw,
                    )
                else:
                    log.error(
                        "row=%d label=%r retries exhausted, last response: %r",
                        job.row_index, job.label_name, raw,
                    )

            out[(job.row_index, job.label_name)] = (parsed, raw)

        await tqdm.gather(
            *(one(j) for j in jobs),
            total=len(jobs),
            desc="classifying",
            unit="label",
        )

    invalid_count = sum(1 for parsed, _ in out.values() if parsed is None)
    if invalid_count:
        log.error(
            "classify_binary_jobs: %d / %d labels had exhausted retries",
            invalid_count, len(jobs),
        )
    else:
        log.info("classify_binary_jobs: all %d jobs resolved", len(jobs))

    return out
