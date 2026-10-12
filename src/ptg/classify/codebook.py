from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ptg.domain.models import Codebook, CodebookEntry, CodebookExample

DEFAULT_CODEBOOK_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "codebooks" / "f25_codebook_2026_06_14.yaml"
)


def load_default_codebook() -> Codebook:
    return load_codebook(DEFAULT_CODEBOOK_PATH)


def load_codebook(path: str | Path) -> Codebook:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    version = str(raw.get("version", ""))
    labels = raw.get("labels", [])
    if not version:
        raise ValueError("Codebook is missing required 'version'.")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Codebook must contain a non-empty 'labels' list.")

    entries = tuple(_parse_entry(item) for item in labels)
    _validate_unique(entries)
    return Codebook(version=version, entries=entries)


def _parse_entry(item: dict[str, Any]) -> CodebookEntry:
    if not isinstance(item, dict):
        raise ValueError("Each codebook label entry must be a mapping.")

    required = ("id", "label", "category", "definition")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"Codebook entry missing required fields: {missing}")

    positive_examples = tuple(_parse_positive_example(x) for x in item.get("positive_examples") or [])
    negative_examples = tuple(_parse_negative_example(x) for x in item.get("negative_examples") or [])

    return CodebookEntry(
        id=str(item["id"]),
        label=str(item["label"]),
        category=str(item["category"]),
        definition=str(item["definition"]),
        aliases=tuple(str(alias) for alias in (item.get("aliases") or [])),
        positive_examples=positive_examples,
        negative_examples=negative_examples,
    )


def _parse_positive_example(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(item)


def _parse_negative_example(item: Any) -> CodebookExample:
    if isinstance(item, dict):
        return CodebookExample(text=str(item.get("text", "")), reason=str(item.get("reason", "")))
    return CodebookExample(text=str(item))


def _validate_unique(entries: tuple[CodebookEntry, ...]) -> None:
    ids = [entry.id for entry in entries]
    labels = [entry.label.casefold() for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Codebook contains duplicate ids.")
    if len(labels) != len(set(labels)):
        raise ValueError("Codebook contains duplicate labels.")

    alias_to_label: dict[str, str] = {}
    for entry in entries:
        for alias in entry.aliases:
            key = " ".join(alias.strip().casefold().split())
            existing = alias_to_label.get(key)
            if existing and existing != entry.label:
                raise ValueError(f"Alias {alias!r} maps to multiple labels: {existing!r}, {entry.label!r}")
            alias_to_label[key] = entry.label


_DEFAULT = load_default_codebook()
ENTRIES: tuple[CodebookEntry, ...] = _DEFAULT.entries
LABEL_NAMES: list[str] = _DEFAULT.allowed_labels
BY_LABEL: dict[str, CodebookEntry] = {entry.label: entry for entry in ENTRIES}
