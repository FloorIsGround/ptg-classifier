from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SheetRef:
    team: int
    meet: int
    semester: str  # e.g. "F" or "S" (with year implied in template)


@dataclass(frozen=True)
class TranscriptRow:
    row_index: int
    speaker: str
    transcript: str
    prev_lines: list[str]


@dataclass(frozen=True)
class IndexedMeta:
    id: str
    sheet_name: str
    team: int
    meet: int
    speaker: str
    transcript: str
    prev_context: str
    codes: list[str]
    code_ids: list[str]
    codes_key: str


@dataclass(frozen=True)
class CodebookExample:
    text: str
    reason: str = ""


@dataclass(frozen=True)
class CodebookEntry:
    id: str
    label: str
    category: str
    definition: str
    aliases: tuple[str, ...] = ()
    positive_examples: tuple[str, ...] = ()
    negative_examples: tuple[CodebookExample, ...] = ()

    @property
    def examples(self) -> tuple[str, ...]:
        """Backward-compatible alias used by the existing binary pipeline."""
        return self.positive_examples


@dataclass(frozen=True)
class Codebook:
    version: str
    entries: tuple[CodebookEntry, ...]

    @property
    def allowed_ids(self) -> list[str]:
        return [entry.id for entry in self.entries]

    @property
    def allowed_labels(self) -> list[str]:
        return [entry.label for entry in self.entries]

    def contains_all(self, ids: Iterable[str]) -> bool:
        allowed = set(self.allowed_ids)
        return all(i in allowed for i in ids)

    def entry_by_label(self, label: str) -> CodebookEntry:
        canonical = self.canonicalize_label(label)
        if canonical is None:
            raise KeyError(label)
        for entry in self.entries:
            if entry.label == canonical:
                return entry
        raise KeyError(label)

    def entry_by_id(self, code_id: str) -> CodebookEntry:
        for entry in self.entries:
            if entry.id == code_id:
                return entry
        raise KeyError(code_id)

    def canonicalize_label(self, raw_label: str) -> str | None:
        wanted = _normalize_label_key(raw_label)
        if not wanted:
            return None
        for entry in self.entries:
            if _normalize_label_key(entry.label) == wanted:
                return entry.label
            for alias in entry.aliases:
                if _normalize_label_key(alias) == wanted:
                    return entry.label
        return None


def _normalize_label_key(label: str) -> str:
    return " ".join(str(label).strip().casefold().split())
