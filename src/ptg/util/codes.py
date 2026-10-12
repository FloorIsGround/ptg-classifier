from __future__ import annotations

import re
import unicodedata

import pandas as pd

from ptg.classify.codebook import load_default_codebook
from ptg.domain.models import Codebook

# Ported from the original helper_functions.py


def _normalize_codes_string(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    # Remove zero-width & bidi marks (Unicode category Cf)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    # Normalize weird spaces and comma variants
    s = re.sub(r"[\u00A0\u2007\u202F]", " ", s)
    s = re.sub(r"[，、﹐､]", ",", s)
    # Tighten spaces around commas and collapse whitespace
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_codes(cell) -> list[str]:
    s = _normalize_codes_string(cell)
    if not s:
        return []
    parts = [p.strip(" '\"\t") for p in s.split(",") if p.strip()]
    # de-dupe but preserve order (case-insensitive)
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        k = p.casefold()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def canonicalize_codes(
    raw_codes: list[str],
    codebook: Codebook | None = None,
    *,
    on_unknown: str = "raise",
) -> list[str]:
    """Map raw workbook labels/aliases to canonical codebook labels.

    By default, raises ValueError for unknown labels so tests and explicit
    migrations cannot silently mix legacy/provisional labels with the
    authoritative clean codebook. Index rebuilds pass on_unknown="drop" to
    ignore older labels that are not part of the clean codebook.
    """

    if codebook is None:
        codebook = load_default_codebook()

    seen: set[str] = set()
    out: list[str] = []
    for raw in raw_codes:
        canonical = codebook.canonicalize_label(raw)
        if canonical is None:
            if on_unknown == "drop":
                continue
            if on_unknown != "raise":
                raise ValueError(f"Unsupported on_unknown policy: {on_unknown}")
            raise ValueError(f"Unknown code label: {raw}")
        key = canonical.casefold()
        if key not in seen:
            seen.add(key)
            out.append(canonical)
    return out


def parse_and_canonicalize(cell, *, codebook: Codebook | None = None, on_unknown: str = "raise") -> list[str]:
    """Parse a workbook cell and canonicalize against the clean codebook."""
    return canonicalize_codes(parse_codes(cell), codebook, on_unknown=on_unknown)
