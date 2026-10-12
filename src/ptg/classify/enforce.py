from __future__ import annotations

import json
import re
from typing import Literal

from ptg.domain.errors import ValidationError
from ptg.domain.models import Codebook


def parse_id_list(text: str) -> list[str]:
    """Parse a model output that should be a JSON list of strings."""
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValidationError(f"Expected JSON list, got parse error: {e}") from e
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValidationError("Expected JSON list of strings")
    return obj


def validate_ids(ids: list[str], codebook: Codebook) -> None:
    allowed = set(codebook.allowed_ids)
    bad = [i for i in ids if i not in allowed]
    if bad:
        raise ValidationError(f"Unknown code ids: {bad}")


def parse_yes_no(text: str) -> Literal["yes", "no"] | None:
    """Normalise a binary LM response to 'yes', 'no', or None (invalid).

    Handles case variations and trailing punctuation. Returns None if the
    response cannot be resolved to yes or no — caller should refire.
    """
    cleaned = re.sub(r"[^\w]", "", text.strip()).casefold()
    if cleaned == "yes":
        return "yes"
    if cleaned == "no":
        return "no"
    return None
