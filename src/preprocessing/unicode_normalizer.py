"""
Centralized Unicode and text normalization for resume documents.

Responsibilities
----------------
1. Normalize Unicode representation.
2. Repair common UTF-8/Windows-1252 mojibake when safe.
3. Normalize whitespace.
4. Normalize common bullet and dash variants.
5. Preserve technical tokens and meaningful punctuation.
6. Never perform semantic/spelling correction.

This module is intentionally domain-independent.
It must work for resumes from different industries and layouts.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------
# Common mojibake patterns
# ---------------------------------------------------------------------
#
# These occur when UTF-8 bytes are incorrectly decoded as a legacy
# single-byte encoding such as Windows-1252.
#
# We only attempt repair when suspicious characters are actually
# present. Normal Unicode text is returned unchanged.
# ---------------------------------------------------------------------

_MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "â€",
    "â€“",
    "â€”",
    "â€¢",
    "â–",
    "âœ",
    "â„",
    "ðŸ",
    "�",
)


# ---------------------------------------------------------------------
# Canonical punctuation mappings
# ---------------------------------------------------------------------

PUNCTUATION_MAP = {
    # Bullets
    "\u2022": "•",
    "\u2023": "•",
    "\u25E6": "•",
    "\u2043": "•",
    "\u2219": "•",
    "\u25AA": "•",
    "\u25AB": "•",
    "\u25B6": "•",
    "\u25BA": "•",

    # Dashes
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "–",
    "\u2014": "—",

    # Quotes
    "\u2018": "'",
    "\u2019": "'",
    "\u201C": '"',
    "\u201D": '"',

    # Spaces
    "\u00A0": " ",
    "\u2007": " ",
    "\u202F": " ",
}


def _has_mojibake(text: str) -> bool:
    """Return True when text contains common mojibake markers."""

    return any(
        marker in text
        for marker in _MOJIBAKE_MARKERS
    )


def _repair_mojibake(text: str) -> str:
    """
    Conservatively repair common UTF-8 mojibake.

    The repair is only accepted when the resulting text contains
    fewer known mojibake markers than the original.
    """

    if not text or not _has_mojibake(text):
        return text

    candidates = []

    # ------------------------------------------------------------
    # Attempt 1: Windows-1252 -> UTF-8
    # ------------------------------------------------------------

    try:
        candidates.append(
            text.encode("cp1252").decode("utf-8")
        )
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # ------------------------------------------------------------
    # Attempt 2: Latin-1 -> UTF-8
    # ------------------------------------------------------------

    try:
        candidates.append(
            text.encode("latin1").decode("utf-8")
        )
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    if not candidates:
        return text

    def mojibake_score(value: str) -> int:
        return sum(
            value.count(marker)
            for marker in _MOJIBAKE_MARKERS
        )

    original_score = mojibake_score(text)

    # Only accept a repair that actually reduces corruption.
    valid_candidates = [
        candidate
        for candidate in candidates
        if mojibake_score(candidate) < original_score
    ]

    if not valid_candidates:
        return text

    return min(
        valid_candidates,
        key=mojibake_score,
    )


def normalize_unicode(
    text: Optional[str],
) -> str:
    """
    Normalize a single text string.

    Pipeline:
        None/empty
            ↓
        Unicode NFC normalization
            ↓
        conservative mojibake repair
            ↓
        Unicode normalization again
            ↓
        punctuation normalization
            ↓
        whitespace normalization
    """

    if not text:
        return ""

    text = str(text)

    # Canonical Unicode composition.
    text = unicodedata.normalize(
        "NFC",
        text,
    )

    # Repair obvious encoding corruption.
    text = _repair_mojibake(text)

    # Normalize again after possible byte conversion.
    text = unicodedata.normalize(
        "NFC",
        text,
    )

    # Canonical punctuation.
    for source, target in PUNCTUATION_MAP.items():
        text = text.replace(
            source,
            target,
        )

    # Normalize tabs and unusual whitespace.
    text = text.replace(
        "\t",
        " ",
    )

    text = re.sub(
        r"[ \u00A0]+",
        " ",
        text,
    )

    # Normalize excessive blank lines while preserving line structure.
    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_lines(
    text: Optional[str],
) -> list[str]:
    """
    Normalize a multiline document while preserving line boundaries.
    """

    normalized = normalize_unicode(text)

    if not normalized:
        return []

    return [
        line.strip()
        for line in normalized.splitlines()
        if line.strip()
    ]


def normalize_for_matching(
    text: Optional[str],
) -> str:
    """
    Produce a normalized representation suitable for matching.

    This function is intentionally separate from normalize_unicode()
    because matching text and display text have different requirements.

    Example:
        'Data   Science'
        ->
        'data science'
    """

    normalized = normalize_unicode(text)

    normalized = normalized.casefold()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()