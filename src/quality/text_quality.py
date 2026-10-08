"""Text extraction quality analysis."""

import re
from typing import Dict, List


MOJIBAKE_MARKERS = (
    "â€",
    "â€“",
    "â€”",
    "â€¢",
    "Ã©",
    "Ã¨",
    "Ã¡",
    "Ã³",
    "Â",
)


def normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace without changing content meaning."""

    return re.sub(
        r"[ \t]+",
        " ",
        text or "",
    ).strip()


def character_statistics(text: str) -> Dict[str, float]:
    """Return general character-quality statistics."""

    if not text:
        return {
            "length": 0,
            "alphabetic_ratio": 0.0,
            "digit_ratio": 0.0,
            "whitespace_ratio": 0.0,
            "non_ascii_ratio": 0.0,
        }

    length = len(text)

    alphabetic = sum(
        char.isalpha()
        for char in text
    )

    digits = sum(
        char.isdigit()
        for char in text
    )

    whitespace = sum(
        char.isspace()
        for char in text
    )

    non_ascii = sum(
        ord(char) > 127
        for char in text
    )

    return {
        "length": length,
        "alphabetic_ratio": alphabetic / length,
        "digit_ratio": digits / length,
        "whitespace_ratio": whitespace / length,
        "non_ascii_ratio": non_ascii / length,
    }


def detect_mojibake(text: str) -> List[str]:
    """Detect common UTF-8/Windows-1252 decoding artifacts."""

    found = []

    for marker in MOJIBAKE_MARKERS:

        if marker in text:
            found.append(marker)

    return found


def text_quality_score(text: str) -> float:
    """Estimate extraction quality from text characteristics."""

    if not text or not text.strip():
        return 0.0

    stats = character_statistics(text)

    score = 1.0

    mojibake = detect_mojibake(text)

    if mojibake:
        score -= min(
            0.35,
            0.08 * len(mojibake),
        )

    if stats["alphabetic_ratio"] < 0.20:
        score -= 0.20

    if stats["non_ascii_ratio"] > 0.35:
        score -= 0.20

    return max(
        0.0,
        min(1.0, score),
    )