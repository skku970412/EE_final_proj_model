"""Normalization and validation helpers for Korean license plate strings."""

import re

# Representative KR plate patterns; expand as needed for project-specific formats.
PLATE_PATTERNS = [
    r"\d{3}[가-힣]\d{4}",  # 3 digits + Hangul + 4 digits (current format)
    r"\d{2}[가-힣]\d{4}",  # 2 digits + Hangul + 4 digits (older format)
    r"[가-힣]\d{4}",       # Simplified pattern (e.g., special plates)
]

# Common confusions between Latin letters and digits.
SIMILAR = {
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "|": "1",
    "S": "5",
    "B": "8",
}


def normalize(text: str) -> str:
    """Remove whitespace and map visually similar characters to their numeric forms."""
    text = text.replace(" ", "").strip()
    return "".join(SIMILAR.get(ch, ch) for ch in text)


def valid_plate(text: str) -> bool:
    """Return True if the string matches a known Korean plate pattern."""
    norm = normalize(text)
    return any(re.fullmatch(pattern, norm) for pattern in PLATE_PATTERNS)
