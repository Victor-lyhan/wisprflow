"""Tooth reference extraction.

Scope note: this covers the Universal Numbering System (1-32 permanent, A-T
primary) used in US practices, which is what the evaluation metric needs.
Conversion between Universal, FDI two-digit, and Palmer notation is part of the
domain layer proper and lands with the lexicon work.

Extraction is conservative by design. A bare "three" in running speech is far
more often a probing depth, a count of carpules, or an ordinary number than a
tooth reference, so only explicitly marked forms are matched. Over-extraction
would corrupt the tooth-accuracy metric in the direction that flatters the
system, which is the wrong direction for a clinical tool.
"""

from __future__ import annotations

import re

from ..normalize import words_to_digits

__all__ = ["extract_tooth_numbers", "MAX_UNIVERSAL", "PRIMARY_LETTERS"]

MAX_UNIVERSAL = 32
PRIMARY_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRST")

# "tooth 14", "tooth number 14", "tooth #14", "#14", "number 14"
_NUMERIC = re.compile(
    r"(?:tooth|teeth)\s*(?:number|no\.?|num|#)?\s*#?\s*(\d{1,2})"
    r"|(?:^|\s)#\s*(\d{1,2})"
    r"|(?:tooth|teeth)\s+(\d{1,2})",
    flags=re.IGNORECASE,
)

# Primary dentition: "tooth K", "tooth letter K"
_LETTER = re.compile(
    r"(?:tooth|teeth)\s*(?:letter)?\s*\b([A-T])\b",
    flags=re.IGNORECASE,
)


def extract_tooth_numbers(text: str) -> list[str]:
    """Return tooth references in order of appearance.

    Permanent teeth come back as strings of their Universal number; primary
    teeth as uppercase letters. Values outside 1-32 are discarded -- "tooth 45"
    is either a misrecognition or FDI notation, and guessing between those would
    silently invent a different tooth.
    """
    normalized = words_to_digits(text)
    found: list[str] = []

    for match in _NUMERIC.finditer(normalized):
        raw = next((g for g in match.groups() if g), None)
        if raw is None:
            continue
        value = int(raw)
        if 1 <= value <= MAX_UNIVERSAL:
            found.append(str(value))

    for match in _LETTER.finditer(normalized):
        letter = match.group(1).upper()
        if letter in PRIMARY_LETTERS:
            found.append(letter)

    return found
