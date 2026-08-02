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

__all__ = [
    "extract_tooth_numbers",
    "universal_to_fdi",
    "fdi_to_universal",
    "universal_to_palmer",
    "palmer_to_universal",
    "tooth_name",
    "MAX_UNIVERSAL",
    "PRIMARY_LETTERS",
]

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


# --------------------------------------------------------------------------- #
# Notation conversion
# --------------------------------------------------------------------------- #
#
# Three systems are in active use and they disagree in ways that silently
# produce the wrong tooth:
#
#   Universal (US)  1-32 permanent, A-T primary. A single running number
#                   starting at the upper-right third molar and sweeping to the
#                   lower-right third molar.
#   FDI (ISO 3950)  Two digits: quadrant (1-4 permanent, 5-8 primary) then
#                   position 1-8 outward from the midline. International default.
#   Palmer          Quadrant plus position 1-8, written with a bracket symbol.
#                   Rendered here as UR/UL/LL/LR + position, since the symbols
#                   have no plain-text form.
#
# The collision that matters: "tooth 18" is the upper-right third molar in FDI
# and the lower-left second molar in Universal -- opposite corners of the mouth.
# Conversions are therefore explicit and never inferred from context.

# Universal 1-16 run right-to-left across the maxilla, 17-32 left-to-right
# across the mandible. FDI quadrants number outward from the midline in both.
_QUADRANTS = ("UR", "UL", "LL", "LR")


def universal_to_fdi(tooth: int | str) -> str:
    """Convert a Universal number (1-32) or primary letter (A-T) to FDI."""
    if isinstance(tooth, str) and tooth.strip().upper() in PRIMARY_LETTERS:
        index = ord(tooth.strip().upper()) - ord("A")  # 0-19
        quadrant, position = divmod(index, 5)
        # Primary quadrants are 5-8; upper-right and lower-left count inward.
        if quadrant in (0, 2):
            position = 5 - position
        else:
            position = position + 1
        return f"{quadrant + 5}{position}"

    value = int(tooth)
    if not 1 <= value <= MAX_UNIVERSAL:
        raise ValueError(f"Universal tooth number out of range: {value}")

    quadrant, offset = divmod(value - 1, 8)
    # Upper right (1-8) and lower left (17-24) run toward the midline, so their
    # FDI position counts down; the other two count up.
    position = 8 - offset if quadrant in (0, 2) else offset + 1
    return f"{quadrant + 1}{position}"


def fdi_to_universal(code: int | str) -> str:
    """Convert an FDI two-digit code to Universal notation."""
    text = str(code).strip()
    if len(text) != 2 or not text.isdigit():
        raise ValueError(f"FDI code must be two digits: {code!r}")

    quadrant, position = int(text[0]), int(text[1])
    if not 1 <= position <= 8:
        raise ValueError(f"FDI position out of range: {code!r}")

    if 5 <= quadrant <= 8:  # primary dentition
        if position > 5:
            raise ValueError(f"Primary FDI position out of range: {code!r}")
        index = quadrant - 5
        offset = (5 - position) if index in (0, 2) else (position - 1)
        return chr(ord("A") + index * 5 + offset)

    if not 1 <= quadrant <= 4:
        raise ValueError(f"FDI quadrant out of range: {code!r}")

    offset = (8 - position) if quadrant in (1, 3) else (position - 1)
    return str((quadrant - 1) * 8 + offset + 1)


def universal_to_palmer(tooth: int | str) -> str:
    """Convert Universal notation to a plain-text Palmer form (e.g. ``UR6``)."""
    fdi = universal_to_fdi(tooth)
    quadrant = int(fdi[0])
    label = _QUADRANTS[(quadrant - 1) % 4]
    return f"{label}{fdi[1]}"


def palmer_to_universal(palmer: str) -> str:
    """Convert plain-text Palmer (``UR6``, ``LL3``) to Universal notation."""
    text = palmer.strip().upper()
    if len(text) < 3 or text[:2] not in _QUADRANTS:
        raise ValueError(f"Palmer notation must be like 'UR6': {palmer!r}")

    position = text[2:]
    if not position.isdigit() or not 1 <= int(position) <= 8:
        raise ValueError(f"Palmer position out of range: {palmer!r}")

    quadrant = _QUADRANTS.index(text[:2]) + 1
    return fdi_to_universal(f"{quadrant}{position}")


_POSITION_NAMES = {
    1: "central incisor",
    2: "lateral incisor",
    3: "canine",
    4: "first premolar",
    5: "second premolar",
    6: "first molar",
    7: "second molar",
    8: "third molar",
}

_ARCH_NAMES = {"UR": "upper right", "UL": "upper left", "LL": "lower left", "LR": "lower right"}


def tooth_name(tooth: int | str) -> str:
    """Human-readable name for a Universal tooth number.

    Used in correction prompts and review output: "tooth 1" carries no meaning to
    a reader checking a transcript, while "upper right third molar" is verifiable
    against what the clinician remembers doing.
    """
    palmer = universal_to_palmer(tooth)
    arch, position = palmer[:2], int(palmer[2:])
    return f"{_ARCH_NAMES[arch]} {_POSITION_NAMES[position]}"
