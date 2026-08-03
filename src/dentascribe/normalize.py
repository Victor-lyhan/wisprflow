"""Text normalization for scoring.

Normalization decides what counts as an error, so the choices here are part of
the metric definition, not incidental cleanup.

The one that matters most for dental audio is number handling. A reference
reading "probing depths are three two three" against a hypothesis of "R3, 2, 3"
differs in both spelling and digit form. Without number normalization the
comparison scores a spurious error on every charted depth, and perio charting is
a large fraction of what gets dictated -- so digits and number words are folded
to a single form before alignment.
"""

from __future__ import annotations

import re

__all__ = ["normalize_text", "words_to_digits", "tokenize"]

_UNITS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_HYPHEN_NUM = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s]"
    r"(one|two|three|four|five|six|seven|eight|nine)\b"
)
# Splits a token into (leading punctuation, core, trailing punctuation) so that
# "nineteen." still matches as a number word. Without this, any number word at a
# sentence boundary silently fails to normalize.
_AFFIX = re.compile(r"^(\W*)(.*?)(\W*)$", flags=re.UNICODE)
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _map_token(token: str) -> str:
    leading, core, trailing = _AFFIX.match(token).groups()  # type: ignore[union-attr]
    low = core.lower()
    if low in _UNITS:
        core = str(_UNITS[low])
    elif low in _TENS:
        core = str(_TENS[low])
    return f"{leading}{core}{trailing}"


def _apply_multipliers(text: str) -> str:
    """Combine a number with an explicit multiplier word.

    Restricted to runs containing "hundred" or "thousand". That restriction is
    what keeps a charted sequence safe: "three two three" has no multiplier, so
    it stays three separate depths rather than collapsing to 323. Meanwhile
    anaesthetic concentrations -- "one to one hundred thousand" against
    "1:100,000" -- do normalize to the same value, and those are clinically
    meaningful to get right.
    """
    text = re.sub(r"\b(\d+)\s+hundred\s+thousand\b", lambda m: str(int(m[1]) * 100_000), text)
    text = re.sub(r"\bhundred\s+thousand\b", "100000", text)
    text = re.sub(r"\b(\d+)\s+thousand\b", lambda m: str(int(m[1]) * 1000), text)
    text = re.sub(r"\bthousand\b", "1000", text)
    text = re.sub(r"\b(\d+)\s+hundred\b", lambda m: str(int(m[1]) * 100), text)
    text = re.sub(r"\bhundred\b", "100", text)
    return text


def words_to_digits(text: str) -> str:
    """Fold English number words into digits.

    Standalone words map up to 99. Larger values are only formed when an explicit
    multiplier word is present -- see :func:`_apply_multipliers` for why that
    restriction matters for perio charting.
    """
    text = _THOUSANDS_SEP.sub("", text)  # "100,000" -> "100000"
    text = _HYPHEN_NUM.sub(lambda m: str(_TENS[m.group(1)] + _UNITS[m.group(2)]), text)
    mapped = " ".join(_map_token(token) for token in text.split())
    return _apply_multipliers(mapped)


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_punctuation: bool = True,
    numbers_to_digits: bool = True,
) -> str:
    """Normalize for comparison. Same settings must be used on both sides."""
    if lowercase:
        text = text.lower()
    if numbers_to_digits:
        # Before punctuation stripping, so hyphenated forms are still visible.
        text = words_to_digits(text)
    if strip_punctuation:
        # Commas separating charted numbers become spaces, not deletions, so
        # "3, 2, 3" and "3 2 3" agree.
        text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str, **kwargs: bool) -> list[str]:
    """Normalize and split into words."""
    normalized = normalize_text(text, **kwargs)
    return normalized.split() if normalized else []
