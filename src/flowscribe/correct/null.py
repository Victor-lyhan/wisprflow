"""Pass-through corrector."""

from __future__ import annotations

from ..contracts import Edit, Transcript

__all__ = ["NullCorrector"]


class NullCorrector:
    """Makes no changes.

    Returns the transcript unchanged with an empty edit list, which is the
    correct baseline for measuring whether a real corrector helps: any accuracy
    difference against this backend is attributable to the corrector alone.
    """

    name = "null"

    def __init__(self, **options: object) -> None:
        self._options = options

    def correct(self, transcript: Transcript) -> tuple[Transcript, list[Edit]]:
        return transcript, []
