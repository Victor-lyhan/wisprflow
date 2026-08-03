"""Pass-through diarizer."""

from __future__ import annotations

from ..contracts import AudioChunk, Utterance

__all__ = ["NullDiarizer"]


class NullDiarizer:
    """Assigns no speakers.

    Leaves ``speaker`` as ``None`` rather than labelling everything
    ``SPEAKER_00``. A single fabricated label is indistinguishable downstream
    from a genuine single-speaker result, and in a multi-party operatory that
    would silently misattribute the patient's words to the dentist.
    """

    name = "passthrough"
    online = False

    def __init__(self, **options: object) -> None:
        self._options = options

    def assign(
        self,
        utterances: list[Utterance],
        *,
        audio: AudioChunk | None = None,
        num_speakers: int | None = None,
    ) -> list[Utterance]:
        return utterances

    def reset(self) -> None:
        pass
