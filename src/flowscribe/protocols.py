"""Stage interfaces.

Each pipeline stage is a ``Protocol`` rather than a base class, so a consuming
project can satisfy it with any object -- including one that already exists in
that project and knows nothing about flowscribe.

Batch is modeled as a degenerate case of streaming: an ``AudioSource`` for a
finished file simply yields its chunks and sets ``is_last``. There is no separate
batch code path to keep in sync, and the contracts cannot quietly acquire
batch-only assumptions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from .contracts import (
    AudioChunk,
    AudioMeta,
    Edit,
    Transcript,
    Utterance,
    VadSegment,
)

__all__ = [
    "AudioSource",
    "Preprocessor",
    "VAD",
    "ASREngine",
    "StreamingPolicy",
    "Diarizer",
    "Corrector",
    "Sink",
]


@runtime_checkable
class AudioSource(Protocol):
    """Produces audio, whether from a finished file or an in-progress recording.

    A source for a growing file blocks between chunks until more data lands and
    only sets ``is_last`` once the writer signals completion. That is the whole
    of what "start transcribing while still recording" requires at this layer.
    """

    def meta(self) -> AudioMeta:
        """Describe the audio. ``duration`` is ``None`` when still recording."""
        ...

    def chunks(self) -> Iterator[AudioChunk]: ...


@runtime_checkable
class Preprocessor(Protocol):
    """Normalizes a chunk before recognition (resample, downmix, denoise)."""

    def process(self, chunk: AudioChunk) -> AudioChunk: ...


@runtime_checkable
class VAD(Protocol):
    """Finds speech spans, so the ASR is not asked to decode silence or drill noise."""

    def segments(self, chunk: AudioChunk) -> list[VadSegment]: ...

    def has_speech(self, chunk: AudioChunk) -> bool:
        """Whether the chunk contains any speech at all.

        Separate from :meth:`segments` because it is the question the streaming
        loop actually asks -- it only needs to decide whether decoding this block
        is worth doing, not where within it the speech lies.
        """
        ...

    def reset(self) -> None:
        """Drop internal state between sessions. VADs are usually stateful."""
        ...


@runtime_checkable
class ASREngine(Protocol):
    """Turns audio into text.

    ``hotwords`` carries a small domain vocabulary for backends that support
    contextual biasing. It is deliberately small: Whisper's prompt channel only
    consumes its final 224 tokens and grows hallucination-prone when stuffed, so
    the full dental lexicon belongs in the correction stage, not here.
    """

    name: str

    def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        hotwords: Iterable[str] = (),
    ) -> list[Utterance]: ...

    def close(self) -> None: ...


@runtime_checkable
class StreamingPolicy(Protocol):
    """Decides when a hypothesis is stable enough to emit as final.

    The reference implementation is LocalAgreement-2: text is confirmed once two
    successive decodes over a growing buffer agree on it, which is what keeps
    live output from flickering as context arrives.
    """

    def update(self, hypothesis: list[Utterance]) -> tuple[list[Utterance], list[Utterance]]:
        """Return ``(confirmed, pending)`` given the newest hypothesis."""
        ...

    def flush(self) -> list[Utterance]:
        """Confirm everything still pending, at end of stream."""
        ...

    def reset(self) -> None: ...


@runtime_checkable
class Diarizer(Protocol):
    """Attributes utterances to speakers.

    Offline implementations need the whole recording and are used for the final
    tier; online implementations label incrementally and are used live. Both
    satisfy this interface, which is what lets the pipeline swap one for the
    other between tiers without the surrounding code changing.
    """

    name: str
    online: bool

    def assign(
        self,
        utterances: list[Utterance],
        *,
        audio: AudioChunk | None = None,
        num_speakers: int | None = None,
    ) -> list[Utterance]: ...

    def reset(self) -> None: ...


@runtime_checkable
class Corrector(Protocol):
    """Repairs domain terminology in a transcript.

    Returns the corrected transcript *and* the edits it made. Implementations
    must not add clinical content; the edit list is what makes that checkable
    rather than merely promised.
    """

    name: str

    def correct(self, transcript: Transcript) -> tuple[Transcript, list[Edit]]: ...


@runtime_checkable
class Sink(Protocol):
    """Renders a finished result to some destination."""

    def emit(self, transcript: Transcript) -> str: ...
