"""Data contracts shared by every stage of the pipeline.

Two families of type live here, split on purpose:

* Hot-path types (``AudioChunk``, ``VadSegment``) are plain slotted dataclasses.
  During streaming these are created many times per second and carry raw PCM;
  running pydantic validation over a numpy array on every chunk buys nothing.
* Serialized types (``Word``, ``Utterance``, ``Transcript``, events) are pydantic
  models, because they cross process boundaries -- written to disk, pushed over a
  WebSocket, handed to a consuming project -- and there the validation and schema
  generation earn their cost.

Nothing in this module imports an inference engine, so it stays importable with
only the core dependency set installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AudioChunk",
    "AudioMeta",
    "VadSegment",
    "Word",
    "Utterance",
    "Transcript",
    "Tier",
    "Edit",
    "Flag",
    "TranscriptionResult",
    "PartialUtterance",
    "FinalUtterance",
    "SpeakerRelabel",
    "TranscriptComplete",
    "Event",
    "utterance_id",
]

# The pipeline resamples everything to this rate; every ASR backend we target
# (Whisper, Parakeet, Silero VAD, pyannote) expects 16 kHz mono.
TARGET_SAMPLE_RATE = 16_000

Tier = Literal["live", "final"]
"""Which pass produced a transcript.

``live`` output is provisional: it is emitted while audio is still arriving, so
its speaker labels come from online clustering and may be revised. ``final``
output is authoritative -- produced after the recording ends, with full-context
ASR and offline diarization.
"""


# --------------------------------------------------------------------------- #
# Hot path
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AudioChunk:
    """A block of mono float32 PCM in ``[-1.0, 1.0]``.

    ``start`` is seconds from the beginning of the session, not from the
    beginning of the chunk, so downstream stages can emit absolute timestamps
    without threading an offset through every call.
    """

    pcm: np.ndarray
    sample_rate: int = TARGET_SAMPLE_RATE
    start: float = 0.0
    is_last: bool = False

    @property
    def duration(self) -> float:
        return len(self.pcm) / self.sample_rate

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(slots=True)
class VadSegment:
    """A span the VAD believes contains speech, in absolute session seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------- #
# Serialized
# --------------------------------------------------------------------------- #


class AudioMeta(BaseModel):
    """Provenance for the audio a transcript was derived from."""

    model_config = ConfigDict(frozen=True)

    source: str
    duration: float | None = None
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = 1


class Word(BaseModel):
    """A single token with timing. Backends that cannot produce word-level
    timestamps leave the word list empty rather than faking it."""

    text: str
    start: float
    end: float
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Utterance(BaseModel):
    """One contiguous span of speech attributed to one speaker.

    ``speaker`` is a diarization label (``SPEAKER_00``); ``role`` is the clinical
    interpretation of that label (``dentist``, ``assistant``, ``patient``). They
    are separate because diarization can be correct while role assignment is
    still unknown, and conflating them loses that distinction.
    """

    id: str
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)
    speaker: str | None = None
    role: str | None = None
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_final: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


class Transcript(BaseModel):
    """An ordered set of utterances plus the provenance needed to reproduce it."""

    utterances: list[Utterance] = Field(default_factory=list)
    tier: Tier = "final"
    audio: AudioMeta | None = None
    engine: str | None = None
    model: str | None = None
    language: str | None = None

    @property
    def text(self) -> str:
        return " ".join(u.text.strip() for u in self.utterances if u.text.strip())

    @property
    def duration(self) -> float:
        return max((u.end for u in self.utterances), default=0.0)

    def speaker_labels(self) -> list[str]:
        seen: dict[str, None] = {}
        for u in self.utterances:
            if u.speaker is not None:
                seen.setdefault(u.speaker, None)
        return list(seen)


class Edit(BaseModel):
    """A single change the corrector made, recorded so it can be audited.

    Every correction is logged rather than silently applied. A dental transcript
    that has been rewritten by an LLM without a diff is not verifiable, and
    published accuracy studies find clinically significant ASR errors in every
    system tested -- so the reviewer needs to see exactly what changed.
    """

    utterance_id: str
    before: str
    after: str
    kind: Literal["term", "tooth_number", "drug", "surface", "punctuation", "other"] = "other"
    reason: str | None = None


class Flag(BaseModel):
    """A span a human should verify. Nothing is changed.

    Raised for terms that are confusable with a clinically different term where
    *both* readings are valid vocabulary -- "reversible pulpitis" against
    "irreversible pulpitis", "mesial" against "distal". Similarity search cannot
    detect these, because the transcript looks entirely correct.

    Deliberately not auto-corrected. Choosing between them is a clinical
    judgement about what was actually said, and an LLM silently inverting a
    diagnosis is a worse failure than leaving the original error in place. The
    reviewer decides; the system only points.
    """

    utterance_id: str
    term: str
    alternatives: list[str]
    reason: str = "clinically confusable term"


class TranscriptionResult(BaseModel):
    """What the pipeline returns: both transcripts, the diff, and review flags.

    ``verbatim`` is raw ASR output. ``corrected`` is the LLM-corrected version,
    or ``None`` when correction is disabled. Both are retained so that correction
    can be measured rather than trusted.
    """

    verbatim: Transcript
    corrected: Transcript | None = None
    edits: list[Edit] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)

    @property
    def best(self) -> Transcript:
        """The transcript a caller should display if it only wants one."""
        return self.corrected if self.corrected is not None else self.verbatim


# --------------------------------------------------------------------------- #
# Streaming events
# --------------------------------------------------------------------------- #


class PartialUtterance(BaseModel):
    """An unstable hypothesis. Will be superseded; never persist this."""

    type: Literal["partial"] = "partial"
    utterance: Utterance


class FinalUtterance(BaseModel):
    """A hypothesis confirmed by the streaming policy. Text is stable from here;
    the speaker label may still be revised by the final pass."""

    type: Literal["final"] = "final"
    utterance: Utterance


class SpeakerRelabel(BaseModel):
    """Offline diarization disagreed with the online guess for an utterance.

    Emitted during finalization. Consumers that displayed a live label should
    apply this to stay consistent with the authoritative transcript.
    """

    type: Literal["relabel"] = "relabel"
    utterance_id: str
    speaker: str
    role: str | None = None


class TranscriptComplete(BaseModel):
    """Terminal event. Carries the authoritative result."""

    type: Literal["complete"] = "complete"
    result: TranscriptionResult


Event = Annotated[
    PartialUtterance | FinalUtterance | SpeakerRelabel | TranscriptComplete,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def utterance_id(index: int) -> str:
    """Deterministic utterance ids.

    Deliberately not a UUID: tests compare transcripts across runs, and a
    ``SpeakerRelabel`` has to reference an id that survives re-transcription of
    the same audio.
    """
    return f"u{index:05d}"
