"""Output formats.

Each sink renders a :class:`~flowscribe.contracts.Transcript` to a string. JSON
is the canonical form -- it is the only one that round-trips, since the subtitle
and plain-text formats discard word timings, confidences, and provenance.
"""

from __future__ import annotations

from ..contracts import Transcript, Utterance

__all__ = ["JsonSink", "JsonlSink", "TextSink", "SrtSink", "VttSink"]


def _timestamp(seconds: float, *, comma: bool = True) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (WebVTT)."""
    seconds = max(0.0, seconds)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:  # rounding carried into the next second
        millis, secs = 0, secs + 1
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _speaker_prefix(utterance: Utterance) -> str:
    """Prefer the clinical role over the raw diarization label when known."""
    label = utterance.role or utterance.speaker
    return f"{label}: " if label else ""


class JsonSink:
    """Full fidelity. The format to persist."""

    name = "json"
    extension = ".json"

    def emit(self, transcript: Transcript) -> str:
        return transcript.model_dump_json(indent=2)


class JsonlSink:
    """One utterance per line. Streams and greps well."""

    name = "jsonl"
    extension = ".jsonl"

    def emit(self, transcript: Transcript) -> str:
        return "\n".join(u.model_dump_json() for u in transcript.utterances)


class TextSink:
    """Readable plain text, speaker-prefixed when diarization ran.

    Consecutive utterances from one speaker are merged into a paragraph, since a
    label repeated on every sentence makes a clinical transcript hard to read.
    """

    name = "text"
    extension = ".txt"

    def emit(self, transcript: Transcript) -> str:
        lines: list[str] = []
        current: str | None = None
        buffer: list[str] = []

        for u in transcript.utterances:
            text = u.text.strip()
            if not text:
                continue
            label = u.role or u.speaker
            if label != current and buffer:
                lines.append(" ".join(buffer))
                buffer = []
            if label != current:
                current = label
                buffer.append(f"{label}: {text}" if label else text)
            else:
                buffer.append(text)

        if buffer:
            lines.append(" ".join(buffer))
        return "\n\n".join(lines)


class SrtSink:
    """SubRip subtitles."""

    name = "srt"
    extension = ".srt"

    def emit(self, transcript: Transcript) -> str:
        blocks = []
        for i, u in enumerate(transcript.utterances, start=1):
            text = u.text.strip()
            if not text:
                continue
            blocks.append(
                f"{i}\n{_timestamp(u.start)} --> {_timestamp(u.end)}\n{_speaker_prefix(u)}{text}"
            )
        return "\n\n".join(blocks)


class VttSink:
    """WebVTT subtitles."""

    name = "vtt"
    extension = ".vtt"

    def emit(self, transcript: Transcript) -> str:
        blocks = ["WEBVTT"]
        for u in transcript.utterances:
            text = u.text.strip()
            if not text:
                continue
            blocks.append(
                f"{_timestamp(u.start, comma=False)} --> {_timestamp(u.end, comma=False)}\n"
                f"{_speaker_prefix(u)}{text}"
            )
        return "\n\n".join(blocks)
