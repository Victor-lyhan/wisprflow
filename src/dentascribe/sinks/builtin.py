"""Output formats.

Each sink renders a :class:`~dentascribe.contracts.Transcript` to a string. JSON
is the canonical form -- it is the only one that round-trips, since the subtitle
and plain-text formats discard word timings, confidences, and provenance.
"""

from __future__ import annotations

from ..contracts import Transcript

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
    """Readable plain text, one paragraph per utterance."""

    name = "text"
    extension = ".txt"

    def emit(self, transcript: Transcript) -> str:
        return "\n\n".join(u.text.strip() for u in transcript.utterances if u.text.strip())


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
            blocks.append(f"{i}\n{_timestamp(u.start)} --> {_timestamp(u.end)}\n{text}")
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
                f"{_timestamp(u.start, comma=False)} --> {_timestamp(u.end, comma=False)}\n{text}"
            )
        return "\n\n".join(blocks)
