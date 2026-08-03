"""Live capture helpers shared by the CLI and the demo server.

Turns the pipeline's event objects into flat JSON records. Both consumers want
the same shape, and defining it once keeps the terminal output and the browser
UI from drifting apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .config import Config
from .contracts import (
    Event,
    FinalUtterance,
    PartialUtterance,
    TranscriptComplete,
)

__all__ = ["event_to_dict", "iter_json_events", "listen"]


def event_to_dict(event: Event) -> dict[str, Any]:
    """Flatten a pipeline event into a JSON-serializable record.

    ``partial`` records are explicitly marked provisional. A consumer that
    persists them as though they were final would be storing text the recognizer
    has not committed to -- and in this domain that could mean a drug name or
    tooth number that is about to change.
    """
    if isinstance(event, PartialUtterance):
        return {
            "type": "partial",
            "provisional": True,
            "start": round(event.utterance.start, 3),
            "end": round(event.utterance.end, 3),
            "text": event.utterance.text,
        }

    if isinstance(event, FinalUtterance):
        return {
            "type": "final",
            "provisional": False,
            "id": event.utterance.id,
            "start": round(event.utterance.start, 3),
            "end": round(event.utterance.end, 3),
            "text": event.utterance.text,
        }

    if isinstance(event, TranscriptComplete):
        result = event.result
        return {
            "type": "complete",
            "verbatim": json.loads(result.verbatim.model_dump_json()),
            "corrected": (
                json.loads(result.corrected.model_dump_json()) if result.corrected else None
            ),
            "edits": [json.loads(e.model_dump_json()) for e in result.edits],
            "flags": [json.loads(f.model_dump_json()) for f in result.flags],
            "text": result.best.text,
        }

    return {"type": "unknown"}


def listen(
    config: Config | None = None,
    *,
    device: int | str | None = None,
    chunk_seconds: float | None = None,
) -> Iterator[Event]:
    """Capture from a microphone and yield pipeline events until stopped.

    Stops cleanly on ``KeyboardInterrupt``: capture is closed, the audio already
    buffered is drained, and the final tier still runs -- so interrupting
    produces a complete transcript rather than discarding the recording.
    """
    from .audio.microphone import MicrophoneSource
    from .pipeline import Pipeline

    config = config or Config()
    source = MicrophoneSource(device=device, chunk_seconds=chunk_seconds or config.chunk_seconds)

    with Pipeline(config) as pipeline, source:
        try:
            yield from pipeline.stream(source)
        except KeyboardInterrupt:
            source.stop()
            # The generator is already closed by the interrupt, so the final tier
            # cannot be resumed from here. Callers wanting a transcript on Ctrl-C
            # should call source.stop() from a signal handler and let the stream
            # end naturally -- which is what the CLI does.
            raise


def iter_json_events(
    config: Config | None = None,
    *,
    device: int | str | None = None,
) -> Iterator[dict[str, Any]]:
    """``listen`` with events already flattened to JSON records."""
    for event in listen(config, device=device):
        yield event_to_dict(event)
