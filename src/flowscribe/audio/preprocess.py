"""Preprocessing and source construction."""

from __future__ import annotations

from pathlib import Path

from ..contracts import AudioChunk
from ..protocols import AudioSource
from .source import FileAudioSource, GrowingWavSource

__all__ = ["NullPreprocessor", "open_source"]


class NullPreprocessor:
    """Pass-through.

    Sources already deliver 16 kHz mono float32, so no preprocessing is required
    by default. Denoising is deliberately *not* on by default: published dental
    ASR work finds background noise degrades accuracy, but aggressive denoisers
    can also strip consonant detail and make recognition worse. Whether
    DeepFilterNet helps here is an empirical question for the noise sweep, not an
    assumption to bake into the default path.
    """

    def process(self, chunk: AudioChunk) -> AudioChunk:
        return chunk


def open_source(
    path: str | Path,
    *,
    live: bool = False,
    chunk_seconds: float = 1.0,
    **kwargs: object,
) -> AudioSource:
    """Open a path as the appropriate source.

    ``live=True`` selects the growing-file reader, for transcribing a recording
    that is still being written.
    """
    if live:
        return GrowingWavSource(path, chunk_seconds=chunk_seconds, **kwargs)  # type: ignore[arg-type]
    return FileAudioSource(path, chunk_seconds=chunk_seconds)
