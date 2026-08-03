"""Silero VAD via ONNX Runtime.

Finds speech so the recognizer is not asked to decode silence. In a dental
operatory that is not a marginal saving: much of an appointment is the operator
working without speaking, and a long procedure spends most of its wall-clock
time on audio containing no words at all.

Silero is MIT-licensed, roughly 2 MB, and runs through the same ONNX Runtime the
Parakeet backend uses -- so enabling it adds no new native dependency on Windows.

Note on noise: this detects *speech*, not *useful* speech. A high-speed handpiece
is broadband noise and can read as voiced. Whether VAD helps or hurts under drill
noise is an empirical question for the noise sweep, which is why VAD gates
decoding but never discards audio.

Requires: ``pip install 'flowscribe[vad]'``
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..contracts import AudioChunk, VadSegment
from ..errors import MissingDependency

__all__ = ["SileroVAD"]


class SileroVAD:
    """Speech detection over a chunk of audio."""

    name = "silero"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 400,
        speech_pad_ms: int = 30,
        max_speech_s: float = 20.0,
        device: str = "auto",
        model_dir: str | None = None,
        **options: object,
    ) -> None:
        try:
            import onnx_asr
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise MissingDependency(
                "onnx-asr is not installed. Install with: pip install 'flowscribe[vad]'"
            ) from exc

        from ..asr.parakeet_onnx import _resolve_providers

        self.threshold = threshold
        self._options = options
        # Padding each side of a detection: clipping a word's onset costs more
        # accuracy than decoding a little extra silence.
        self._kwargs: dict[str, float] = {
            "threshold": threshold,
            "min_speech_duration_ms": float(min_speech_ms),
            "min_silence_duration_ms": float(min_silence_ms),
            "speech_pad_ms": float(speech_pad_ms),
            "max_speech_duration_s": float(max_speech_s),
        }

        self.providers = _resolve_providers(device)
        # See ParakeetOnnxEngine: `path` is a pre-staged model directory, not a
        # download cache.
        if model_dir:
            os.environ.setdefault("HF_HUB_CACHE", str(model_dir))
        self._vad: Any = onnx_asr.load_vad("silero", providers=self.providers)

    def segments(self, chunk: AudioChunk) -> list[VadSegment]:
        """Speech spans within ``chunk``, in absolute session seconds."""
        if len(chunk.pcm) == 0:
            return []

        # Silero supports 8 kHz and 16 kHz only; the pipeline always delivers
        # 16 kHz, so an unexpected rate is a bug worth surfacing rather than
        # silently resampling here.
        if chunk.sample_rate not in (8_000, 16_000):
            raise ValueError(
                f"Silero VAD supports 8 kHz or 16 kHz, got {chunk.sample_rate}. "
                "Audio should have been resampled before reaching the VAD."
            )

        waveform = np.ascontiguousarray(chunk.pcm, dtype=np.float32)[None, :]
        lengths = np.array([waveform.shape[1]], dtype=np.int64)

        batches = self._vad.segment_batch(waveform, lengths, chunk.sample_rate, **self._kwargs)
        spans = list(next(iter(batches), iter(())))

        return [
            VadSegment(
                start=chunk.start + start / chunk.sample_rate,
                end=chunk.start + end / chunk.sample_rate,
            )
            for start, end in spans
        ]

    def has_speech(self, chunk: AudioChunk) -> bool:
        """Whether the chunk contains any speech.

        The cheap question the streaming loop actually asks -- it only needs to
        know whether decoding this block is worth doing.
        """
        return bool(self.segments(chunk))

    def reset(self) -> None:
        # Stateless per call: each chunk is segmented independently. That loses a
        # little accuracy at block boundaries, but the streaming loop decodes a
        # rolling buffer rather than isolated blocks, so a word split across a
        # boundary is still decoded with its surrounding context.
        pass
