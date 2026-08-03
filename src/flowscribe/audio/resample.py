"""Resampling to the 16 kHz mono float32 form every downstream model expects.

PyAV is the single resampling authority here rather than an external ffmpeg
binary, because production is Windows and requiring operators to install and
PATH-configure ffmpeg is a real deployment failure mode. PyAV ships the codecs
in its wheel.
"""

from __future__ import annotations

import av
import numpy as np

from ..contracts import TARGET_SAMPLE_RATE
from ..errors import AudioError

__all__ = ["PCMResampler", "to_float32_mono"]


def to_float32_mono(pcm: np.ndarray) -> np.ndarray:
    """Coerce a PCM array to 1-D float32 in ``[-1.0, 1.0]``.

    Accepts int16/int32/float arrays, shaped ``(samples,)``, ``(channels, samples)``
    or ``(samples, channels)``. Multi-channel input is averaged rather than
    channel-selected: in an operatory the speakers are not reliably mic-separated,
    so dropping a channel can drop a speaker.
    """
    arr = np.asarray(pcm)

    if arr.ndim == 2:
        # Disambiguate orientation by assuming channels is the smaller axis.
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        arr = arr.mean(axis=0)
    elif arr.ndim != 1:
        raise AudioError(f"Expected 1-D or 2-D PCM, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_val
    else:
        arr = arr.astype(np.float32, copy=False)

    return np.ascontiguousarray(arr)


class PCMResampler:
    """Streaming resampler for raw PCM blocks.

    Stateful and order-dependent: it holds the filter tail between calls so that
    block boundaries do not produce clicks. Use one instance per audio stream and
    call :meth:`flush` at the end.
    """

    def __init__(self, src_rate: int, src_channels: int = 1) -> None:
        self.src_rate = src_rate
        self.src_channels = src_channels
        self._passthrough = src_rate == TARGET_SAMPLE_RATE and src_channels == 1
        self._resampler = (
            None
            if self._passthrough
            else av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)
        )

    def _emit(self, frames: list[av.AudioFrame]) -> np.ndarray:
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([f.to_ndarray().reshape(-1).astype(np.float32) for f in frames])

    def push(self, pcm: np.ndarray) -> np.ndarray:
        """Resample one block. May return fewer samples than given (or none)."""
        if self._passthrough:
            return to_float32_mono(pcm)

        assert self._resampler is not None
        arr = np.asarray(pcm)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.shape[0] > arr.shape[1]:
            arr = arr.T

        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
        arr = np.ascontiguousarray(arr.astype(np.float32))

        layout = "mono" if arr.shape[0] == 1 else "stereo" if arr.shape[0] == 2 else None
        if layout is None:
            arr = arr.mean(axis=0, keepdims=True)
            layout = "mono"

        try:
            frame = av.AudioFrame.from_ndarray(arr, format="fltp", layout=layout)
            frame.sample_rate = self.src_rate
            frame.pts = None
            return self._emit(self._resampler.resample(frame))
        except (av.FFmpegError, ValueError) as exc:
            raise AudioError(f"Resampling failed: {exc}") from exc

    def flush(self) -> np.ndarray:
        """Drain the filter tail. Call once, after the last :meth:`push`."""
        if self._passthrough or self._resampler is None:
            return np.zeros(0, dtype=np.float32)
        return self._emit(self._resampler.resample(None))
