"""Audio sources.

Four implementations covering the ways audio reaches the pipeline. All satisfy
:class:`~dentascribe.protocols.AudioSource`, so the pipeline cannot tell a
finished file from a recording in progress -- which is the whole reason batch
needed no separate code path.

* :class:`ArrayAudioSource`  -- in-memory PCM; used by tests and by callers that
  already hold audio.
* :class:`FileAudioSource`   -- any container PyAV can decode.
* :class:`GrowingWavSource`  -- a WAV file still being written to.
* :class:`QueueAudioSource`  -- push-based; the entry point for a microphone or
  WebSocket feed.
"""

from __future__ import annotations

import queue
import struct
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np
import numpy.typing as npt

from ..contracts import TARGET_SAMPLE_RATE, AudioChunk, AudioMeta
from ..errors import AudioError
from .resample import PCMResampler, to_float32_mono

__all__ = [
    "ArrayAudioSource",
    "FileAudioSource",
    "GrowingWavSource",
    "QueueAudioSource",
]


def _blocks(
    pcm: npt.NDArray[np.float32], block: int, start_offset: float = 0.0
) -> Iterator[AudioChunk]:
    """Slice a buffer into fixed-size chunks with absolute timestamps."""
    total = len(pcm)
    for i in range(0, total, block):
        piece = pcm[i : i + block]
        yield AudioChunk(
            pcm=piece,
            sample_rate=TARGET_SAMPLE_RATE,
            start=start_offset + i / TARGET_SAMPLE_RATE,
            is_last=(i + block) >= total,
        )


class ArrayAudioSource:
    """Wraps PCM already in memory."""

    def __init__(
        self,
        pcm: npt.NDArray[Any],
        sample_rate: int = TARGET_SAMPLE_RATE,
        *,
        chunk_seconds: float = 1.0,
        name: str = "<array>",
    ) -> None:
        resampler = PCMResampler(sample_rate, src_channels=1)
        audio = np.concatenate([resampler.push(pcm), resampler.flush()])
        self._pcm = to_float32_mono(audio)
        self._block = max(1, int(chunk_seconds * TARGET_SAMPLE_RATE))
        self._name = name

    def meta(self) -> AudioMeta:
        return AudioMeta(
            source=self._name,
            duration=len(self._pcm) / TARGET_SAMPLE_RATE,
            sample_rate=TARGET_SAMPLE_RATE,
        )

    def chunks(self) -> Iterator[AudioChunk]:
        if len(self._pcm) == 0:
            yield AudioChunk(pcm=np.zeros(0, dtype=np.float32), is_last=True)
            return
        yield from _blocks(self._pcm, self._block)


class FileAudioSource:
    """Decodes a finished audio file of any format PyAV supports."""

    def __init__(self, path: str | Path, *, chunk_seconds: float = 1.0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise AudioError(f"Audio file not found: {self.path}")
        self._block = max(1, int(chunk_seconds * TARGET_SAMPLE_RATE))

    def meta(self) -> AudioMeta:
        try:
            with av.open(str(self.path)) as container:
                if not container.streams.audio:
                    raise AudioError(f"No audio stream in {self.path}")
                stream = container.streams.audio[0]
                duration = (
                    float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base
                    else (container.duration / av.time_base if container.duration else None)
                )
                return AudioMeta(
                    source=str(self.path),
                    duration=duration,
                    sample_rate=TARGET_SAMPLE_RATE,
                    channels=1,
                )
        except av.FFmpegError as exc:
            raise AudioError(f"Could not open {self.path}: {exc}") from exc

    def chunks(self) -> Iterator[AudioChunk]:
        # Decode into a rolling buffer and emit whole blocks, so chunk size stays
        # uniform regardless of how the container happens to frame its packets.
        buffer = np.zeros(0, dtype=np.float32)
        emitted = 0

        try:
            with av.open(str(self.path)) as container:
                if not container.streams.audio:
                    raise AudioError(f"No audio stream in {self.path}")
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)

                for frame in container.decode(stream):
                    for out in resampler.resample(frame):
                        buffer = np.concatenate(
                            [buffer, out.to_ndarray().reshape(-1).astype(np.float32)]
                        )
                    while len(buffer) >= self._block:
                        piece, buffer = buffer[: self._block], buffer[self._block :]
                        yield AudioChunk(
                            pcm=piece,
                            start=emitted / TARGET_SAMPLE_RATE,
                            is_last=False,
                        )
                        emitted += len(piece)

                for out in resampler.resample(None):
                    buffer = np.concatenate(
                        [buffer, out.to_ndarray().reshape(-1).astype(np.float32)]
                    )
        except av.FFmpegError as exc:
            raise AudioError(f"Decoding {self.path} failed: {exc}") from exc

        # Whatever is left is the tail; mark it so consumers can finalize.
        yield AudioChunk(pcm=buffer, start=emitted / TARGET_SAMPLE_RATE, is_last=True)


class GrowingWavSource:
    """Reads a WAV file that is still being recorded.

    Polls for newly written bytes and yields them as they appear. WAV
    specifically, because its payload is uncompressed and append-only -- a
    partially written compressed container generally cannot be decoded, so this
    is the format a recorder should target when live transcription is wanted.

    The writer signals completion either by creating ``<path>.done`` or by the
    caller invoking :meth:`mark_complete`. Without a signal this cannot
    distinguish "finished" from "paused", so it would block forever -- hence
    ``timeout``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_seconds: float = 1.0,
        poll_interval: float = 0.1,
        timeout: float = 300.0,
    ) -> None:
        self.path = Path(path)
        self._block_seconds = chunk_seconds
        self._poll = poll_interval
        self._timeout = timeout
        self._complete = threading.Event()
        self._header: tuple[int, int, int, int] | None = None  # rate, channels, bits, offset

    def mark_complete(self) -> None:
        """Tell the source no more audio is coming."""
        self._complete.set()

    def _done_flag(self) -> bool:
        return self._complete.is_set() or self.path.with_suffix(self.path.suffix + ".done").exists()

    def _read_header(self) -> tuple[int, int, int, int]:
        """Parse enough of the RIFF header to locate and interpret the PCM payload."""
        with self.path.open("rb") as fh:
            riff = fh.read(12)
            if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
                raise AudioError(f"Not a RIFF/WAVE file: {self.path}")

            rate = channels = bits = 0
            while True:
                head = fh.read(8)
                if len(head) < 8:
                    raise AudioError(f"No data chunk found yet in {self.path}")
                cid, size = struct.unpack("<4sI", head)
                if cid == b"fmt ":
                    fmt = fh.read(size)
                    _, channels, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
                elif cid == b"data":
                    if not rate:
                        raise AudioError(f"data chunk precedes fmt chunk in {self.path}")
                    if bits != 16:
                        raise AudioError(f"Only 16-bit PCM is supported, got {bits}-bit")
                    return rate, channels, bits, fh.tell()
                else:
                    fh.seek(size + (size % 2), 1)  # RIFF chunks are word-aligned

    def meta(self) -> AudioMeta:
        # Duration is None: the recording has no end yet, and reporting a length
        # that will change would mislead anything computing progress.
        return AudioMeta(source=str(self.path), duration=None, sample_rate=TARGET_SAMPLE_RATE)

    def chunks(self) -> Iterator[AudioChunk]:
        deadline = time.monotonic() + self._timeout
        while not self.path.exists():
            if time.monotonic() > deadline:
                raise AudioError(f"Timed out waiting for {self.path} to appear")
            time.sleep(self._poll)

        while self._header is None:
            try:
                self._header = self._read_header()
            except AudioError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(self._poll)

        rate, channels, bits, data_offset = self._header
        frame_bytes = channels * (bits // 8)
        resampler = PCMResampler(rate, src_channels=channels)
        block = max(1, int(self._block_seconds * TARGET_SAMPLE_RATE))

        pos = data_offset
        buffer = np.zeros(0, dtype=np.float32)
        emitted = 0

        with self.path.open("rb") as fh:
            while True:
                fh.seek(pos)
                raw = fh.read()
                # Never decode a partial frame; leave the remainder for next poll.
                usable = len(raw) - (len(raw) % frame_bytes)

                if usable:
                    pos += usable
                    ints = np.frombuffer(raw[:usable], dtype="<i2")
                    if channels > 1:
                        ints = ints.reshape(-1, channels).T
                    buffer = np.concatenate([buffer, resampler.push(ints)])
                    deadline = time.monotonic() + self._timeout

                while len(buffer) >= block:
                    piece, buffer = buffer[:block], buffer[block:]
                    yield AudioChunk(pcm=piece, start=emitted / TARGET_SAMPLE_RATE)
                    emitted += len(piece)

                if self._done_flag():
                    buffer = np.concatenate([buffer, resampler.flush()])
                    while len(buffer) > block:
                        piece, buffer = buffer[:block], buffer[block:]
                        yield AudioChunk(pcm=piece, start=emitted / TARGET_SAMPLE_RATE)
                        emitted += len(piece)
                    yield AudioChunk(pcm=buffer, start=emitted / TARGET_SAMPLE_RATE, is_last=True)
                    return

                if not usable:
                    if time.monotonic() > deadline:
                        raise AudioError(
                            f"No new audio in {self.path} for {self._timeout:.0f}s "
                            "and no completion signal; giving up."
                        )
                    time.sleep(self._poll)


class QueueAudioSource:
    """Push-based source for live capture.

    A producer thread -- microphone callback, WebSocket handler -- calls
    :meth:`push`, and the pipeline consumes :meth:`chunks` as audio arrives.
    Backpressure is bounded: if the consumer falls behind by ``maxsize`` chunks,
    :meth:`push` drops the oldest rather than growing without limit, since
    unbounded buffering during a long procedure would exhaust memory.
    """

    _SENTINEL = object()

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        *,
        channels: int = 1,
        maxsize: int = 256,
        name: str = "<stream>",
    ) -> None:
        self._q: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._resampler = PCMResampler(sample_rate, src_channels=channels)
        self._name = name
        self._elapsed = 0.0
        self.dropped = 0

    def push(self, pcm: npt.NDArray[Any]) -> None:
        """Submit captured audio. Safe to call from another thread."""
        resampled = self._resampler.push(pcm)
        if len(resampled) == 0:
            return
        try:
            self._q.put_nowait(resampled)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self._q.put_nowait(resampled)

    def close(self) -> None:
        """Signal end of stream."""
        tail = self._resampler.flush()
        if len(tail):
            self._q.put(tail)
        self._q.put(self._SENTINEL)

    def meta(self) -> AudioMeta:
        return AudioMeta(source=self._name, duration=None, sample_rate=TARGET_SAMPLE_RATE)

    def chunks(self) -> Iterator[AudioChunk]:
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                yield AudioChunk(
                    pcm=np.zeros(0, dtype=np.float32), start=self._elapsed, is_last=True
                )
                return
            chunk = AudioChunk(pcm=item, start=self._elapsed)
            self._elapsed = chunk.end
            yield chunk
