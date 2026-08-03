"""Live microphone capture.

The missing piece between the streaming pipeline and a real operatory: everything
downstream could already consume audio as it arrived, but nothing produced it.

Design note on threading. PortAudio invokes the capture callback on a real-time
thread, and anything slow there causes dropouts -- clicks and lost words, not
merely lag. So the callback does the minimum possible: copy the buffer and hand
it to a queue. Resampling happens on the consuming thread, where a stall costs
latency instead of corrupting the recording.

Requires: ``pip install 'dentascribe[mic]'``
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from ..contracts import TARGET_SAMPLE_RATE, AudioChunk, AudioMeta
from ..errors import AudioError, MissingDependency
from .resample import PCMResampler

__all__ = ["MicrophoneSource", "list_input_devices", "default_input_device"]


def _require_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        # OSError too: sounddevice raises it when the PortAudio shared library
        # is missing, which looks nothing like an import failure to a user.
        raise MissingDependency(
            "sounddevice is not installed or PortAudio is unavailable. "
            "Install with: pip install 'dentascribe[mic]'"
        ) from exc
    return sd


def list_input_devices() -> list[dict[str, Any]]:
    """Capture-capable devices, with their index and channel count."""
    sd = _require_sounddevice()
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            devices.append(
                {
                    "index": index,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": int(device["default_samplerate"]),
                }
            )
    return devices


def default_input_device() -> dict[str, Any]:
    """The system default capture device."""
    sd = _require_sounddevice()
    try:
        device = sd.query_devices(kind="input")
    except Exception as exc:  # noqa: BLE001 - sounddevice raises bare exceptions
        raise AudioError(f"No default input device: {exc}") from exc
    return {
        "name": device["name"],
        "channels": device["max_input_channels"],
        "sample_rate": int(device["default_samplerate"]),
    }


class MicrophoneSource:
    """Captures from an input device and yields chunks as they arrive.

    Satisfies :class:`~dentascribe.protocols.AudioSource`, so the pipeline cannot
    tell live capture from a file -- the same ``stream()`` loop drives both.

    Use as a context manager, or call :meth:`start` and :meth:`stop` yourself::

        with MicrophoneSource() as mic:
            for event in pipeline.stream(mic):
                ...
    """

    def __init__(
        self,
        *,
        device: int | str | None = None,
        sample_rate: int | None = None,
        channels: int = 1,
        chunk_seconds: float = 1.0,
        blocksize: int = 0,
        max_queue: int = 512,
    ) -> None:
        self._sd = _require_sounddevice()

        if sample_rate is None or channels is None:
            info = self._sd.query_devices(device, kind="input")
            sample_rate = sample_rate or int(info["default_samplerate"])
            channels = min(channels, int(info["max_input_channels"])) or 1

        self.device = device
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self._block = max(1, int(chunk_seconds * TARGET_SAMPLE_RATE))
        self._blocksize = blocksize

        self._raw: queue.Queue[npt.NDArray[np.float32] | None] = queue.Queue(maxsize=max_queue)
        self._stream: Any = None
        self._stop = threading.Event()
        self._elapsed = 0.0

        self.dropped_blocks = 0
        self.overflows = 0

    # -- capture ------------------------------------------------------------ #

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio real-time callback. Must stay cheap."""
        if status:
            # input overflow means the consumer fell behind and audio was lost.
            # Counted rather than raised: a dropout should not end a recording.
            self.overflows += 1
        try:
            self._raw.put_nowait(indata.copy())
        except queue.Full:
            self.dropped_blocks += 1

    def start(self) -> MicrophoneSource:
        if self._stream is not None:
            return self
        try:
            self._stream = self._sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self._blocksize,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - sounddevice raises bare exceptions
            raise AudioError(
                f"Could not open input device {self.device!r}: {exc}. "
                "On macOS, grant microphone permission to the terminal application "
                "in System Settings > Privacy & Security > Microphone."
            ) from exc
        return self

    def stop(self) -> None:
        """Stop capture and signal end of stream."""
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        # Unblocks a consumer waiting on the queue.
        try:
            self._raw.put_nowait(None)
        except queue.Full:
            pass

    def __enter__(self) -> MicrophoneSource:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- AudioSource protocol ------------------------------------------------ #

    def meta(self) -> AudioMeta:
        name = "<microphone>"
        try:
            name = str(self._sd.query_devices(self.device, kind="input")["name"])
        except Exception:  # noqa: BLE001 - naming is cosmetic
            pass
        # duration is None: a live recording has no end until it is stopped, and
        # reporting a length that keeps changing would mislead any progress
        # display downstream.
        return AudioMeta(source=name, duration=None, sample_rate=TARGET_SAMPLE_RATE)

    def chunks(self) -> Iterator[AudioChunk]:
        """Yield 16 kHz mono blocks as they are captured."""
        if self._stream is None:
            self.start()

        resampler = PCMResampler(self.sample_rate, src_channels=self.channels)
        buffer = np.zeros(0, dtype=np.float32)

        while True:
            item = self._raw.get()
            if item is None:
                break

            # Interleaved (frames, channels) from PortAudio; the resampler wants
            # (channels, frames).
            block = item.T if item.ndim == 2 else item.reshape(1, -1)
            buffer = np.concatenate([buffer, resampler.push(block)])

            while len(buffer) >= self._block:
                piece, buffer = buffer[: self._block], buffer[self._block :]
                chunk = AudioChunk(pcm=piece, start=self._elapsed)
                self._elapsed = chunk.end
                yield chunk

        buffer = np.concatenate([buffer, resampler.flush()])
        while len(buffer) > self._block:
            piece, buffer = buffer[: self._block], buffer[self._block :]
            chunk = AudioChunk(pcm=piece, start=self._elapsed)
            self._elapsed = chunk.end
            yield chunk

        yield AudioChunk(pcm=buffer, start=self._elapsed, is_last=True)
