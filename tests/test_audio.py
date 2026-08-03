"""Audio ingestion, resampling, and the growing-file reader."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from dentascribe.audio import (
    ArrayAudioSource,
    FileAudioSource,
    GrowingWavSource,
    PCMResampler,
    QueueAudioSource,
    to_float32_mono,
)
from dentascribe.contracts import TARGET_SAMPLE_RATE
from dentascribe.errors import AudioError


class TestToFloat32Mono:
    def test_int16_scales_to_unit_range(self) -> None:
        out = to_float32_mono(np.array([32767, -32768, 0], dtype=np.int16))
        assert out.dtype == np.float32
        assert out[0] == pytest.approx(1.0, abs=1e-4)
        assert out[2] == 0.0

    def test_downmixes_by_averaging(self) -> None:
        """Averaging rather than channel selection: operatory mics are not
        reliably speaker-separated, so dropping a channel can drop a speaker."""
        stereo = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)  # (channels, samples)
        assert to_float32_mono(stereo).tolist() == pytest.approx([0.5, 0.5])

    def test_handles_transposed_layout(self) -> None:
        assert len(to_float32_mono(np.zeros((100, 2), dtype=np.float32))) == 100

    def test_rejects_3d(self) -> None:
        with pytest.raises(AudioError):
            to_float32_mono(np.zeros((2, 2, 2)))


class TestPCMResampler:
    def test_upsamples_to_target_rate(self) -> None:
        pcm = np.zeros(8_000, dtype=np.float32)  # 1s at 8 kHz
        r = PCMResampler(8_000)
        out = np.concatenate([r.push(pcm), r.flush()])
        assert len(out) == pytest.approx(TARGET_SAMPLE_RATE, rel=0.02)

    def test_downsamples_to_target_rate(self) -> None:
        pcm = np.zeros(44_100, dtype=np.float32)  # 1s at 44.1 kHz
        r = PCMResampler(44_100)
        out = np.concatenate([r.push(pcm), r.flush()])
        assert len(out) == pytest.approx(TARGET_SAMPLE_RATE, rel=0.02)

    def test_passthrough_at_target_rate(self) -> None:
        pcm = np.zeros(16_000, dtype=np.float32)
        assert len(PCMResampler(16_000).push(pcm)) == 16_000

    def test_streaming_matches_single_shot_length(self) -> None:
        """Filter state must carry across blocks, or boundaries produce clicks
        and drift."""
        pcm = np.sin(np.linspace(0, 100, 44_100)).astype(np.float32)

        one = PCMResampler(44_100)
        whole = np.concatenate([one.push(pcm), one.flush()])

        many = PCMResampler(44_100)
        pieces = [many.push(pcm[i : i + 4410]) for i in range(0, len(pcm), 4410)]
        pieces.append(many.flush())
        streamed = np.concatenate(pieces)

        assert len(streamed) == pytest.approx(len(whole), abs=64)


class TestArrayAudioSource:
    def test_chunk_boundaries_and_timestamps(self, sine: np.ndarray) -> None:
        chunks = list(ArrayAudioSource(sine, chunk_seconds=1.0).chunks())
        assert len(chunks) == 3
        assert [c.start for c in chunks] == pytest.approx([0.0, 1.0, 2.0])
        assert chunks[-1].is_last

    def test_resamples_on_construction(self) -> None:
        src = ArrayAudioSource(np.zeros(8_000, dtype=np.float32), sample_rate=8_000)
        assert src.meta().sample_rate == TARGET_SAMPLE_RATE
        assert src.meta().duration == pytest.approx(1.0, rel=0.02)

    def test_empty_input_still_terminates(self) -> None:
        chunks = list(ArrayAudioSource(np.zeros(0, dtype=np.float32)).chunks())
        assert len(chunks) == 1
        assert chunks[0].is_last


class TestFileAudioSource:
    def test_reads_wav(self, wav_file: Path) -> None:
        src = FileAudioSource(wav_file)
        total = sum(len(c.pcm) for c in src.chunks())
        assert total == pytest.approx(16_000 * 3, abs=256)

    def test_reports_duration(self, wav_file: Path) -> None:
        assert FileAudioSource(wav_file).meta().duration == pytest.approx(3.0, rel=0.05)

    def test_final_chunk_is_flagged(self, wav_file: Path) -> None:
        chunks = list(FileAudioSource(wav_file).chunks())
        assert chunks[-1].is_last
        assert sum(c.is_last for c in chunks) == 1

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AudioError, match="not found"):
            FileAudioSource(tmp_path / "nope.wav")

    def test_corrupt_file_raises_audio_error_from_meta(self, tmp_path: Path) -> None:
        """Regression: the handler named ``av.AVError``, which does not exist in
        PyAV 12+. Evaluating it while unwinding raised ``AttributeError`` instead
        of a clean ``AudioError``, so every decode failure surfaced as an
        unrelated crash. The two pre-existing error tests missed it because
        neither reaches PyAV -- one checks existence first, the other parses WAV
        headers directly.
        """
        path = tmp_path / "corrupt.wav"
        path.write_bytes(b"RIFF____WAVEgarbage" * 50)
        with pytest.raises(AudioError):
            FileAudioSource(path).meta()

    def test_corrupt_file_raises_audio_error_from_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.wav"
        path.write_bytes(b"RIFF____WAVEgarbage" * 50)
        with pytest.raises(AudioError):
            list(FileAudioSource(path).chunks())

    def test_empty_file_raises_audio_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")
        with pytest.raises(AudioError):
            FileAudioSource(path).meta()


class TestGrowingWavSource:
    """The reader that makes 'transcribe while still recording' possible."""

    def _writer(self, path: Path, pcm: np.ndarray, rate: int, channels: int, delay: float):
        def run() -> None:
            fh = wave.open(str(path), "wb")
            fh.setnchannels(channels)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            step = (rate // 10) * channels
            data = (pcm * 32767).astype("<i2")
            for i in range(0, len(data), step):
                fh.writeframes(data[i : i + step].tobytes())
                time.sleep(delay)
            fh.close()
            path.with_suffix(path.suffix + ".done").touch()

        return threading.Thread(target=run, daemon=True)

    def test_reads_while_being_written(self, tmp_path: Path) -> None:
        path = tmp_path / "live.wav"
        pcm = np.zeros(16_000 * 2, dtype=np.float32)
        self._writer(path, pcm, 16_000, 1, 0.01).start()

        total = sum(
            len(c.pcm) for c in GrowingWavSource(path, chunk_seconds=0.5, timeout=15).chunks()
        )
        assert total == pytest.approx(16_000 * 2, abs=512)

    def test_yields_audio_before_writing_completes(self, tmp_path: Path) -> None:
        """The core latency property: audio must be available mid-recording, not
        only once the file is closed."""
        path = tmp_path / "live.wav"
        pcm = np.zeros(16_000 * 3, dtype=np.float32)
        started = time.monotonic()
        self._writer(path, pcm, 16_000, 1, 0.05).start()

        first_at = None
        for chunk in GrowingWavSource(path, chunk_seconds=0.25, timeout=20).chunks():
            if first_at is None and len(chunk.pcm):
                first_at = time.monotonic() - started
            if chunk.is_last:
                break
        finished_at = time.monotonic() - started

        assert first_at is not None
        assert first_at < finished_at / 2

    def test_resamples_stereo_input(self, tmp_path: Path) -> None:
        path = tmp_path / "live.wav"
        mono = np.zeros(44_100, dtype=np.float32)
        stereo = np.repeat(mono[:, None], 2, axis=1).reshape(-1)
        self._writer(path, stereo, 44_100, 2, 0.01).start()

        total = sum(
            len(c.pcm) for c in GrowingWavSource(path, chunk_seconds=0.5, timeout=15).chunks()
        )
        assert total == pytest.approx(TARGET_SAMPLE_RATE, rel=0.05)

    def test_mark_complete_ends_stream(self, tmp_path: Path) -> None:
        path = tmp_path / "live.wav"
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16_000)
            fh.writeframes(np.zeros(16_000, dtype="<i2").tobytes())

        src = GrowingWavSource(path, chunk_seconds=0.5, timeout=15)
        src.mark_complete()
        assert list(src.chunks())[-1].is_last

    def test_times_out_without_completion_signal(self, tmp_path: Path) -> None:
        """Without a signal, 'paused' and 'finished' are indistinguishable, so
        this must give up rather than block forever."""
        path = tmp_path / "stalled.wav"
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16_000)
            fh.writeframes(np.zeros(1600, dtype="<i2").tobytes())

        with pytest.raises(AudioError, match="No new audio"):
            list(GrowingWavSource(path, chunk_seconds=0.5, timeout=0.5).chunks())

    def test_rejects_non_wav(self, tmp_path: Path) -> None:
        path = tmp_path / "bogus.wav"
        path.write_bytes(b"NOTRIFFDATA" * 10)
        with pytest.raises(AudioError):
            list(GrowingWavSource(path, timeout=0.5).chunks())


class TestQueueAudioSource:
    def test_push_then_close(self) -> None:
        src = QueueAudioSource()
        src.push(np.zeros(16_000, dtype=np.float32))
        src.close()
        chunks = list(src.chunks())
        assert sum(len(c.pcm) for c in chunks) == 16_000
        assert chunks[-1].is_last

    def test_timestamps_advance(self) -> None:
        src = QueueAudioSource()
        for _ in range(3):
            src.push(np.zeros(16_000, dtype=np.float32))
        src.close()
        starts = [c.start for c in src.chunks() if len(c.pcm)]
        assert starts == pytest.approx([0.0, 1.0, 2.0])

    def test_drops_oldest_when_consumer_falls_behind(self) -> None:
        """Bounded rather than unbounded buffering: a long procedure with a slow
        consumer must not exhaust memory."""
        src = QueueAudioSource(maxsize=2)
        for _ in range(5):
            src.push(np.zeros(1600, dtype=np.float32))
        assert src.dropped > 0
