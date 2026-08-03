"""Shared fixtures.

The fake engines here matter beyond convenience: they let the pipeline, offline
guarantee, and contract tests run in CI with no model downloads, so the suite
stays fast and works on a machine with networking disabled.
"""

from __future__ import annotations

import wave
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from flowscribe.contracts import AudioChunk, Edit, Transcript, Utterance, utterance_id


@pytest.fixture
def sine() -> np.ndarray:
    """Three seconds of 16 kHz mono tone."""
    t = np.linspace(0, 3, 16_000 * 3, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


@pytest.fixture
def wav_file(tmp_path: Path, sine: np.ndarray) -> Path:
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16_000)
        fh.writeframes((sine * 32767).astype("<i2").tobytes())
    return path


def write_wav(path: Path, pcm: np.ndarray, rate: int = 16_000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((pcm * 32767).astype("<i2").tobytes())
    return path


class FakeASREngine:
    """Deterministic engine returning canned text."""

    name = "fake"

    def __init__(self, text: str = "tooth number three mesial occlusal", **kwargs: object) -> None:
        self.text = text
        self.calls = 0
        self.last_hotwords: list[str] = []
        self.kwargs = kwargs

    def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        hotwords: Iterable[str] = (),
    ) -> list[Utterance]:
        self.calls += 1
        self.last_hotwords = list(hotwords)
        if len(audio.pcm) == 0:
            return []
        return [
            Utterance(
                id=utterance_id(0),
                start=audio.start,
                end=audio.end,
                text=self.text,
                language=language or "en",
                confidence=0.9,
            )
        ]

    def close(self) -> None:
        pass


class FakeDiarizer:
    """Alternates speakers so attribution is observable in tests."""

    name = "fake"
    online = False

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def assign(
        self,
        utterances: list[Utterance],
        *,
        audio: AudioChunk | None = None,
        num_speakers: int | None = None,
    ) -> list[Utterance]:
        return [
            u.model_copy(update={"speaker": f"SPEAKER_{i % 2:02d}"})
            for i, u in enumerate(utterances)
        ]

    def reset(self) -> None:
        pass


class FakeCorrector:
    """Applies a fixed substitution and records it as an edit."""

    name = "fake"

    def __init__(self, before: str = "buckle", after: str = "buccal", **kwargs: object) -> None:
        self.before = before
        self.after = after
        self.kwargs = kwargs

    def correct(self, transcript: Transcript) -> tuple[Transcript, list[Edit]]:
        edits: list[Edit] = []
        updated: list[Utterance] = []
        for u in transcript.utterances:
            if self.before in u.text:
                new_text = u.text.replace(self.before, self.after)
                edits.append(
                    Edit(
                        utterance_id=u.id,
                        before=u.text,
                        after=new_text,
                        kind="term",
                        reason="test fixture",
                    )
                )
                updated.append(u.model_copy(update={"text": new_text}))
            else:
                updated.append(u)
        return transcript.model_copy(update={"utterances": updated}), edits


@pytest.fixture
def fake_asr() -> FakeASREngine:
    return FakeASREngine()


@pytest.fixture
def fake_diarizer() -> FakeDiarizer:
    return FakeDiarizer()


@pytest.fixture
def fake_corrector() -> FakeCorrector:
    return FakeCorrector()


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        utterances=[
            Utterance(id="u00000", start=0.0, end=2.0, text="Tooth number three has caries."),
            Utterance(id="u00001", start=2.0, end=4.0, text="Probing depths on the buckle."),
        ],
        tier="final",
        engine="fake",
    )
