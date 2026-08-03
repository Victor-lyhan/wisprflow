"""Data contracts, output sinks, and the cross-backend contract suite."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flowscribe.contracts import (
    AudioChunk,
    Transcript,
    TranscriptionResult,
    Utterance,
    Word,
    utterance_id,
)
from flowscribe.sinks import JsonlSink, JsonSink, SrtSink, TextSink, VttSink


class TestContracts:
    def test_audio_chunk_timing(self) -> None:
        chunk = AudioChunk(pcm=np.zeros(16_000, dtype=np.float32), start=2.0)
        assert chunk.duration == 1.0
        assert chunk.end == 3.0

    def test_utterance_ids_sort_lexicographically(self) -> None:
        """Zero-padded so ordering by id matches ordering by time."""
        ids = [utterance_id(i) for i in (0, 2, 10, 100)]
        assert ids == sorted(ids)

    def test_transcript_text_joins_utterances(self) -> None:
        t = Transcript(
            utterances=[
                Utterance(id="u00000", start=0, end=1, text="tooth three"),
                Utterance(id="u00001", start=1, end=2, text="mesial occlusal"),
            ]
        )
        assert t.text == "tooth three mesial occlusal"

    def test_transcript_skips_blank_utterances(self) -> None:
        t = Transcript(
            utterances=[
                Utterance(id="u00000", start=0, end=1, text="tooth three"),
                Utterance(id="u00001", start=1, end=2, text="   "),
            ]
        )
        assert t.text == "tooth three"

    def test_best_prefers_corrected(self) -> None:
        verbatim = Transcript(utterances=[Utterance(id="u0", start=0, end=1, text="buckle")])
        corrected = Transcript(utterances=[Utterance(id="u0", start=0, end=1, text="buccal")])
        assert TranscriptionResult(verbatim=verbatim).best is verbatim
        assert TranscriptionResult(verbatim=verbatim, corrected=corrected).best is corrected

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError):
            Word(text="x", start=0, end=1, confidence=1.5)

    def test_transcript_serializes_round_trip(self) -> None:
        t = Transcript(
            utterances=[
                Utterance(
                    id="u00000",
                    start=0,
                    end=1,
                    text="tooth three",
                    words=[Word(text="tooth", start=0, end=0.5, confidence=0.9)],
                )
            ],
            engine="fake",
        )
        assert Transcript.model_validate_json(t.model_dump_json()) == t


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        utterances=[
            Utterance(id="u00000", start=0.0, end=2.5, text="Tooth three has caries."),
            Utterance(
                id="u00001",
                start=2.5,
                end=4.0,
                text="Understood.",
            ),
            Utterance(
                id="u00002",
                start=4.0,
                end=6.0,
                text="Any pain?",
            ),
        ],
        engine="fake",
    )


class TestSinks:
    def test_json_round_trips(self, transcript: Transcript) -> None:
        assert Transcript.model_validate(json.loads(JsonSink().emit(transcript))) == transcript

    def test_jsonl_line_per_utterance(self, transcript: Transcript) -> None:
        lines = JsonlSink().emit(transcript).splitlines()
        assert len(lines) == 3
        assert all(json.loads(line)["id"] for line in lines)

    def test_text_separates_utterances_by_blank_line(self) -> None:
        t = Transcript(
            utterances=[
                Utterance(id="u0", start=0, end=1, text="First."),
                Utterance(id="u1", start=1, end=2, text="Second."),
            ]
        )
        assert TextSink().emit(t) == "First.\n\nSecond."

    def test_single_utterance(self) -> None:
        t = Transcript(utterances=[Utterance(id="u0", start=0, end=1, text="Hello.")])
        assert TextSink().emit(t) == "Hello."

    def test_srt_numbering_and_timestamps(self, transcript: Transcript) -> None:
        out = SrtSink().emit(transcript)
        assert out.startswith("1\n")
        assert "00:00:00,000 --> 00:00:02,500" in out

    def test_vtt_header_and_timestamps(self, transcript: Transcript) -> None:
        out = VttSink().emit(transcript)
        assert out.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.500" in out

    def test_timestamps_beyond_an_hour(self) -> None:
        t = Transcript(utterances=[Utterance(id="u0", start=3725.5, end=3726.0, text="x")])
        assert "01:02:05,500" in SrtSink().emit(t)

    def test_empty_transcript(self) -> None:
        empty = Transcript()
        for sink in (JsonSink(), JsonlSink(), TextSink(), SrtSink(), VttSink()):
            assert isinstance(sink.emit(empty), str)


# --------------------------------------------------------------------------- #
# Cross-backend contract suite
# --------------------------------------------------------------------------- #


class ASREngineContract:
    """Behaviour every ASR backend must exhibit.

    Subclass and supply ``engine`` to prove a new backend is a drop-in
    replacement. This is what makes "backends are interchangeable" a checked
    property rather than a design intention -- swapping Whisper for Parakeet
    should require no change above this line.
    """

    @pytest.fixture
    def engine(self):
        raise NotImplementedError

    @pytest.fixture
    def audio(self, sine: np.ndarray) -> AudioChunk:
        return AudioChunk(pcm=sine, start=0.0, is_last=True)

    def test_declares_a_name(self, engine) -> None:
        assert isinstance(engine.name, str) and engine.name

    def test_returns_utterances(self, engine, audio: AudioChunk) -> None:
        assert all(isinstance(u, Utterance) for u in engine.transcribe(audio))

    def test_empty_audio_returns_empty(self, engine) -> None:
        empty = AudioChunk(pcm=np.zeros(0, dtype=np.float32), is_last=True)
        assert engine.transcribe(empty) == []

    def test_timestamps_are_ordered_and_in_range(self, engine, audio: AudioChunk) -> None:
        for u in engine.transcribe(audio):
            assert u.start <= u.end
            assert u.start >= audio.start
            assert u.end <= audio.end + 1.0  # backends may pad slightly

    def test_ids_are_unique(self, engine, audio: AudioChunk) -> None:
        ids = [u.id for u in engine.transcribe(audio)]
        assert len(ids) == len(set(ids))

    def test_respects_chunk_offset(self, engine, sine: np.ndarray) -> None:
        """Timestamps must be absolute session time, or a transcript assembled
        from pieces has every span in the wrong place."""
        offset = AudioChunk(pcm=sine, start=100.0, is_last=True)
        for u in engine.transcribe(offset):
            assert u.start >= 100.0

    def test_accepts_hotwords(self, engine, audio: AudioChunk) -> None:
        engine.transcribe(audio, hotwords=["mesial", "buccal"])

    def test_close_is_idempotent(self, engine) -> None:
        engine.close()
        engine.close()


class TestFakeEngineContract(ASREngineContract):
    """Proves the contract suite is meaningful before it is aimed at real engines."""

    @pytest.fixture
    def engine(self):
        from tests.conftest import FakeASREngine

        return FakeASREngine()


@pytest.mark.slow
class TestFasterWhisperContract(ASREngineContract):
    """Real backend. Downloads a model, so it is excluded from the default run:

    pytest -m slow
    """

    @pytest.fixture
    def engine(self):
        pytest.importorskip("faster_whisper")
        from flowscribe.asr.faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine(model="tiny.en", device="cpu", compute_type="int8")
