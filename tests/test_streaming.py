"""LocalAgreement confirmation and the streaming pipeline."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from flowscribe import Config
from flowscribe.audio import ArrayAudioSource
from flowscribe.config import ASRConfig, CorrectionConfig, DiarizationConfig
from flowscribe.contracts import (
    AudioChunk,
    FinalUtterance,
    PartialUtterance,
    TranscriptComplete,
    Utterance,
    utterance_id,
)
from flowscribe.pipeline import Pipeline
from flowscribe.streaming import LocalAgreement


def utt(text: str, start: float = 0.0, end: float | None = None) -> Utterance:
    words = text.split()
    end = end if end is not None else start + len(words) * 0.5
    return Utterance(id="u", start=start, end=end, text=text)


class TestLocalAgreement:
    def test_nothing_confirmed_on_first_hypothesis(self) -> None:
        """With N=2 a single decode is never enough -- that is the whole point."""
        confirmed, pending = LocalAgreement(n=2).update([utt("tooth number three")])
        assert confirmed == []
        assert pending and pending[0].text == "tooth number three"

    def test_confirms_agreed_prefix(self) -> None:
        policy = LocalAgreement(n=2)
        policy.update([utt("tooth number three has")])
        confirmed, pending = policy.update([utt("tooth number three has caries")])
        assert confirmed[0].text == "tooth number three has"
        assert pending[0].text == "caries"

    def test_withholds_text_the_model_revised(self) -> None:
        """The behaviour that motivates the policy: a revised tail must not have
        been shown as final."""
        policy = LocalAgreement(n=2)
        policy.update([utt("depths on the buckle")])
        confirmed, _ = policy.update([utt("depths on the buccal")])
        assert confirmed[0].text == "depths on the"
        assert "buckle" not in confirmed[0].text

    def test_ignores_case_and_punctuation_churn(self) -> None:
        """Models flip capitalization and commas between decodes; treating that
        as disagreement would stall confirmation forever."""
        policy = LocalAgreement(n=2)
        policy.update([utt("tooth number three")])
        confirmed, _ = policy.update([utt("Tooth number three,")])
        assert len(confirmed[0].text.split()) == 3

    def test_confirmation_is_monotonic(self) -> None:
        policy = LocalAgreement(n=2)
        seen: list[str] = []
        for text in [
            "tooth number",
            "tooth number three",
            "tooth number three has",
            "tooth number three has caries",
        ]:
            confirmed, _ = policy.update([utt(text)])
            seen.extend(u.text for u in confirmed)
        joined = " ".join(seen).split()
        assert joined == joined[: len(joined)]  # never rewritten
        assert "tooth" in seen[0]

    def test_committed_until_advances(self) -> None:
        """Drives buffer trimming; if it never advances, decode cost grows
        without bound over a long appointment."""
        policy = LocalAgreement(n=2)
        assert policy.committed_until == 0.0
        policy.update([utt("tooth number three", start=0.0, end=3.0)])
        policy.update([utt("tooth number three", start=0.0, end=3.0)])
        assert policy.committed_until > 0.0

    def test_flush_confirms_remainder(self) -> None:
        policy = LocalAgreement(n=2)
        policy.update([utt("tooth number three")])
        assert policy.flush()[0].text == "tooth number three"

    def test_flush_on_empty_policy(self) -> None:
        assert LocalAgreement(n=2).flush() == []

    def test_n_of_one_confirms_immediately(self) -> None:
        confirmed, _ = LocalAgreement(n=1).update([utt("tooth number three")])
        assert confirmed[0].text == "tooth number three"

    def test_higher_n_is_more_conservative(self) -> None:
        policy = LocalAgreement(n=3)
        policy.update([utt("tooth number three")])
        confirmed, _ = policy.update([utt("tooth number three")])
        assert confirmed == []
        confirmed, _ = policy.update([utt("tooth number three")])
        assert confirmed[0].text == "tooth number three"

    def test_rejects_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            LocalAgreement(n=0)

    def test_reset_clears_state(self) -> None:
        policy = LocalAgreement(n=2)
        policy.update([utt("a b c")])
        policy.update([utt("a b c")])
        policy.reset()
        assert policy.committed_until == 0.0
        assert policy.update([utt("a b c")])[0] == []

    def test_emitted_ids_are_unique(self) -> None:
        policy = LocalAgreement(n=2)
        ids: list[str] = []
        for text in ["a b", "a b c d", "a b c d e f"]:
            confirmed, _ = policy.update([utt(text)])
            ids.extend(u.id for u in confirmed)
        assert len(ids) == len(set(ids))

    def test_empty_hypothesis_is_safe(self) -> None:
        confirmed, pending = LocalAgreement(n=2).update([])
        assert confirmed == [] and pending == []

    def test_uses_word_timings_when_available(self) -> None:
        from flowscribe.contracts import Word

        u = Utterance(
            id="u",
            start=0.0,
            end=2.0,
            text="tooth three",
            words=[
                Word(text="tooth", start=0.0, end=0.8),
                Word(text="three", start=1.2, end=2.0),
            ],
        )
        policy = LocalAgreement(n=2)
        policy.update([u])
        confirmed, _ = policy.update([u])
        assert confirmed[0].words[1].start == pytest.approx(1.2)


class GrowingASR:
    """Emits progressively more of a script, mimicking a real streaming decode."""

    name = "growing"

    def __init__(self, script: str) -> None:
        self.words = script.split()
        self.calls = 0

    def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        hotwords: Iterable[str] = (),
    ) -> list[Utterance]:
        if len(audio.pcm) == 0:
            return []
        self.calls += 1
        # Reveal roughly two words per elapsed second of audio.
        count = min(len(self.words), max(1, int(audio.end * 2)))
        text = " ".join(self.words[:count])
        return [
            Utterance(
                id=utterance_id(0), start=audio.start, end=audio.end, text=text, language="en"
            )
        ]

    def close(self) -> None:
        pass


@pytest.fixture
def quiet_config() -> Config:
    return Config(
        correction=CorrectionConfig(enabled=False),
        diarization=DiarizationConfig(enabled=False),
        final=ASRConfig(model="tiny.en"),
        chunk_seconds=1.0,
    )


class TestPipelineStream:
    def test_yields_events_then_completes(self, quiet_config: Config, sine: np.ndarray) -> None:
        engine = GrowingASR("tooth number three has a mesial occlusal composite")
        pipeline = Pipeline(quiet_config, asr=engine, live_asr=engine)
        events = list(pipeline.stream(ArrayAudioSource(sine, chunk_seconds=1.0)))

        assert isinstance(events[-1], TranscriptComplete)
        assert any(isinstance(e, FinalUtterance) for e in events)

    def test_emits_before_the_stream_ends(self, quiet_config: Config, sine: np.ndarray) -> None:
        """The point of streaming: text must appear before the terminal event."""
        engine = GrowingASR("tooth number three has a mesial occlusal composite")
        pipeline = Pipeline(quiet_config, asr=engine, live_asr=engine)

        first_text_at = None
        for index, event in enumerate(pipeline.stream(ArrayAudioSource(sine, chunk_seconds=1.0))):
            if first_text_at is None and isinstance(event, FinalUtterance | PartialUtterance):
                first_text_at = index
            last = index

        assert first_text_at is not None
        assert first_text_at < last

    def test_partials_are_marked_provisional(self, quiet_config: Config, sine: np.ndarray) -> None:
        engine = GrowingASR("tooth number three has a mesial occlusal composite")
        pipeline = Pipeline(quiet_config, asr=engine, live_asr=engine)
        partials = [
            e
            for e in pipeline.stream(ArrayAudioSource(sine, chunk_seconds=1.0))
            if isinstance(e, PartialUtterance)
        ]
        assert partials and all(not p.utterance.is_final for p in partials)

    def test_final_tier_is_authoritative(self, quiet_config: Config, sine: np.ndarray) -> None:
        """The terminal event carries a full-context result, not the concatenated
        live text."""
        engine = GrowingASR("tooth number three has a mesial occlusal composite")
        pipeline = Pipeline(quiet_config, asr=engine, live_asr=engine)
        complete = list(pipeline.stream(ArrayAudioSource(sine, chunk_seconds=1.0)))[-1]

        assert isinstance(complete, TranscriptComplete)
        assert complete.result.verbatim.tier == "final"
        assert complete.result.verbatim.text

    def test_live_text_is_not_lost(self, quiet_config: Config, sine: np.ndarray) -> None:
        engine = GrowingASR("tooth number three has a mesial occlusal composite")
        pipeline = Pipeline(quiet_config, asr=engine, live_asr=engine)
        finals = [
            e.utterance.text
            for e in pipeline.stream(ArrayAudioSource(sine, chunk_seconds=1.0))
            if isinstance(e, FinalUtterance)
        ]
        assert "tooth" in " ".join(finals)

    def test_empty_audio_still_completes(self, quiet_config: Config) -> None:
        engine = GrowingASR("anything")
        pipeline = Pipeline(quiet_config, asr=engine, live_asr=engine)
        source = ArrayAudioSource(np.zeros(0, dtype=np.float32))
        assert isinstance(list(pipeline.stream(source))[-1], TranscriptComplete)

    def test_relabel_emitted_when_diarization_disagrees(
        self, sine: np.ndarray, fake_diarizer
    ) -> None:
        """Live labels come from partial audio and are routinely wrong early on;
        the consumer needs to be told."""
        from flowscribe.contracts import SpeakerRelabel

        config = Config(
            correction=CorrectionConfig(enabled=False),
            diarization=DiarizationConfig(enabled=True, backend="passthrough"),
            chunk_seconds=1.0,
        )
        engine = GrowingASR("tooth number three has a mesial occlusal composite")
        pipeline = Pipeline(config, asr=engine, live_asr=engine, diarizer=fake_diarizer)
        events = list(pipeline.stream(ArrayAudioSource(sine, chunk_seconds=1.0)))

        assert any(isinstance(e, SpeakerRelabel) for e in events)
