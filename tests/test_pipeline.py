"""Pipeline assembly, config, registry, and the offline guarantee."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

from flowscribe import Config
from flowscribe.audio import ArrayAudioSource
from flowscribe.config import ASRConfig, CorrectionConfig
from flowscribe.errors import BackendNotFound, ConfigError, OfflineViolation
from flowscribe.pipeline import Pipeline
from flowscribe.registry import ASR, SINK, Registry


class TestRegistry:
    def test_builtin_backends_discoverable(self) -> None:
        assert "faster-whisper" in ASR.names()
        assert {"json", "text", "srt", "vtt"} <= set(SINK.names())

    def test_listing_does_not_import(self) -> None:
        """Listing must stay free, or an uninstalled extra breaks `backends`."""
        Registry("flowscribe.asr").names()

    def test_runtime_registration_wins(self) -> None:
        reg: Registry = Registry("flowscribe.test")

        class Custom:
            pass

        reg.register("custom", Custom)
        assert reg.get("custom") is Custom
        assert "custom" in reg.names()

    def test_unknown_backend_lists_alternatives(self) -> None:
        with pytest.raises(BackendNotFound, match="Available"):
            Registry("flowscribe.asr").get("nonexistent")

    def test_create_instantiates(self) -> None:
        reg: Registry = Registry("flowscribe.test")

        class Custom:
            def __init__(self, value: int = 0) -> None:
                self.value = value

        reg.register("custom", Custom)
        assert reg.create("custom", value=7).value == 7


class TestConfig:
    def test_defaults_are_offline(self) -> None:
        """PHI-safe by default; opting out has to be explicit."""
        assert Config().offline_only is True

    def test_live_tier_is_cheaper_than_final(self) -> None:
        config = Config()
        assert config.live.beam_size < config.final.beam_size

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("offline_only: false\nfinal:\n  model: tiny.en\n  language: en\n")
        config = Config.from_yaml(path)
        assert config.offline_only is False
        assert config.final.model == "tiny.en"

    def test_yaml_overrides_win(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("offline_only: true\n")
        assert Config.from_yaml(path, offline_only=False).offline_only is False

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            Config.from_yaml(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("key: [unclosed\n")
        with pytest.raises(ConfigError):
            Config.from_yaml(path)

    def test_apply_offline_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        Config(offline_only=True).apply_offline_env()
        import os

        assert os.environ["HF_HUB_OFFLINE"] == "1"


class TestOfflineGuarantee:
    """The basis for handling patient recordings."""

    def test_rejects_non_loopback_correction_endpoint(self) -> None:
        config = Config(
            offline_only=True,
            correction=CorrectionConfig(
                enabled=True, backend="passthrough", endpoint="http://api.example.com"
            ),
        )
        with pytest.raises(OfflineViolation, match="loopback"):
            Pipeline(config)

    @pytest.mark.parametrize(
        "endpoint",
        ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"],
    )
    def test_allows_loopback(self, endpoint: str) -> None:
        config = Config(
            offline_only=True,
            correction=CorrectionConfig(enabled=True, backend="passthrough", endpoint=endpoint),
        )
        Pipeline(config)  # must not raise

    def test_remote_endpoint_allowed_when_opted_out(self) -> None:
        config = Config(
            offline_only=False,
            correction=CorrectionConfig(
                enabled=True, backend="passthrough", endpoint="http://api.example.com"
            ),
        )
        Pipeline(config)

    def test_runs_with_networking_blocked(
        self, monkeypatch: pytest.MonkeyPatch, fake_asr, sine: np.ndarray
    ) -> None:
        """End-to-end with every outbound socket refused.

        This is the check that makes the offline claim real rather than a
        configuration flag nothing enforces.
        """

        def refuse(*args: object, **kwargs: object):
            raise OSError("network access blocked during test")

        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)

        pipeline = Pipeline(Config(offline_only=True), asr=fake_asr)
        result = pipeline.transcribe(ArrayAudioSource(sine))
        assert result.verbatim.text


class TestPipeline:
    def test_produces_verbatim_transcript(self, fake_asr, sine: np.ndarray) -> None:
        result = Pipeline(Config(), asr=fake_asr).transcribe(ArrayAudioSource(sine))
        assert "tooth number three" in result.verbatim.text
        assert result.verbatim.tier == "final"
        assert fake_asr.calls == 1

    def test_corrected_is_none_when_disabled(self, fake_asr, sine: np.ndarray) -> None:
        config = Config(correction=CorrectionConfig(enabled=False))
        result = Pipeline(config, asr=fake_asr).transcribe(ArrayAudioSource(sine))
        assert result.corrected is None
        assert result.best is result.verbatim

    def test_retains_both_transcripts(self, sine: np.ndarray) -> None:
        """Both must survive so correction can be measured, not trusted."""
        from tests.conftest import FakeASREngine, FakeCorrector

        config = Config(correction=CorrectionConfig(enabled=True, backend="passthrough"))
        pipeline = Pipeline(
            config,
            asr=FakeASREngine(text="depths on the buckle"),
            corrector=FakeCorrector(),
        )
        result = pipeline.transcribe(ArrayAudioSource(sine))

        assert "buckle" in result.verbatim.text
        assert "buccal" in result.corrected.text
        assert result.best is result.corrected

    def test_edits_are_recorded(self, sine: np.ndarray) -> None:
        """Every change auditable: an LLM-rewritten clinical transcript without a
        diff is not verifiable."""
        from tests.conftest import FakeASREngine, FakeCorrector

        config = Config(correction=CorrectionConfig(enabled=True, backend="passthrough"))
        pipeline = Pipeline(
            config, asr=FakeASREngine(text="depths on the buckle"), corrector=FakeCorrector()
        )
        result = pipeline.transcribe(ArrayAudioSource(sine))

        assert len(result.edits) == 1
        assert result.edits[0].before != result.edits[0].after
        assert result.edits[0].kind == "term"

    def test_diarizer_assigns_speakers(self, fake_asr, fake_diarizer, sine: np.ndarray) -> None:
        from flowscribe.config import DiarizationConfig

        config = Config(diarization=DiarizationConfig(enabled=True, backend="passthrough"))
        pipeline = Pipeline(config, asr=fake_asr, diarizer=fake_diarizer)
        result = pipeline.transcribe(ArrayAudioSource(sine))
        assert all(u.speaker is not None for u in result.verbatim.utterances)

    def test_null_diarizer_leaves_speakers_unset(self, fake_asr, sine: np.ndarray) -> None:
        """Not SPEAKER_00: a fabricated single label is indistinguishable from a
        real single-speaker result and would misattribute the patient."""
        from flowscribe.config import DiarizationConfig

        config = Config(diarization=DiarizationConfig(enabled=True, backend="passthrough"))
        result = Pipeline(config, asr=fake_asr).transcribe(ArrayAudioSource(sine))
        assert all(u.speaker is None for u in result.verbatim.utterances)

    def test_utterance_ids_are_contiguous(self, sine: np.ndarray) -> None:
        from tests.conftest import FakeASREngine

        result = Pipeline(Config(), asr=FakeASREngine()).transcribe(ArrayAudioSource(sine))
        ids = [u.id for u in result.verbatim.utterances]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)

    def test_hotwords_reach_the_engine(self, fake_asr, sine: np.ndarray) -> None:
        Pipeline(Config(), asr=fake_asr).transcribe(
            ArrayAudioSource(sine), hotwords=["mesial", "buccal"]
        )
        assert fake_asr.last_hotwords == ["mesial", "buccal"]

    def test_empty_audio_is_not_an_error(self, fake_asr) -> None:
        source = ArrayAudioSource(np.zeros(0, dtype=np.float32))
        result = Pipeline(Config(), asr=fake_asr).transcribe(source)
        assert result.verbatim.utterances == []

    def test_stats_are_populated(self, fake_asr, sine: np.ndarray) -> None:
        pipeline = Pipeline(Config(), asr=fake_asr)
        pipeline.transcribe(ArrayAudioSource(sine))
        assert pipeline.stats.audio_seconds == pytest.approx(3.0, rel=0.05)
        assert pipeline.stats.real_time_factor > 0

    def test_context_manager_closes(self, fake_asr, sine: np.ndarray) -> None:
        with Pipeline(Config(), asr=fake_asr) as pipeline:
            pipeline.transcribe(ArrayAudioSource(sine))

    def test_lazy_stages_not_built_when_disabled(self) -> None:
        """A config with correction off must never import an LLM client."""
        config = Config(
            correction=CorrectionConfig(enabled=False),
            final=ASRConfig(backend="does-not-exist"),
        )
        pipeline = Pipeline(config)
        assert pipeline.corrector is None
