"""Configuration.

Config selects a backend per stage and carries that backend's options. Adding a
new engine therefore never requires touching this module -- unknown keys go
through to the backend constructor.

Settings resolve from, in increasing precedence: defaults, a YAML file,
``DENTASCRIBE_*`` environment variables, and explicit keyword arguments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError

__all__ = ["Config", "ASRConfig", "VADConfig", "CorrectionConfig"]


def _default_model_dir() -> Path:
    return Path(os.environ.get("DENTASCRIBE_MODEL_DIR", Path.home() / ".dentascribe" / "models"))


class ASRConfig(BaseModel):
    """Recognition settings for one tier."""

    backend: str = "faster-whisper"
    model: str = "large-v3"
    language: str | None = None
    """BCP-47 code, or ``None`` to auto-detect. Auto-detect is the default because
    this deployment is multilingual; pin it when a clinic is known monolingual,
    since detection can flip mid-recording on short or noisy segments."""

    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "auto"
    """Quantization. ``auto`` picks int8 on CPU and float16 on CUDA."""

    beam_size: int = 5
    vad_filter: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class VADConfig(BaseModel):
    enabled: bool = False
    """Skip decoding blocks that contain no speech.

    Off by default because it needs the `vad` extra. Worth enabling for real
    appointments: much of an operatory recording is the operator working without
    speaking, and decoding that audio is pure waste."""

    backend: str = "silero"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 400
    """Silence needed to close an utterance. Tuned longer than a general-purpose
    default: clinical speech is full of mid-sentence pauses while an operator is
    working, and splitting there fragments terms across utterances."""

    options: dict[str, Any] = Field(default_factory=dict)


class CorrectionConfig(BaseModel):
    enabled: bool = True
    backend: str = "passthrough"
    model: str = "qwen3:8b"
    endpoint: str = "http://127.0.0.1:11434"
    """Local inference server. Loopback only -- a remote endpoint would send PHI
    off the machine, so ``offline_only`` rejects any non-loopback host."""

    max_edit_ratio: float = 0.25
    """Reject a correction that rewrites more than this fraction of an utterance.

    A constrained corrector fixes terms; one that rewrites a quarter of the words
    is paraphrasing or hallucinating, and its output should be discarded rather
    than shown to a clinician."""

    lexicon_path: Path | None = None
    user_codes_path: Path | None = None
    """Optional practice-supplied procedure code list.

    CDT and SNODENT are ADA copyright and require a commercial license, so they
    are never bundled. A licensed practice points this at its own file."""

    options: dict[str, Any] = Field(default_factory=dict)


class Config(BaseSettings):
    """Top-level pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DENTASCRIBE_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    final: ASRConfig = Field(default_factory=ASRConfig)
    """Authoritative pass: full context, best available model."""

    live: ASRConfig = Field(default_factory=lambda: ASRConfig(model="large-v3-turbo", beam_size=1))
    """Streaming pass: smaller and greedy, because latency dominates and its
    output is provisional anyway. Intended to move to ``parakeet-onnx`` once that
    backend lands -- it is markedly faster on CPU, which is the Windows target."""

    vad: VADConfig = Field(default_factory=VADConfig)
    correction: CorrectionConfig = Field(default_factory=CorrectionConfig)

    flag_confusable: bool = True
    """Flag clinically confusable terms for human review.

    Cheap (a lexicon lookup, no model) and independent of the corrector, so it
    stays on even with correction disabled. Catches the error class where a
    misrecognition lands on another valid clinical term."""

    offline_only: bool = True
    """Forbid all network access during inference.

    On by default. Models are fetched in an explicit provisioning step
    (``dentascribe fetch-models``); anything that would download at inference time
    is a bug, and this makes it fail loudly instead of silently reaching out
    while patient audio is in memory."""

    model_dir: Path = Field(default_factory=_default_model_dir)
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "transcripts")

    chunk_seconds: float = 1.0
    """Audio block size for streaming. Smaller lowers latency and raises overhead."""

    confirm_after: int = 2
    """LocalAgreement-N: how many successive decodes must agree before text is
    emitted as final. Two is the published setting and the usual quality/latency
    knee."""

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> Config:
        """Load from YAML, with keyword overrides winning."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(data).__name__}")
        return cls(**{**data, **overrides})

    def apply_offline_env(self) -> None:
        """Put the model libraries into offline mode and silence telemetry.

        Called before any backend is constructed. These libraries check their
        environment at import time, so setting them afterwards has no effect.
        """
        if not self.offline_only:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
