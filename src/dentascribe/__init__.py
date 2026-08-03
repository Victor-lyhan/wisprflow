"""dentascribe -- modular, fully-local speech-to-text for dental clinical audio.

Typical use::

    from dentascribe import Config, transcribe_file

    result = transcribe_file("visit.wav", Config())
    print(result.verbatim.text)     # raw ASR
    print(result.best.text)         # corrected, when correction is enabled
    for edit in result.edits:       # every change the corrector made
        print(edit.before, "->", edit.after)

Only ``contracts`` and ``config`` are imported eagerly; engines load on first use
so that uninstalled optional extras cost nothing.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .config import (
    ASRConfig,
    Config,
    CorrectionConfig,
    VADConfig,
)
from .contracts import (
    AudioChunk,
    AudioMeta,
    Edit,
    Event,
    FinalUtterance,
    Flag,
    PartialUtterance,
    Transcript,
    TranscriptComplete,
    TranscriptionResult,
    Utterance,
    VadSegment,
    Word,
)
from .errors import (
    AudioError,
    BackendNotFound,
    ConfigError,
    DentascribeError,
    MissingDependency,
    OfflineViolation,
)

try:
    __version__ = version("dentascribe")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # config
    "Config",
    "ASRConfig",
    "VADConfig",
    "CorrectionConfig",
    # contracts
    "AudioChunk",
    "AudioMeta",
    "VadSegment",
    "Word",
    "Utterance",
    "Transcript",
    "TranscriptionResult",
    "Edit",
    "Flag",
    "Event",
    "PartialUtterance",
    "FinalUtterance",
    "TranscriptComplete",
    # errors
    "DentascribeError",
    "BackendNotFound",
    "MissingDependency",
    "ConfigError",
    "AudioError",
    "OfflineViolation",
    # lazy
    "Pipeline",
    "transcribe_file",
    "stream_file",
]


def __getattr__(name: str) -> Any:
    """Defer pipeline import so ``import dentascribe`` stays cheap."""
    if name in ("Pipeline", "transcribe_file", "stream_file"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
