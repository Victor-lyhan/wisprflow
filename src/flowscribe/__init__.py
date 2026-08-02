"""flowscribe -- modular, fully-local speech-to-text for dental clinical audio.

Typical use::

    from flowscribe import Config, transcribe_file

    result = transcribe_file("visit.wav", Config())
    print(result.verbatim.text)     # raw ASR
    print(result.best.text)         # corrected, when correction is enabled
    for edit in result.edits:       # every change the corrector made
        print(edit.before, "->", edit.after)

Only ``contracts`` and ``config`` are imported eagerly; engines load on first use
so that uninstalled optional extras cost nothing.
"""

from .config import (
    ASRConfig,
    Config,
    CorrectionConfig,
    DiarizationConfig,
    VADConfig,
)
from .contracts import (
    AudioChunk,
    AudioMeta,
    Edit,
    Event,
    FinalUtterance,
    PartialUtterance,
    SpeakerRelabel,
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
    FlowscribeError,
    MissingDependency,
    OfflineViolation,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "Config",
    "ASRConfig",
    "VADConfig",
    "DiarizationConfig",
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
    "Event",
    "PartialUtterance",
    "FinalUtterance",
    "SpeakerRelabel",
    "TranscriptComplete",
    # errors
    "FlowscribeError",
    "BackendNotFound",
    "MissingDependency",
    "ConfigError",
    "AudioError",
    "OfflineViolation",
    # lazy
    "Pipeline",
    "transcribe_file",
]


def __getattr__(name: str):
    """Defer pipeline import so ``import flowscribe`` stays cheap."""
    if name in ("Pipeline", "transcribe_file"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
