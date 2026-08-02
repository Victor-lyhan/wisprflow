"""Exception hierarchy.

Everything derives from ``FlowscribeError`` so an embedding application can
catch this library's failures without catching unrelated ones.
"""

from __future__ import annotations

__all__ = [
    "FlowscribeError",
    "BackendNotFound",
    "MissingDependency",
    "ConfigError",
    "AudioError",
    "OfflineViolation",
]


class FlowscribeError(Exception):
    """Base class for all flowscribe errors."""


class BackendNotFound(FlowscribeError):
    """A backend name does not resolve to anything registered."""


class MissingDependency(FlowscribeError):
    """A backend exists but its optional extra is not installed."""


class ConfigError(FlowscribeError):
    """Configuration is invalid or internally inconsistent."""


class AudioError(FlowscribeError):
    """Audio could not be read, decoded, or resampled."""


class OfflineViolation(FlowscribeError):
    """Something attempted a network call while ``offline_only`` was set.

    This is a hard error rather than a warning: the offline guarantee is the
    basis for handling patient recordings, and a silent fallback to downloading
    a model at inference time would break it exactly when it matters.
    """
