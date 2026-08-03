"""Exception hierarchy.

Everything derives from ``DentascribeError`` so an embedding application can
catch this library's failures without catching unrelated ones.
"""

from __future__ import annotations

__all__ = [
    "DentascribeError",
    "BackendNotFound",
    "MissingDependency",
    "ConfigError",
    "AudioError",
    "OfflineViolation",
]


class DentascribeError(Exception):
    """Base class for all dentascribe errors."""


class BackendNotFound(DentascribeError):
    """A backend name does not resolve to anything registered."""


class MissingDependency(DentascribeError):
    """A backend exists but its optional extra is not installed."""


class ConfigError(DentascribeError):
    """Configuration is invalid or internally inconsistent."""


class AudioError(DentascribeError):
    """Audio could not be read, decoded, or resampled."""


class OfflineViolation(DentascribeError):
    """Something attempted a network call while ``offline_only`` was set.

    This is a hard error rather than a warning: the offline guarantee is the
    basis for handling patient recordings, and a silent fallback to downloading
    a model at inference time would break it exactly when it matters.
    """
