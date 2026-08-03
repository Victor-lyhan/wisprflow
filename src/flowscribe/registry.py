"""Backend discovery.

Backends resolve by name through ``importlib.metadata`` entry points. flowscribe's
own engines are registered the same way as anyone else's, so an external project
adding a backend uses a supported path rather than a fork:

    [project.entry-points."flowscribe.asr"]
    my-engine = "my_package.engines:MyEngine"

Resolution is lazy. Listing available backends never imports them, so having
``flowscribe[parakeet]`` uninstalled costs nothing until something actually
asks for that engine -- at which point the missing dependency is reported with
the extra that provides it, instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Any, Generic, TypeVar

from .errors import BackendNotFound, MissingDependency

T = TypeVar("T")

__all__ = ["Registry", "ASR", "CORRECTOR", "SINK", "VAD_REGISTRY"]

# Which optional extra provides which backend. Used only to turn an ImportError
# into an actionable message.
_EXTRA_HINTS: dict[str, str] = {
    "faster-whisper": "whisper",
    "whisper": "whisper",
    "parakeet-onnx": "parakeet",
    "parakeet": "parakeet",
    "mlx-whisper": "mlx",
    "ollama": "llm",
    "llama-cpp": "llm",
    "silero": "vad",
    "deepfilternet": "denoise",
}


class Registry(Generic[T]):
    """A named collection of backends for one pipeline stage."""

    def __init__(self, group: str) -> None:
        self.group = group
        self._local: dict[str, type[T]] = {}
        self._entry_points: dict[str, EntryPoint] | None = None

    def register(self, name: str, backend: type[T]) -> type[T]:
        """Register a backend directly.

        Used by tests and by projects that build backends at runtime rather than
        declaring them in package metadata. Takes precedence over entry points,
        which is what lets a test substitute a fake for a real engine.
        """
        self._local[name] = backend
        return backend

    def _discover(self) -> dict[str, EntryPoint]:
        if self._entry_points is None:
            self._entry_points = {ep.name: ep for ep in entry_points(group=self.group)}
        return self._entry_points

    def names(self) -> list[str]:
        """All known backend names. Does not import anything."""
        return sorted(set(self._local) | set(self._discover()))

    def get(self, name: str) -> type[T]:
        """Resolve a backend class by name, importing it on first use."""
        if name in self._local:
            return self._local[name]

        ep = self._discover().get(name)
        if ep is None:
            raise BackendNotFound(
                f"No {self.group.split('.')[-1]} backend named {name!r}. "
                f"Available: {', '.join(self.names()) or '(none)'}"
            )

        try:
            return ep.load()  # type: ignore[no-any-return]
        except ImportError as exc:
            extra = _EXTRA_HINTS.get(name)
            hint = (
                f" Install it with: pip install 'flowscribe[{extra}]'"
                if extra
                else " Its dependencies are not installed."
            )
            raise MissingDependency(f"Backend {name!r} could not be loaded.{hint}") from exc

    def create(self, name: str, **kwargs: Any) -> T:
        """Resolve and instantiate in one step."""
        return self.get(name)(**kwargs)


# One registry per stage. Import these rather than constructing your own so that
# a runtime `register()` is visible to the pipeline builder.
ASR: Registry[Any] = Registry("flowscribe.asr")
CORRECTOR: Registry[Any] = Registry("flowscribe.corrector")
SINK: Registry[Any] = Registry("flowscribe.sink")
VAD_REGISTRY: Registry[Any] = Registry("flowscribe.vad")
