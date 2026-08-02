"""Evaluation datasets.

A dataset is a JSONL manifest, one sample per line::

    {"id": "perio-01", "audio": "audio/perio-01.wav", "reference": "tooth number three ..."}

Paths are resolved relative to the manifest, so a dataset directory relocates
intact. ``reference`` is the ground-truth transcript.

On sourcing: there is no public dental ASR corpus -- recent dental and
orthodontic ASR studies each built private ones. Two datasets are expected here:
PriMock57 (public mock primary-care consultations) to validate the pipeline
end to end, and a locally recorded dental gold set to measure domain accuracy.
Reading pre-written scripts makes the script itself the reference, which yields
exactly-aligned ground truth without a transcription pass.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ConfigError

__all__ = ["Sample", "load_manifest", "iter_manifest"]


@dataclass
class Sample:
    """One evaluation item."""

    id: str
    audio: Path
    reference: str
    language: str | None = None
    speakers: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def iter_manifest(path: str | Path) -> Iterator[Sample]:
    """Stream samples from a JSONL manifest."""
    manifest = Path(path)
    if not manifest.exists():
        raise ConfigError(f"Manifest not found: {manifest}")

    root = manifest.parent
    for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{manifest}:{lineno}: invalid JSON: {exc}") from exc

        missing = {"audio", "reference"} - record.keys()
        if missing:
            raise ConfigError(
                f"{manifest}:{lineno}: missing field(s): {', '.join(sorted(missing))}"
            )

        audio = Path(record["audio"])
        if not audio.is_absolute():
            audio = root / audio

        yield Sample(
            id=str(record.get("id", audio.stem)),
            audio=audio,
            reference=record["reference"],
            language=record.get("language"),
            speakers=record.get("speakers"),
            metadata=record.get("metadata", {}),
        )


def load_manifest(path: str | Path) -> list[Sample]:
    """Load all samples, verifying that the audio exists.

    Fails on the first missing file rather than scoring what happens to be
    present -- a run over a partial dataset that reports a headline number is
    worse than one that refuses to start.
    """
    samples = list(iter_manifest(path))
    if not samples:
        raise ConfigError(f"No samples in {path}")

    missing = [s.audio for s in samples if not s.audio.exists()]
    if missing:
        shown = ", ".join(str(p) for p in missing[:3])
        more = f" (and {len(missing) - 3} more)" if len(missing) > 3 else ""
        raise ConfigError(f"Missing audio: {shown}{more}")

    return samples
