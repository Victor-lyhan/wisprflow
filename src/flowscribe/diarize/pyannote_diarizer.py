"""Offline speaker diarization via pyannote.audio.

Used for the final tier: it needs the whole recording to cluster speakers, which
is precisely why the live tier cannot rely on it and why the two-tier design
exists.

Licensing: ``pyannote/speaker-diarization-3.1`` is MIT and the newer
``community-1`` pipeline is CC-BY-4.0. Both are self-hostable commercially. Both
are also **gated on Hugging Face**: you must accept the model conditions with
your account and supply a token once, at provisioning time. After that the
weights are cached and inference runs entirely offline, which is what keeps the
PHI guarantee intact.

    huggingface-cli login              # once
    flowscribe fetch-models --diarizer pyannote/speaker-diarization-3.1

Requires: ``pip install 'flowscribe[diarize]'``
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..contracts import TARGET_SAMPLE_RATE, AudioChunk, Utterance
from ..errors import MissingDependency

__all__ = ["PyannoteDiarizer"]

DEFAULT_PIPELINE = "pyannote/speaker-diarization-3.1"


class PyannoteDiarizer:
    """Assigns speaker labels by clustering over the whole recording."""

    name = "pyannote"
    online = False

    def __init__(
        self,
        model: str = DEFAULT_PIPELINE,
        *,
        min_speakers: int = 1,
        max_speakers: int = 4,
        device: str = "auto",
        token: str | None = None,
        model_dir: str | None = None,
        **options: object,
    ) -> None:
        try:
            import torch
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise MissingDependency(
                "pyannote.audio is not installed. Install with: pip install 'flowscribe[diarize]'"
            ) from exc

        self.model = model
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._options = options
        self._torch = torch

        pipeline = PyannotePipeline.from_pretrained(
            model,
            use_auth_token=token or os.environ.get("HF_TOKEN"),
            cache_dir=model_dir,
        )
        if pipeline is None:
            raise MissingDependency(
                f"Could not load {model!r}. It is gated on Hugging Face: accept the "
                "model conditions on its model page, then run `huggingface-cli login` "
                "or set HF_TOKEN. This is a one-time provisioning step; inference "
                "afterwards is offline."
            )

        self.device = self._resolve_device(device, torch)
        self._pipeline = pipeline.to(torch.device(self.device))

    @staticmethod
    def _resolve_device(device: str, torch: Any) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        # MPS is deliberately not selected: several pyannote ops fall back to CPU
        # on Metal and the mixed placement is slower than staying on CPU, besides
        # having produced incorrect results in some torch releases.
        return "cpu"

    def assign(
        self,
        utterances: list[Utterance],
        *,
        audio: AudioChunk | None = None,
        num_speakers: int | None = None,
    ) -> list[Utterance]:
        """Label each utterance with the speaker who dominates its time span."""
        if not utterances or audio is None or len(audio.pcm) == 0:
            return utterances

        waveform = self._torch.from_numpy(
            np.ascontiguousarray(audio.pcm, dtype=np.float32)
        ).unsqueeze(0)

        constraints: dict[str, int] = {}
        if num_speakers is not None:
            constraints["num_speakers"] = num_speakers
        else:
            constraints["min_speakers"] = self.min_speakers
            constraints["max_speakers"] = self.max_speakers

        annotation = self._pipeline(
            {"waveform": waveform, "sample_rate": TARGET_SAMPLE_RATE}, **constraints
        )

        turns = [
            (segment.start, segment.end, str(speaker))
            for segment, _, speaker in annotation.itertracks(yield_label=True)
        ]
        if not turns:
            return utterances

        return [u.model_copy(update={"speaker": self._dominant(u, turns)}) for u in utterances]

    @staticmethod
    def _dominant(utterance: Utterance, turns: list[tuple[float, float, str]]) -> str | None:
        """Speaker holding the most time within an utterance.

        Diarization turns and ASR segments are produced independently and rarely
        align, so an utterance routinely overlaps two or more turns. Attributing
        it to whoever holds the most of it is the standard reconciliation; a
        boundary-based rule would misassign short interjections, which in an
        operatory are usually the patient answering.
        """
        totals: dict[str, float] = {}
        for start, end, speaker in turns:
            overlap = min(utterance.end, end) - max(utterance.start, start)
            if overlap > 0:
                totals[speaker] = totals.get(speaker, 0.0) + overlap
        if not totals:
            return None
        return max(totals, key=lambda s: totals[s])

    def reset(self) -> None:
        pass
