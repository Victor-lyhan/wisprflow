"""faster-whisper ASR backend.

CTranslate2 rather than PyTorch Whisper, because it runs well on CPU with INT8
quantization and installs cleanly on Windows without CUDA -- which is the
production constraint. Whisper is the default engine for the final tier for its
language coverage (99 languages against Parakeet's 25).

Requires: ``pip install 'flowscribe[whisper]'``
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..contracts import AudioChunk, Utterance, Word, utterance_id
from ..errors import MissingDependency

__all__ = ["FasterWhisperEngine"]

# Whisper's decoder prompt only consumes its final 224 tokens, and stuffing it
# raises the hallucination rate. Domain vocabulary is handled properly by the
# correction stage; this is a small nudge, not the lexicon.
MAX_HOTWORDS = 40


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    # No Metal path here: CTranslate2 has no Apple GPU backend, so Mac runs CPU.
    # That is acceptable for a dev machine and is why mlx-whisper exists as a
    # separate local-iteration backend.
    return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _confidence(avg_logprob: float | None) -> float | None:
    """Map Whisper's average log-probability into ``[0, 1]``.

    Monotonic but not calibrated -- useful for ranking and for flagging
    low-confidence spans to a reviewer, not as a probability of correctness.
    """
    if avg_logprob is None:
        return None
    return max(0.0, min(1.0, math.exp(avg_logprob)))


class FasterWhisperEngine:
    """Whisper via CTranslate2."""

    name = "faster-whisper"

    def __init__(
        self,
        model: str = "large-v3",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        model_dir: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        cpu_threads: int = 0,
        **options: object,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingDependency(
                "faster-whisper is not installed. Install with: pip install 'flowscribe[whisper]'"
            ) from exc

        self.device = _resolve_device(device)
        self.compute_type = _resolve_compute_type(compute_type, self.device)
        self.model_name = model
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._options = options

        self._model = WhisperModel(
            model,
            device=self.device,
            compute_type=self.compute_type,
            download_root=model_dir,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        hotwords: Iterable[str] = (),
    ) -> list[Utterance]:
        if len(audio.pcm) == 0:
            return []

        terms = list(hotwords)[:MAX_HOTWORDS]
        kwargs: dict[str, object] = {
            "beam_size": self.beam_size,
            "language": language,
            "word_timestamps": True,
            "vad_filter": self.vad_filter,
            **self._options,
        }
        if terms:
            kwargs["hotwords"] = " ".join(terms)

        segments, info = self._model.transcribe(audio.pcm, **kwargs)
        detected = getattr(info, "language", None)

        utterances: list[Utterance] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue

            words = [
                Word(
                    text=w.word.strip(),
                    # Absolute session time, not time within this chunk, so
                    # timestamps stay valid when audio is processed in pieces.
                    start=audio.start + w.start,
                    end=audio.start + w.end,
                    confidence=getattr(w, "probability", None),
                )
                for w in (seg.words or [])
                if w.word.strip()
            ]

            utterances.append(
                Utterance(
                    id=utterance_id(len(utterances)),
                    start=audio.start + seg.start,
                    end=audio.start + seg.end,
                    text=text,
                    words=words,
                    language=language or detected,
                    confidence=_confidence(getattr(seg, "avg_logprob", None)),
                    is_final=True,
                )
            )

        return utterances

    def close(self) -> None:
        self._model = None
