"""Parakeet TDT via ONNX Runtime.

The intended production engine. NVIDIA's Parakeet-TDT-0.6b-v3 is a quarter of
Whisper large-v3's size at comparable accuracy, and the community ONNX export
runs it through ONNX Runtime without NeMo -- which is what makes one codebase
serve macOS development and Windows CPU production.

Measured on this repo's dental sample: 15.9s of audio in 0.80s on CPU (RTF 0.05),
no GPU involved. That is the number that matters for a clinic machine with no
dedicated graphics.

Licence: CC-BY-4.0. Covers 25 languages with automatic language identification,
against Whisper's 99 -- so Whisper remains the default for the final tier where
coverage matters more than speed, and this is the live-tier and CPU-box choice.

Requires: ``pip install 'flowscribe[parakeet]'``
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..contracts import AudioChunk, Utterance, Word, utterance_id
from ..errors import MissingDependency

__all__ = ["ParakeetOnnxEngine", "DEFAULT_MODEL"]

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"

# Preference order for execution providers. CUDA first where present; DirectML
# next because on Windows it reaches any GPU -- NVIDIA, AMD or Intel -- which
# matters when the clinic hardware is unknown. CPU is the floor and is always
# available.
#
# CoreML is deliberately absent despite being available on Apple Silicon. It
# supports under half this model's nodes (1012 of 2115), so ONNX Runtime splits
# the graph into ~291 partitions and the transfer overhead dominates: measured
# RTF 3.69 against 0.05 for plain CPU on the same audio -- roughly 70x slower.
# It remains reachable via an explicit `device="coreml"`.
_PROVIDER_PREFERENCE = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)

# A pause longer than this ends an utterance. Deliberately generous: clinical
# dictation is full of mid-sentence pauses while the operator is working, and
# splitting there fragments terms like "mesial occlusal distal" across
# utterances, which then breaks phrase matching in the lexicon.
UTTERANCE_GAP_SECONDS = 0.8

_SENTENCE_END = (".", "?", "!")


def _resolve_providers(device: str) -> list[str]:
    """Pick execution providers available in this ONNX Runtime build."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())

    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise MissingDependency(
                "device='cuda' requested but CUDAExecutionProvider is unavailable. "
                "Install onnxruntime-gpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    chosen = [p for p in _PROVIDER_PREFERENCE if p in available]
    return chosen or ["CPUExecutionProvider"]


def _assemble_words(
    tokens: list[str], starts: list[float], logprobs: list[float] | None, offset: float, end: float
) -> list[Word]:
    """Merge subword tokens into words.

    The tokenizer emits pieces like ``[' To', 'oth', ' num', 'ber']`` where a
    leading space marks a word boundary. Timestamps are per token, so a word's
    span runs from its first piece to the start of the next word.
    """
    groups: list[tuple[list[str], float, list[float]]] = []
    # A whitespace-only token carries no text but *is* a word boundary. Dropping
    # it outright made the following token glue onto the previous word, which
    # produced "number3", "tooth number19" and "epinephrine1 to100,000" -- and
    # the run-together numbers then defeated tooth extraction entirely.
    boundary_pending = False

    for index, token in enumerate(tokens):
        confidence = (
            float(np.exp(logprobs[index]))
            if logprobs is not None and index < len(logprobs)
            else None
        )
        text = token.strip()
        if not text:
            boundary_pending = True
            continue

        starts_word = token.startswith(" ") or boundary_pending or not groups
        boundary_pending = False

        if starts_word:
            groups.append(([text], starts[index], [confidence] if confidence is not None else []))
        else:
            groups[-1][0].append(text)
            if confidence is not None:
                groups[-1][2].append(confidence)

    words: list[Word] = []
    for position, (pieces, start, confidences) in enumerate(groups):
        stop = groups[position + 1][1] if position + 1 < len(groups) else end
        text = "".join(pieces).strip()
        if not text:
            continue
        words.append(
            Word(
                text=text,
                start=offset + start,
                end=offset + max(stop, start),
                confidence=sum(confidences) / len(confidences) if confidences else None,
            )
        )
    return words


def _segment(words: list[Word], offset: float) -> list[Utterance]:
    """Group words into utterances on sentence punctuation or a long pause.

    Parakeet returns one flat sequence, unlike Whisper which segments for you. A
    transcript that is a single unbroken utterance is unreadable for review and
    gives the streaming policy nothing to anchor on, so boundaries are derived
    here.
    """
    if not words:
        return []

    utterances: list[Utterance] = []
    current: list[Word] = []

    for index, word in enumerate(words):
        current.append(word)
        following = words[index + 1] if index + 1 < len(words) else None
        ends_sentence = word.text.endswith(_SENTENCE_END)
        long_pause = following is not None and (following.start - word.end) >= UTTERANCE_GAP_SECONDS

        if following is None or ends_sentence or long_pause:
            confidences = [w.confidence for w in current if w.confidence is not None]
            utterances.append(
                Utterance(
                    id=utterance_id(len(utterances)),
                    start=current[0].start,
                    end=current[-1].end,
                    text=" ".join(w.text for w in current),
                    words=list(current),
                    confidence=sum(confidences) / len(confidences) if confidences else None,
                    is_final=True,
                )
            )
            current = []

    return utterances


class ParakeetOnnxEngine:
    """Parakeet TDT through ONNX Runtime."""

    name = "parakeet-onnx"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        quantization: str | None = None,
        model_dir: str | None = None,
        **options: object,
    ) -> None:
        try:
            import onnx_asr
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise MissingDependency(
                "onnx-asr is not installed. Install with: pip install 'flowscribe[parakeet]'"
            ) from exc

        self.model_name = model
        self.providers = _resolve_providers(device)
        self.device = self.providers[0].replace("ExecutionProvider", "").lower()
        self._options = options

        # onnx-asr's `path` means "a directory already containing the .onnx
        # files", not "where to cache downloads" -- the opposite of
        # faster-whisper's `download_root`. Passing the pipeline's model_dir
        # through as `path` makes it look for weights that were never put there.
        # So model_dir steers the Hugging Face cache instead, and an explicitly
        # pre-staged model directory is a separate option.
        raw_path = options.pop("path", None)
        local_path = str(raw_path) if raw_path is not None else None
        if local_path is None and model_dir:
            os.environ.setdefault("HF_HUB_CACHE", str(model_dir))

        loaded = onnx_asr.load_model(
            model,
            path=local_path,
            quantization=quantization,
            providers=self.providers,
        )
        # Word timings are needed by the streaming policy and by any consumer
        # aligning text back to audio, so timestamps are always on.
        self._model: Any = loaded.with_timestamps()

    def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        hotwords: Iterable[str] = (),
    ) -> list[Utterance]:
        if len(audio.pcm) == 0:
            return []

        kwargs: dict[str, Any] = {"sample_rate": audio.sample_rate}
        if language:
            kwargs["language"] = language

        result = self._model.recognize(np.ascontiguousarray(audio.pcm, dtype=np.float32), **kwargs)

        tokens = list(getattr(result, "tokens", []) or [])
        starts = list(getattr(result, "timestamps", []) or [])
        if not tokens or not starts:
            text = str(getattr(result, "text", "") or "").strip()
            if not text:
                return []
            # No timing available: emit one utterance spanning the chunk rather
            # than fabricating word timings.
            return [
                Utterance(
                    id=utterance_id(0),
                    start=audio.start,
                    end=audio.end,
                    text=text,
                    language=language,
                    is_final=True,
                )
            ]

        logprobs = list(getattr(result, "logprobs", []) or []) or None
        words = _assemble_words(tokens, starts, logprobs, offset=audio.start, end=audio.duration)
        utterances = _segment(words, audio.start)
        if language:
            utterances = [u.model_copy(update={"language": language}) for u in utterances]
        return utterances

    def close(self) -> None:
        self._model = None
