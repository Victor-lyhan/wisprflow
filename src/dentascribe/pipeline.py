"""Pipeline assembly and execution.

Wires configured backends into a runnable pipeline. Stages are resolved lazily,
so a config with correction disabled never imports an LLM client.

Two entry points share one set of stages: ``transcribe()`` runs the final tier
over complete audio, and ``stream()`` emits provisional text during capture then
finalizes through that same tier. They call ``_run_final`` in common, so they
cannot drift apart.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .audio.preprocess import NullPreprocessor, open_source
from .config import Config
from .contracts import (
    TARGET_SAMPLE_RATE,
    AudioChunk,
    AudioMeta,
    Edit,
    Event,
    FinalUtterance,
    PartialUtterance,
    Transcript,
    TranscriptComplete,
    TranscriptionResult,
    utterance_id,
)
from .dental.lexicon import load_confusions
from .dental.review import flag_confusions
from .errors import OfflineViolation
from .protocols import VAD, ASREngine, AudioSource, Corrector, Preprocessor
from .registry import ASR, CORRECTOR, VAD_REGISTRY
from .streaming.policy import LocalAgreement

__all__ = ["Pipeline", "RunStats", "transcribe_file", "stream_file"]


@dataclass
class RunStats:
    """Timing for one run. Reported by the eval harness."""

    audio_seconds: float = 0.0
    ingest_seconds: float = 0.0
    asr_seconds: float = 0.0
    correct_seconds: float = 0.0
    edits: int = 0
    vad_skipped_blocks: int = 0

    @property
    def total_seconds(self) -> float:
        return self.ingest_seconds + self.asr_seconds + self.correct_seconds

    @property
    def real_time_factor(self) -> float:
        """Processing time divided by audio duration. Below 1.0 is faster than real time."""
        return self.total_seconds / self.audio_seconds if self.audio_seconds else 0.0


def _check_offline(config: Config) -> None:
    """Reject settings that would send audio off the machine.

    Only the correction endpoint can reach the network at inference time, so it
    is the only thing to validate -- but it carries the full transcript, which
    makes it exactly the leak that matters.
    """
    if not config.offline_only or not config.correction.enabled:
        return
    endpoint = config.correction.endpoint
    if not any(
        endpoint.startswith(prefix)
        for prefix in ("http://127.0.0.1", "http://localhost", "http://[::1]", "http://0.0.0.0")
    ):
        raise OfflineViolation(
            f"offline_only is set but the correction endpoint is {endpoint!r}, "
            "which is not loopback. Point it at a local inference server, or set "
            "offline_only=False if sending transcripts to that host is intended."
        )


class Pipeline:
    """A configured, runnable pipeline."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        asr: ASREngine | None = None,
        live_asr: ASREngine | None = None,
        vad: VAD | None = None,
        corrector: Corrector | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        self.config = config or Config()
        _check_offline(self.config)
        # Must precede backend construction: these libraries read their
        # environment at import time.
        self.config.apply_offline_env()

        self._asr = asr
        self._live_asr = live_asr
        self._vad = vad
        self._corrector = corrector
        self.preprocessor: Preprocessor = preprocessor or NullPreprocessor()
        self.stats = RunStats()

    # -- lazily constructed stages ----------------------------------------- #

    @property
    def asr(self) -> ASREngine:
        if self._asr is None:
            cfg = self.config.final
            self._asr = ASR.create(
                cfg.backend,
                model=cfg.model,
                device=cfg.device,
                compute_type=cfg.compute_type,
                beam_size=cfg.beam_size,
                vad_filter=cfg.vad_filter,
                model_dir=str(self.config.model_dir),
                **cfg.options,
            )
        return self._asr

    @property
    def live_asr(self) -> ASREngine:
        """Engine for the live tier.

        Separate from :attr:`asr` because the tiers optimize for different things
        -- the live engine is smaller and greedy since its output is provisional
        and latency dominates, while the final engine is the most accurate
        available. When both tiers name the same backend and model, the same
        instance is reused rather than loading the weights twice.
        """
        if self._live_asr is None:
            cfg, final = self.config.live, self.config.final
            if (cfg.backend, cfg.model) == (final.backend, final.model) and (self._asr is not None):
                self._live_asr = self._asr
            else:
                self._live_asr = ASR.create(
                    cfg.backend,
                    model=cfg.model,
                    device=cfg.device,
                    compute_type=cfg.compute_type,
                    beam_size=cfg.beam_size,
                    vad_filter=cfg.vad_filter,
                    model_dir=str(self.config.model_dir),
                    **cfg.options,
                )
        return self._live_asr

    @property
    def vad(self) -> VAD | None:
        if not self.config.vad.enabled:
            return None
        if self._vad is None:
            cfg = self.config.vad
            self._vad = VAD_REGISTRY.create(
                cfg.backend,
                threshold=cfg.threshold,
                min_speech_ms=cfg.min_speech_ms,
                min_silence_ms=cfg.min_silence_ms,
                model_dir=str(self.config.model_dir),
                **cfg.options,
            )
        return self._vad

    @property
    def corrector(self) -> Corrector | None:
        if not self.config.correction.enabled:
            return None
        if self._corrector is None:
            cfg = self.config.correction
            self._corrector = CORRECTOR.create(
                cfg.backend,
                model=cfg.model,
                endpoint=cfg.endpoint,
                max_edit_ratio=cfg.max_edit_ratio,
                lexicon_path=cfg.lexicon_path,
                user_codes_path=cfg.user_codes_path,
                **cfg.options,
            )
        return self._corrector

    # -- execution ---------------------------------------------------------- #

    def _ingest(self, source: AudioSource) -> tuple[AudioChunk, AudioMeta]:
        """Drain a source into one contiguous buffer.

        The final tier decodes complete audio in one pass: Whisper's accuracy
        depends on surrounding context, so splitting here would cost quality for
        no benefit. A source that is still recording simply blocks until it ends.

        Memory is linear in duration -- roughly 230 MB per hour of float32 at
        16 kHz, which is fine for a single appointment. Multi-hour audio should
        be segmented by the caller.
        """
        pieces: list[npt.NDArray[np.float32]] = []
        for chunk in source.chunks():
            processed = self.preprocessor.process(chunk)
            if len(processed.pcm):
                pieces.append(processed.pcm)

        pcm = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        meta = source.meta()
        if meta.duration is None:
            meta = meta.model_copy(update={"duration": len(pcm) / TARGET_SAMPLE_RATE})
        return AudioChunk(pcm=pcm, sample_rate=TARGET_SAMPLE_RATE, start=0.0, is_last=True), meta

    def transcribe(
        self, source: AudioSource, *, hotwords: Iterable[str] = ()
    ) -> TranscriptionResult:
        """Run the final tier over a source and return both transcripts."""
        self.stats = RunStats()

        t0 = time.perf_counter()
        audio, meta = self._ingest(source)
        self.stats.ingest_seconds = time.perf_counter() - t0

        return self._run_final(audio, meta, hotwords=hotwords)

    def _run_final(
        self, audio: AudioChunk, meta: AudioMeta, *, hotwords: Iterable[str] = ()
    ) -> TranscriptionResult:
        """The authoritative pass over complete audio.

        Shared by :meth:`transcribe` and the finalization step of
        :meth:`stream`, so the two entry points cannot drift into producing
        different results for the same recording.
        """
        self.stats.audio_seconds = meta.duration or 0.0

        t0 = time.perf_counter()
        utterances = self.asr.transcribe(
            audio, language=self.config.final.language, hotwords=hotwords
        )
        self.stats.asr_seconds = time.perf_counter() - t0

        # Renumber so ids are contiguous and stable regardless of what the
        # backend produced, so a consumer can reference them stably.
        utterances = [
            u.model_copy(update={"id": utterance_id(i)}) for i, u in enumerate(utterances)
        ]

        verbatim = Transcript(
            utterances=utterances,
            tier="final",
            audio=meta,
            engine=getattr(self.asr, "name", self.config.final.backend),
            model=self.config.final.model,
            language=self.config.final.language
            or next((u.language for u in utterances if u.language), None),
        )

        flags = (
            flag_confusions(verbatim, load_confusions())
            if self.config.flag_confusable and utterances
            else []
        )

        corrected: Transcript | None = None
        edits: list[Edit] = []
        corrector = self.corrector
        if corrector is not None and utterances:
            t0 = time.perf_counter()
            corrected, edits = corrector.correct(verbatim)
            self.stats.correct_seconds = time.perf_counter() - t0
            self.stats.edits = len(edits)

        return TranscriptionResult(verbatim=verbatim, corrected=corrected, edits=edits, flags=flags)

    def stream(self, source: AudioSource, *, hotwords: Iterable[str] = ()) -> Iterator[Event]:
        """Transcribe while audio is still arriving.

        Yields provisional text as it stabilizes, then an authoritative result
        once the recording ends:

        * :class:`PartialUtterance` -- unstable hypothesis, will be superseded.
        * :class:`FinalUtterance` -- confirmed by the streaming policy; the text
          is stable from that point on.
        * :class:`TranscriptComplete` -- terminal, carrying the final-tier result.

        The two tiers exist because a recognizer revises its output as more
        context arrives. Live text is therefore explicitly provisional, and the
        final pass re-transcribes the complete audio with full context -- it is
        materially more accurate, so it is the one to keep.
        """
        policy = LocalAgreement(self.config.confirm_after)
        engine = self.live_asr
        detector = self.vad
        language = self.config.live.language
        skipped = 0

        buffer = np.zeros(0, dtype=np.float32)
        buffer_start = 0.0
        full: list[npt.NDArray[np.float32]] = []

        for chunk in source.chunks():
            processed = self.preprocessor.process(chunk)
            if len(processed.pcm):
                buffer = np.concatenate([buffer, processed.pcm])
                full.append(processed.pcm)

            if len(buffer) == 0:
                if chunk.is_last:
                    break
                continue

            window = AudioChunk(pcm=buffer, sample_rate=TARGET_SAMPLE_RATE, start=buffer_start)

            # Skip the decode when the buffer holds no speech. Only safe while
            # nothing is pending: text awaiting confirmation needs a further
            # decode to be confirmed, and silence is exactly what follows the
            # last word of an utterance -- gating unconditionally would strand
            # that text until the stream ended.
            if (
                detector is not None
                and not chunk.is_last
                and not policy.pending
                and not detector.has_speech(window)
            ):
                skipped += 1
                continue

            hypothesis = engine.transcribe(window, language=language, hotwords=hotwords)
            confirmed, pending = policy.update(hypothesis)

            for utterance in confirmed:
                yield FinalUtterance(utterance=utterance)
            for utterance in pending:
                yield PartialUtterance(utterance=utterance)

            # Scroll the buffer past confirmed text. Without this, decode cost
            # grows with the length of the appointment and a long procedure
            # eventually stops keeping up with real time.
            trim_to = policy.committed_until
            if trim_to > buffer_start:
                drop = int((trim_to - buffer_start) * TARGET_SAMPLE_RATE)
                if 0 < drop <= len(buffer):
                    buffer = buffer[drop:]
                    buffer_start = trim_to

            if chunk.is_last:
                break

        for utterance in policy.flush():
            yield FinalUtterance(utterance=utterance)

        self.stats.vad_skipped_blocks = skipped

        # -- final tier ----------------------------------------------------- #
        audio = np.concatenate(full) if full else np.zeros(0, dtype=np.float32)
        meta = source.meta()
        if meta.duration is None:
            meta = meta.model_copy(update={"duration": len(audio) / TARGET_SAMPLE_RATE})

        result = self._run_final(
            AudioChunk(pcm=audio, sample_rate=TARGET_SAMPLE_RATE, is_last=True),
            meta,
            hotwords=hotwords,
        )

        yield TranscriptComplete(result=result)

    def close(self) -> None:
        for stage in (self._asr, self._live_asr, self._vad, self._corrector):
            closer = getattr(stage, "close", None)
            if callable(closer):
                closer()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def transcribe_file(
    path: str | Path,
    config: Config | None = None,
    *,
    live: bool = False,
    hotwords: Iterable[str] = (),
) -> TranscriptionResult:
    """Transcribe a file.

    ``live=True`` reads a WAV that is still being written, returning once the
    recording completes.
    """
    config = config or Config()
    source = open_source(path, live=live, chunk_seconds=config.chunk_seconds)
    with Pipeline(config) as pipeline:
        return pipeline.transcribe(source, hotwords=hotwords)


def stream_file(
    path: str | Path,
    config: Config | None = None,
    *,
    live: bool = True,
    hotwords: Iterable[str] = (),
) -> Iterator[Event]:
    """Stream events while transcribing a file.

    ``live=True`` (the default here) reads a WAV that is still being written,
    which is the case this entry point exists for.
    """
    config = config or Config()
    source = open_source(path, live=live, chunk_seconds=config.chunk_seconds)
    with Pipeline(config) as pipeline:
        yield from pipeline.stream(source, hotwords=hotwords)
