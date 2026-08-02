"""Pipeline assembly and execution.

Wires configured backends into a runnable pipeline. Stages are resolved lazily,
so a config with correction disabled never imports an LLM client.

Scope note: this module implements the **final tier** -- full-context
recognition over complete audio, which is the authoritative output. The live
tier (LocalAgreement-2 confirmation and online diarization) builds on these same
stages and lands in a later phase; it is not a separate pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    SpeakerRelabel,
    Transcript,
    TranscriptComplete,
    TranscriptionResult,
    Utterance,
    utterance_id,
)
from .dental.lexicon import load_confusions
from .dental.review import flag_confusions
from .errors import OfflineViolation
from .protocols import ASREngine, AudioSource, Corrector, Diarizer, Preprocessor
from .registry import ASR, CORRECTOR, DIARIZER
from .streaming.policy import LocalAgreement

__all__ = ["Pipeline", "RunStats", "transcribe_file", "stream_file"]


@dataclass
class RunStats:
    """Timing for one run. Reported by the eval harness."""

    audio_seconds: float = 0.0
    ingest_seconds: float = 0.0
    asr_seconds: float = 0.0
    diarize_seconds: float = 0.0
    correct_seconds: float = 0.0
    edits: int = 0

    @property
    def total_seconds(self) -> float:
        return self.ingest_seconds + self.asr_seconds + self.diarize_seconds + self.correct_seconds

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
        diarizer: Diarizer | None = None,
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
        self._diarizer = diarizer
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
    def diarizer(self) -> Diarizer | None:
        if not self.config.diarization.enabled:
            return None
        if self._diarizer is None:
            cfg = self.config.diarization
            self._diarizer = DIARIZER.create(
                cfg.backend,
                min_speakers=cfg.min_speakers,
                max_speakers=cfg.max_speakers,
                model_dir=str(self.config.model_dir),
                **cfg.options,
            )
        return self._diarizer

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
        pieces: list[np.ndarray] = []
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
        # backend produced; SpeakerRelabel events reference these.
        utterances = [
            u.model_copy(update={"id": utterance_id(i)}) for i, u in enumerate(utterances)
        ]

        diarizer = self.diarizer
        if diarizer is not None and utterances:
            t0 = time.perf_counter()
            utterances = diarizer.assign(
                utterances, audio=audio, num_speakers=self.config.diarization.num_speakers
            )
            self.stats.diarize_seconds = time.perf_counter() - t0

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
        * :class:`FinalUtterance` -- confirmed by the streaming policy. Text is
          stable; the speaker label may still be revised.
        * :class:`SpeakerRelabel` -- offline diarization disagreed with the live
          guess for a span already displayed.
        * :class:`TranscriptComplete` -- terminal, carrying the final-tier result.

        The two tiers exist because diarization needs the whole recording to
        cluster speakers, which cannot be reconciled with emitting text during
        capture. So live output is explicitly provisional, and the final pass
        re-transcribes the complete audio with full context.
        """
        policy = LocalAgreement(self.config.confirm_after)
        engine = self.live_asr
        language = self.config.live.language

        buffer = np.zeros(0, dtype=np.float32)
        buffer_start = 0.0
        full: list[np.ndarray] = []
        live_emitted: list[Utterance] = []

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
            hypothesis = engine.transcribe(window, language=language, hotwords=hotwords)
            confirmed, pending = policy.update(hypothesis)

            for utterance in confirmed:
                live_emitted.append(utterance)
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
            live_emitted.append(utterance)
            yield FinalUtterance(utterance=utterance)

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

        yield from _relabel_events(live_emitted, result.best)
        yield TranscriptComplete(result=result)

    def close(self) -> None:
        for stage in (self._asr, self._live_asr, self._diarizer, self._corrector):
            closer = getattr(stage, "close", None)
            if callable(closer):
                closer()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _relabel_events(live: list[Utterance], final: Transcript) -> Iterator[SpeakerRelabel]:
    """Reconcile live speaker guesses against the authoritative diarization.

    Live labels come from online clustering over partial audio and are routinely
    wrong early in a recording, before enough of each speaker has been heard.
    Rather than leaving a consumer displaying stale attribution -- which in an
    operatory means the patient's words shown as the dentist's -- each live
    span is matched to the final utterance it most overlaps in time, and a
    relabel is emitted where they disagree.
    """
    labelled = [u for u in final.utterances if u.speaker is not None]
    if not labelled:
        return

    for utterance in live:
        best: Utterance | None = None
        best_overlap = 0.0
        for candidate in labelled:
            overlap = min(utterance.end, candidate.end) - max(utterance.start, candidate.start)
            if overlap > best_overlap:
                best, best_overlap = candidate, overlap

        if best is not None and best.speaker != utterance.speaker:
            yield SpeakerRelabel(
                utterance_id=utterance.id,
                speaker=best.speaker,  # type: ignore[arg-type]
                role=best.role,
            )


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
