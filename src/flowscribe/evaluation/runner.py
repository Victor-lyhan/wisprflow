"""Evaluation runner.

Scores a configured pipeline over a dataset. When correction is enabled it
scores the verbatim and corrected transcripts separately, because the question
the correction stage has to answer is not "is the output good" but "did
correction make it better", and only a paired comparison answers that.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..audio.source import FileAudioSource
from ..config import Config
from ..dental.lexicon import Lexicon, load_seed_lexicon
from .dataset import Sample
from .metrics import ScoreCard, insertion_rate, score

__all__ = ["EvalResult", "SampleResult", "evaluate", "aggregate"]


@dataclass
class SampleResult:
    """Per-sample outcome."""

    sample_id: str
    verbatim: ScoreCard
    corrected: ScoreCard | None = None
    hypothesis: str = ""
    corrected_text: str | None = None
    insertion_rate: float = 0.0
    audio_seconds: float = 0.0
    processing_seconds: float = 0.0
    edits: int = 0


@dataclass
class EvalResult:
    """Aggregate outcome for a run."""

    samples: list[SampleResult] = field(default_factory=list)
    verbatim: ScoreCard = field(default_factory=ScoreCard)
    corrected: ScoreCard | None = None
    total_audio_seconds: float = 0.0
    total_processing_seconds: float = 0.0
    mean_insertion_rate: float = 0.0

    @property
    def real_time_factor(self) -> float:
        if not self.total_audio_seconds:
            return 0.0
        return self.total_processing_seconds / self.total_audio_seconds

    def report(self) -> str:
        lines = [
            f"samples            {len(self.samples)}",
            f"audio              {self.total_audio_seconds:.1f}s",
            f"real-time factor   {self.real_time_factor:.2f}",
            "",
            f"verbatim   {self.verbatim.summary()}",
        ]
        if self.corrected is not None:
            lines.append(f"corrected  {self.corrected.summary()}")
            delta = self.verbatim.domain_wer - self.corrected.domain_wer
            verdict = "improved" if delta > 0 else "regressed" if delta < 0 else "no change"
            lines += [
                "",
                f"domain WER {verdict} by {abs(delta):.2%} with correction enabled",
                f"insertion rate     {self.mean_insertion_rate:.2%} "
                "(content added by the corrector; should be near zero)",
            ]
        return "\n".join(lines)


def aggregate(cards: Iterable[ScoreCard]) -> ScoreCard:
    """Pool per-sample counts into one score card.

    Micro-averaged: totals are summed and the rate computed once. A macro
    average over per-sample rates would let a five-word sample weigh as much as a
    five-minute one.
    """
    total = ScoreCard()
    ref_words = 0
    word_errors = 0
    cer_weighted = 0.0

    for card in cards:
        ref_words += card.reference_words
        word_errors += (
            card.errors.get("substitutions", 0)
            + card.errors.get("deletions", 0)
            + card.errors.get("insertions", 0)
        )
        cer_weighted += card.cer * card.reference_words

        total.domain_terms += card.domain_terms
        total.domain_hits += card.domain_hits
        total.domain_substitutions += card.domain_substitutions
        total.domain_deletions += card.domain_deletions
        total.domain_insertions += card.domain_insertions
        total.tooth_references += card.tooth_references
        total.tooth_hits += card.tooth_hits

    total.reference_words = ref_words
    total.wer = word_errors / ref_words if ref_words else 0.0
    total.cer = cer_weighted / ref_words if ref_words else 0.0
    total.domain_wer = (
        (total.domain_substitutions + total.domain_deletions) / total.domain_terms
        if total.domain_terms
        else 0.0
    )
    total.tooth_accuracy = (
        total.tooth_hits / total.tooth_references if total.tooth_references else 0.0
    )
    total.errors = {"word_errors": word_errors}
    return total


def evaluate(
    samples: list[Sample],
    config: Config | None = None,
    *,
    lexicon: Lexicon | None = None,
    progress: bool = False,
) -> EvalResult:
    """Run the pipeline over a dataset and score it."""
    from ..pipeline import Pipeline

    config = config or Config()
    lexicon = lexicon if lexicon is not None else load_seed_lexicon()
    result = EvalResult()

    with Pipeline(config) as pipeline:
        for index, sample in enumerate(samples, start=1):
            if progress:
                print(f"[{index}/{len(samples)}] {sample.id}", flush=True)

            source = FileAudioSource(sample.audio, chunk_seconds=config.chunk_seconds)
            transcription = pipeline.transcribe(source)
            stats = pipeline.stats

            verbatim_text = transcription.verbatim.text
            verbatim_card = score(sample.reference, verbatim_text, lexicon)

            corrected_card = None
            corrected_text = None
            ins = 0.0
            if transcription.corrected is not None:
                corrected_text = transcription.corrected.text
                corrected_card = score(sample.reference, corrected_text, lexicon)
                ins = insertion_rate(verbatim_text, corrected_text)

            result.samples.append(
                SampleResult(
                    sample_id=sample.id,
                    verbatim=verbatim_card,
                    corrected=corrected_card,
                    hypothesis=verbatim_text,
                    corrected_text=corrected_text,
                    insertion_rate=ins,
                    audio_seconds=stats.audio_seconds,
                    processing_seconds=stats.total_seconds,
                    edits=stats.edits,
                )
            )
            result.total_audio_seconds += stats.audio_seconds
            result.total_processing_seconds += stats.total_seconds

    result.verbatim = aggregate(s.verbatim for s in result.samples)
    if any(s.corrected is not None for s in result.samples):
        result.corrected = aggregate(s.corrected for s in result.samples if s.corrected)
        rates = [s.insertion_rate for s in result.samples if s.corrected is not None]
        result.mean_insertion_rate = sum(rates) / len(rates) if rates else 0.0

    return result
