"""Accuracy metrics.

Overall WER is reported but is not the metric to steer on. Published dental ASR
evaluation found that systems scoring well overall were still significantly
worse on clinical vocabulary specifically -- an average dominated by ordinary
words conceals exactly the failures that matter in a clinical record. Domain WER
is therefore the primary number here, with tooth-number accuracy as a second
targeted check.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..dental.lexicon import Lexicon
from ..dental.teeth import extract_tooth_numbers
from ..normalize import tokenize

__all__ = ["ScoreCard", "score", "word_error_rate", "domain_word_error_rate", "insertion_rate"]


def _require_jiwer():
    try:
        import jiwer
    except ImportError as exc:  # pragma: no cover
        from ..errors import MissingDependency

        raise MissingDependency(
            "jiwer is required for evaluation. Install with: pip install 'flowscribe[eval]'"
        ) from exc
    return jiwer


@dataclass
class ScoreCard:
    """Metrics for one sample or one aggregated run."""

    wer: float = 0.0
    cer: float = 0.0
    domain_wer: float = 0.0
    domain_terms: int = 0
    domain_hits: int = 0
    domain_substitutions: int = 0
    domain_deletions: int = 0
    domain_insertions: int = 0
    tooth_accuracy: float = 0.0
    tooth_references: int = 0
    tooth_hits: int = 0
    reference_words: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def domain_recall(self) -> float:
        """Fraction of reference domain terms recognized correctly."""
        return self.domain_hits / self.domain_terms if self.domain_terms else 0.0

    def summary(self) -> str:
        return (
            f"WER {self.wer:6.2%} | DWER {self.domain_wer:6.2%} "
            f"({self.domain_hits}/{self.domain_terms} terms) | "
            f"tooth {self.tooth_accuracy:6.2%} ({self.tooth_hits}/{self.tooth_references})"
        )


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER over normalized text."""
    jiwer = _require_jiwer()
    ref, hyp = " ".join(tokenize(reference)), " ".join(tokenize(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def domain_word_error_rate(
    reference: str, hypothesis: str, lexicon: Lexicon
) -> tuple[float, dict[str, int]]:
    """WER restricted to reference positions occupied by domain terms.

    Definition: align reference and hypothesis, then count substitutions and
    deletions that land on a domain-term token, divided by the number of
    domain-term tokens in the reference. Equivalently ``1 - domain_recall``.

    Insertions are counted and reported but excluded from the rate, because an
    inserted token has no reference position to be attributed to. They are
    tracked separately because a corrector that invents dental terminology shows
    up there and nowhere else.
    """
    jiwer = _require_jiwer()

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    domain_idx = lexicon.domain_indices(ref_tokens)

    counts = {
        "domain_terms": len(domain_idx),
        "hits": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }
    if not domain_idx:
        return 0.0, counts
    if not hyp_tokens:
        counts["deletions"] = len(domain_idx)
        return 1.0, counts

    output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))

    for chunk in output.alignments[0]:
        if chunk.type == "insert":
            inserted = hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx]
            counts["insertions"] += len(lexicon.domain_indices(inserted))
            continue

        for ref_i in range(chunk.ref_start_idx, chunk.ref_end_idx):
            if ref_i not in domain_idx:
                continue
            if chunk.type == "equal":
                counts["hits"] += 1
            elif chunk.type == "substitute":
                counts["substitutions"] += 1
            elif chunk.type == "delete":
                counts["deletions"] += 1

    rate = (counts["substitutions"] + counts["deletions"]) / counts["domain_terms"]
    return rate, counts


def tooth_accuracy(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Fraction of reference tooth references recovered, in order.

    Compared as an ordered multiset rather than a set: a transcript naming teeth
    3 and 14 is not equivalent to one naming 14 and 3 twice, and charting order
    carries meaning.
    """
    ref_teeth = extract_tooth_numbers(reference)
    hyp_teeth = extract_tooth_numbers(hypothesis)
    if not ref_teeth:
        return 0.0, 0, 0

    remaining = Counter(hyp_teeth)
    hits = 0
    for tooth in ref_teeth:
        if remaining[tooth] > 0:
            remaining[tooth] -= 1
            hits += 1
    return hits / len(ref_teeth), len(ref_teeth), hits


def insertion_rate(before: str, after: str) -> float:
    """Fraction of tokens in ``after`` that were not present in ``before``.

    The hallucination guard for the correction stage. A constrained corrector
    substitutes wrong terms for right ones and should leave this near zero; a
    corrector that is paraphrasing, expanding abbreviations into prose, or
    inventing clinical content drives it up. This is the check that makes
    "the LLM must not add content" enforceable rather than aspirational.
    """
    before_tokens = Counter(tokenize(before))
    after_tokens = tokenize(after)
    if not after_tokens:
        return 0.0
    novel = sum((Counter(after_tokens) - before_tokens).values())
    return novel / len(after_tokens)


def score(reference: str, hypothesis: str, lexicon: Lexicon | None = None) -> ScoreCard:
    """Compute the full score card for one sample."""
    jiwer = _require_jiwer()
    lexicon = lexicon or Lexicon([])

    ref_norm = " ".join(tokenize(reference))
    hyp_norm = " ".join(tokenize(hypothesis))

    card = ScoreCard(reference_words=len(ref_norm.split()))
    if not ref_norm:
        return card

    output = jiwer.process_words(ref_norm, hyp_norm or " ")
    card.wer = float(output.wer)
    card.cer = float(jiwer.cer(ref_norm, hyp_norm or " "))
    card.errors = {
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
    }

    card.domain_wer, domain = domain_word_error_rate(reference, hypothesis, lexicon)
    card.domain_terms = domain["domain_terms"]
    card.domain_hits = domain["hits"]
    card.domain_substitutions = domain["substitutions"]
    card.domain_deletions = domain["deletions"]
    card.domain_insertions = domain["insertions"]

    card.tooth_accuracy, card.tooth_references, card.tooth_hits = tooth_accuracy(
        reference, hypothesis
    )
    return card
