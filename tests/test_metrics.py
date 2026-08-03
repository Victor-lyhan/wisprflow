"""Evaluation metrics."""

from __future__ import annotations

import pytest

from dentascribe.dental import Lexicon
from dentascribe.evaluation import (
    domain_word_error_rate,
    insertion_rate,
    score,
    tooth_accuracy,
    word_error_rate,
)
from dentascribe.evaluation.metrics import ScoreCard
from dentascribe.evaluation.runner import aggregate


@pytest.fixture
def lexicon() -> Lexicon:
    return Lexicon(["mesial", "buccal", "occlusal", "distal", "composite", "lidocaine"])


class TestWordErrorRate:
    def test_identical_is_zero(self) -> None:
        assert word_error_rate("tooth number three", "tooth number three") == 0.0

    def test_one_substitution_in_three(self) -> None:
        assert word_error_rate("tooth number three", "tooth number four") == pytest.approx(1 / 3)

    def test_ignores_case_and_punctuation(self) -> None:
        assert word_error_rate("Tooth number three.", "tooth NUMBER three") == 0.0

    def test_number_word_and_digit_agree(self) -> None:
        assert word_error_rate("tooth number three", "tooth number 3") == 0.0

    def test_empty_reference(self) -> None:
        assert word_error_rate("", "") == 0.0
        assert word_error_rate("", "spurious") == 1.0


class TestDomainWordErrorRate:
    def test_perfect_recognition(self, lexicon: Lexicon) -> None:
        rate, counts = domain_word_error_rate(
            "the mesial buccal surface", "the mesial buccal surface", lexicon
        )
        assert rate == 0.0
        assert counts["hits"] == 2

    def test_substituted_domain_term_counts(self, lexicon: Lexicon) -> None:
        """The motivating case: "buccal" misrecognized as "buckle"."""
        rate, counts = domain_word_error_rate(
            "depths on the buccal", "depths on the buckle", lexicon
        )
        assert rate == pytest.approx(1.0)
        assert counts["substitutions"] == 1

    def test_ignores_errors_on_non_domain_words(self, lexicon: Lexicon) -> None:
        """This is the whole point of the metric: general-word errors must not
        dilute the domain signal."""
        rate, _ = domain_word_error_rate(
            "the patient has a mesial lesion", "uh patient hasa mesial lesion", lexicon
        )
        assert rate == 0.0

    def test_domain_wer_exceeds_overall_when_terms_fail(self, lexicon: Lexicon) -> None:
        ref = "the patient reports pain on the buccal surface today"
        hyp = "the patient reports pain on the buckle surface today"
        card = score(ref, hyp, lexicon)
        assert card.domain_wer > card.wer

    def test_no_domain_terms_gives_zero(self, lexicon: Lexicon) -> None:
        rate, counts = domain_word_error_rate("the patient arrived", "the patient left", lexicon)
        assert rate == 0.0
        assert counts["domain_terms"] == 0

    def test_counts_inserted_domain_terms(self, lexicon: Lexicon) -> None:
        """A corrector inventing dental terminology shows up here and nowhere
        else, since an insertion has no reference position."""
        _, counts = domain_word_error_rate(
            "the mesial surface", "the mesial buccal surface", lexicon
        )
        assert counts["insertions"] == 1

    def test_empty_hypothesis_is_total_loss(self, lexicon: Lexicon) -> None:
        rate, counts = domain_word_error_rate("the mesial buccal", "", lexicon)
        assert rate == 1.0
        assert counts["deletions"] == 2


class TestToothAccuracy:
    def test_all_recovered(self) -> None:
        acc, total, hits = tooth_accuracy("tooth 3 and tooth 14", "tooth 3 and tooth 14")
        assert (acc, total, hits) == (1.0, 2, 2)

    def test_partial_recovery(self) -> None:
        acc, total, hits = tooth_accuracy("tooth 3 and tooth 14", "tooth 3 and tooth 15")
        assert acc == 0.5
        assert (total, hits) == (2, 1)

    def test_number_word_matches_digit(self) -> None:
        acc, _, _ = tooth_accuracy("tooth number three", "tooth number 3")
        assert acc == 1.0

    def test_multiset_semantics(self) -> None:
        """Naming tooth 3 twice must not be satisfied by naming it once."""
        acc, total, hits = tooth_accuracy("tooth 3 and tooth 3", "tooth 3")
        assert (total, hits) == (2, 1)
        assert acc == 0.5

    def test_no_teeth_in_reference(self) -> None:
        assert tooth_accuracy("the patient arrived", "the patient arrived") == (0.0, 0, 0)


class TestInsertionRate:
    def test_unchanged_text_is_zero(self) -> None:
        assert insertion_rate("tooth three composite", "tooth three composite") == 0.0

    def test_pure_substitution_is_low(self) -> None:
        """A constrained corrector swaps a wrong word for a right one. One new
        token out of four is expected; wholesale rewriting is not."""
        rate = insertion_rate("depths on the buckle", "depths on the buccal")
        assert rate == pytest.approx(0.25)

    def test_hallucinated_content_is_high(self) -> None:
        rate = insertion_rate(
            "tooth three composite",
            "tooth three composite with extensive periapical pathology noted",
        )
        assert rate > 0.5

    def test_reordering_is_not_insertion(self) -> None:
        assert insertion_rate("mesial occlusal distal", "distal occlusal mesial") == 0.0

    def test_empty_after(self) -> None:
        assert insertion_rate("something", "") == 0.0


class TestAggregate:
    def test_micro_averages_by_reference_length(self) -> None:
        """A 5-word sample must not weigh the same as a 100-word one."""
        short = ScoreCard(
            wer=1.0, reference_words=5, errors={"substitutions": 5, "deletions": 0, "insertions": 0}
        )
        long = ScoreCard(
            wer=0.0,
            reference_words=95,
            errors={"substitutions": 0, "deletions": 0, "insertions": 0},
        )
        total = aggregate([short, long])
        assert total.wer == pytest.approx(0.05)  # 5/100, not the 0.5 a macro mean would give

    def test_pools_domain_counts(self) -> None:
        cards = [
            ScoreCard(domain_terms=4, domain_hits=3, domain_substitutions=1, reference_words=10),
            ScoreCard(domain_terms=6, domain_hits=6, reference_words=10),
        ]
        total = aggregate(cards)
        assert total.domain_terms == 10
        assert total.domain_hits == 9
        assert total.domain_wer == pytest.approx(0.1)

    def test_empty_input(self) -> None:
        total = aggregate([])
        assert total.wer == 0.0
        assert total.domain_terms == 0


class TestScoreCard:
    def test_domain_recall_complements_domain_wer(self) -> None:
        card = ScoreCard(domain_terms=10, domain_hits=8, domain_substitutions=2, domain_wer=0.2)
        assert card.domain_recall == pytest.approx(0.8)
        assert card.domain_recall == pytest.approx(1 - card.domain_wer)

    def test_summary_is_renderable(self) -> None:
        assert "WER" in ScoreCard(wer=0.1, domain_wer=0.2).summary()
