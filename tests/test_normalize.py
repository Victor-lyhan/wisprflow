"""Text normalization.

Normalization defines what counts as an error, so these are effectively tests of
the metric itself. Two cases here are regressions for bugs that each produced
badly wrong numbers before being caught.
"""

from __future__ import annotations

import pytest

from flowscribe.normalize import normalize_text, tokenize, words_to_digits


class TestWordsToDigits:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("three", "3"),
            ("nineteen", "19"),
            ("thirty", "30"),
            ("twenty-eight", "28"),
            ("twenty eight", "28"),
            ("zero", "0"),
            ("tooth number fourteen", "tooth number 14"),
        ],
    )
    def test_maps_number_words(self, text: str, expected: str) -> None:
        assert words_to_digits(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("nineteen.", "19."),
            ("tooth number nineteen.", "tooth number 19."),
            ("three, two, three", "3, 2, 3"),
            ("(fourteen)", "(14)"),
        ],
    )
    def test_handles_adjacent_punctuation(self, text: str, expected: str) -> None:
        """Regression: number words at a sentence boundary silently failed to map.

        The token was "nineteen." including the period, which matched no entry in
        the lookup table. Tooth extraction then missed every tooth reference that
        ended a sentence, and tooth accuracy was computed over a short reference
        list -- inflating the score.
        """
        assert words_to_digits(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("one hundred thousand", "100000"),
            ("one to one hundred thousand", "1 to 100000"),
            ("100,000", "100000"),
            ("two hundred thousand", "200000"),
            ("fifty thousand", "50000"),
            ("two hundred", "200"),
        ],
    )
    def test_compound_numerals_with_multipliers(self, text: str, expected: str) -> None:
        """Regression: "one hundred thousand" did not equal "100,000".

        Anaesthetic concentrations are dictated both ways. Treating them as
        different inflated overall WER from 6.98% to 20% on a single sample --
        i.e. most of the apparent error was a normalization artifact.
        """
        assert words_to_digits(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["three two three", "four three four", "two one two", "three, two, three"],
    )
    def test_charted_sequences_stay_separate(self, text: str) -> None:
        """Perio depths must never merge into one number.

        This is the constraint that rules out general compound-numeral parsing:
        "three two three" is three separate probing depths, and collapsing it to
        323 would destroy the reading. Compounding is therefore gated on an
        explicit multiplier word, which a charted run never has.
        """
        result = words_to_digits(text)
        digits = [t for t in result.replace(",", " ").split() if t.isdigit()]
        assert len(digits) == 3
        assert all(len(d) == 1 for d in digits)


class TestNormalizeText:
    def test_lowercases_and_strips_punctuation(self) -> None:
        assert normalize_text("Tooth #3, MOD composite!") == "tooth 3 mod composite"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("a   b \n c") == "a b c"

    def test_comma_separated_numbers_match_spaced(self) -> None:
        assert normalize_text("3, 2, 3") == normalize_text("three two three")

    def test_number_conversion_can_be_disabled(self) -> None:
        assert normalize_text("three", numbers_to_digits=False) == "three"

    def test_empty_input(self) -> None:
        assert normalize_text("") == ""
        assert tokenize("") == []

    def test_tokenize_splits_normalized_text(self) -> None:
        assert tokenize("Tooth number Three.") == ["tooth", "number", "3"]
