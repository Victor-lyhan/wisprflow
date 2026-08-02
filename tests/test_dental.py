"""Dental domain layer: lexicon and tooth notation."""

from __future__ import annotations

import pytest

from flowscribe.dental import Lexicon, extract_tooth_numbers, load_seed_lexicon
from flowscribe.normalize import tokenize


class TestSeedLexicon:
    def test_loads_terms(self) -> None:
        lexicon = load_seed_lexicon()
        assert len(lexicon) > 200

    @pytest.mark.parametrize(
        "term",
        ["mesial", "buccal", "occlusal", "lidocaine", "epinephrine", "gingivitis", "prophylaxis"],
    )
    def test_contains_core_terminology(self, term: str) -> None:
        assert term in load_seed_lexicon()

    def test_excludes_comments_and_blanks(self) -> None:
        lexicon = load_seed_lexicon()
        assert not any(t.startswith("#") for t in lexicon.terms)
        assert "" not in lexicon.terms

    def test_no_cdt_codes_bundled(self) -> None:
        """CDT is ADA copyright and must never ship in the package.

        A practice may use CDT freely in its own records, but redistributing the
        code set inside a product requires a paid licence. Codes are supplied by
        the practice at runtime instead.
        """
        lexicon = load_seed_lexicon()
        import re

        cdt_pattern = re.compile(r"^d\d{4}$")
        assert not [t for t in lexicon.terms if cdt_pattern.match(t)]


class TestLexiconMatching:
    def test_is_case_insensitive(self) -> None:
        assert "MESIAL" in Lexicon(["mesial"])

    def test_matches_multiword_phrases(self) -> None:
        lexicon = Lexicon(["root planing", "scaling"])
        assert list(lexicon.find("completed scaling and root planing today")) == [
            "scaling",
            "root planing",
        ]

    def test_prefers_longest_phrase(self) -> None:
        """ "root planing" must not be scored as two unrelated unigrams."""
        lexicon = Lexicon(["root", "planing", "root planing"])
        assert list(lexicon.find("root planing")) == ["root planing"]

    def test_domain_indices_cover_whole_phrase(self) -> None:
        lexicon = Lexicon(["bleeding on probing"])
        tokens = tokenize("noted bleeding on probing today")
        assert lexicon.domain_indices(tokens) == {1, 2, 3}

    def test_domain_indices_empty_for_no_match(self) -> None:
        assert Lexicon(["mesial"]).domain_indices(tokenize("the patient arrived")) == set()

    def test_extend_does_not_mutate_original(self) -> None:
        base = Lexicon(["mesial"])
        extended = base.extend(["custom code"])
        assert "custom code" in extended
        assert "custom code" not in base

    def test_hotwords_are_bounded_and_single_word(self) -> None:
        """Whisper's prompt reads only its final 224 tokens and degrades when
        stuffed, so the biasing set has to stay small."""
        hotwords = load_seed_lexicon().hotwords(limit=10)
        assert len(hotwords) == 10
        assert all(" " not in w for w in hotwords)


class TestToothExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("tooth number three", ["3"]),
            ("tooth #14", ["14"]),
            ("tooth 30", ["30"]),
            ("tooth no. 8", ["8"]),
            ("teeth number nineteen", ["19"]),
            ("tooth number nineteen.", ["19"]),
            ("tooth number three and tooth number fourteen", ["3", "14"]),
        ],
    )
    def test_extracts_universal_numbers(self, text: str, expected: list[str]) -> None:
        assert extract_tooth_numbers(text) == expected

    def test_extracts_primary_letters(self) -> None:
        assert extract_tooth_numbers("tooth K is mobile") == ["K"]

    @pytest.mark.parametrize("text", ["tooth 45", "tooth 0", "tooth 99"])
    def test_rejects_out_of_range(self, text: str) -> None:
        """Outside 1-32 is either a misrecognition or FDI notation. Guessing
        between them would silently name a different tooth."""
        assert extract_tooth_numbers(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "probing depths are three two three",
            "administered two carpules",
            "the patient waited three minutes",
        ],
    )
    def test_does_not_extract_bare_numbers(self, text: str) -> None:
        """Conservative by design: a bare number in dental speech is usually a
        depth or a count, and over-extraction would flatter the tooth metric."""
        assert extract_tooth_numbers(text) == []

    def test_preserves_order_and_duplicates(self) -> None:
        assert extract_tooth_numbers("tooth 3, tooth 14, tooth 3") == ["3", "14", "3"]
