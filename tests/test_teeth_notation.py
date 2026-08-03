"""Tooth notation conversion.

Three systems are in active clinical use and disagree in ways that name a
different tooth rather than failing loudly. These tests pin the mapping against
known dental anatomy.
"""

from __future__ import annotations

import pytest

from dentascribe.dental.teeth import (
    fdi_to_universal,
    palmer_to_universal,
    tooth_name,
    universal_to_fdi,
    universal_to_palmer,
)


class TestUniversalToFDI:
    @pytest.mark.parametrize(
        ("universal", "fdi", "described"),
        [
            (1, "18", "upper right third molar"),
            (3, "16", "upper right first molar"),
            (8, "11", "upper right central incisor"),
            (9, "21", "upper left central incisor"),
            (14, "26", "upper left first molar"),
            (16, "28", "upper left third molar"),
            (17, "38", "lower left third molar"),
            (24, "31", "lower left central incisor"),
            (25, "41", "lower right central incisor"),
            (30, "46", "lower right first molar"),
            (32, "48", "lower right third molar"),
        ],
    )
    def test_known_anatomy(self, universal: int, fdi: str, described: str) -> None:
        assert universal_to_fdi(universal) == fdi
        assert tooth_name(universal) == described

    def test_round_trips_for_all_permanent_teeth(self) -> None:
        for universal in range(1, 33):
            assert fdi_to_universal(universal_to_fdi(universal)) == str(universal)

    def test_rejects_out_of_range(self) -> None:
        for value in (0, 33, -1):
            with pytest.raises(ValueError):
                universal_to_fdi(value)


class TestPrimaryDentition:
    @pytest.mark.parametrize(
        ("letter", "fdi"),
        [
            ("A", "55"),
            ("E", "51"),
            ("F", "61"),
            ("J", "65"),
            ("K", "75"),
            ("O", "71"),
            ("P", "81"),
            ("T", "85"),
        ],
    )
    def test_known_primary_teeth(self, letter: str, fdi: str) -> None:
        assert universal_to_fdi(letter) == fdi

    def test_round_trips_for_all_primary_teeth(self) -> None:
        for letter in "ABCDEFGHIJKLMNOPQRST":
            assert fdi_to_universal(universal_to_fdi(letter)) == letter

    def test_rejects_invalid_primary_position(self) -> None:
        with pytest.raises(ValueError):
            fdi_to_universal("56")  # primary quadrants only run 1-5


class TestPalmer:
    @pytest.mark.parametrize(
        ("universal", "palmer"),
        [(1, "UR8"), (8, "UR1"), (9, "UL1"), (16, "UL8"), (17, "LL8"), (32, "LR8")],
    )
    def test_known_mappings(self, universal: int, palmer: str) -> None:
        assert universal_to_palmer(universal) == palmer

    def test_round_trips(self) -> None:
        for universal in range(1, 33):
            assert palmer_to_universal(universal_to_palmer(universal)) == str(universal)

    def test_is_case_insensitive(self) -> None:
        assert palmer_to_universal("ur6") == palmer_to_universal("UR6")

    def test_rejects_malformed(self) -> None:
        for bad in ("XX3", "UR", "UR9", "6"):
            with pytest.raises(ValueError):
                palmer_to_universal(bad)


class TestNotationCollision:
    def test_the_same_number_is_a_different_tooth(self) -> None:
        """The reason conversions are explicit and never inferred.

        "Tooth 18" is the upper right third molar in FDI and the lower left
        second molar in Universal -- opposite corners of the mouth. A pipeline
        that guessed the notation from context would silently record the wrong
        tooth, and nothing in the transcript would look wrong.
        """
        assert fdi_to_universal(18) == "1"
        assert universal_to_fdi(18) == "37"
        assert tooth_name(fdi_to_universal(18)) == "upper right third molar"
        assert tooth_name(18) == "lower left second molar"
