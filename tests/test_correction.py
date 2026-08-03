"""Correction guards, prompting, and confusion flagging.

The guards are the safety-critical part: they are what stands between an LLM's
output and a patient record.
"""

from __future__ import annotations

import pytest

from flowscribe.contracts import Transcript, Utterance
from flowscribe.correct.guard import candidate_terms, check, classify
from flowscribe.correct.prompt import build_user_prompt, parse_response, strip_reasoning
from flowscribe.dental import Confusions, Lexicon, flag_confusions, load_confusions
from flowscribe.dental.lexicon import load_seed_lexicon


@pytest.fixture
def lexicon() -> Lexicon:
    return load_seed_lexicon()


@pytest.fixture
def confusions() -> Confusions:
    return load_confusions()


class TestGuard:
    def test_accepts_a_single_term_fix(self) -> None:
        assert check("depths on the buckle are 3 2 3", "depths on the buccal are 3 2 3")

    def test_accepts_identical_text(self) -> None:
        assert check("tooth number three", "tooth number three")

    def test_rejects_empty(self) -> None:
        assert not check("tooth number three", "")

    def test_rejects_wholesale_rewrite(self) -> None:
        """Paraphrasing is the failure mode, not spelling."""
        verdict = check(
            "pt c/o pain ur quad",
            "The patient complains of pain in the upper right quadrant.",
        )
        assert not verdict
        assert "edit ratio" in verdict.reason

    def test_rejects_invented_number(self) -> None:
        """A corrector adding a probing depth or tooth number is dangerous in a
        way a misspelt term is not."""
        verdict = check("probing depths are 3 2", "probing depths are 3 2 3")
        assert not verdict
        assert "numeric token count" in verdict.reason

    def test_rejects_dropped_number(self) -> None:
        verdict = check("probing depths are 3 2 3", "probing depths are 3 2")
        assert not verdict

    def test_allows_changing_a_number_value(self) -> None:
        """Correcting a misheard tooth number is the point; only the count is
        fixed."""
        assert check("tooth number 14 mesial", "tooth number 15 mesial")

    def test_rejects_added_tooth_reference(self) -> None:
        verdict = check(
            "the mesial surface shows caries",
            "the mesial surface of tooth 3 shows caries",
        )
        assert not verdict

    def test_edit_ratio_is_configurable(self) -> None:
        before, after = "one two three four", "one two three five"
        assert check(before, after, max_edit_ratio=0.5)
        assert not check(before, after, max_edit_ratio=0.01)


class TestClassify:
    def test_surface_term(self, lexicon: Lexicon) -> None:
        assert classify("on the buckle", "on the buccal", lexicon) == "surface"

    def test_drug_name(self, lexicon: Lexicon) -> None:
        assert classify("two carpules of lighter came", "two carpules of lidocaine", lexicon) == (
            "drug"
        )

    def test_tooth_number(self, lexicon: Lexicon) -> None:
        assert classify("tooth number 14", "tooth number 15", lexicon) == "tooth_number"

    def test_punctuation_only(self, lexicon: Lexicon) -> None:
        assert classify("tooth three", "Tooth three.", lexicon) == "punctuation"

    def test_non_domain_change(self, lexicon: Lexicon) -> None:
        assert classify("the patient waited", "the patient departed", lexicon) == "other"


class TestCandidateTerms:
    def test_surfaces_the_intended_term(self, lexicon: Lexicon) -> None:
        assert "buccal" in candidate_terms("Probing depths on the buckle are 3, 2, 3.", lexicon)

    def test_includes_component_words_of_phrases(self, lexicon: Lexicon) -> None:
        """Regression: "irreversible" exists in the lexicon only inside the phrase
        "irreversible pulpitis". Indexing whole phrases alone left a
        diagnosis-inverting misrecognition with nothing to correct toward."""
        assert "irreversible" in candidate_terms("presents with reversible pulpits", lexicon)

    def test_excludes_words_already_present(self, lexicon: Lexicon) -> None:
        assert "occlusal" not in candidate_terms("the occlusal surface", lexicon)

    def test_respects_limit(self, lexicon: Lexicon) -> None:
        text = "the buckle mesal distl occlsal composit restoraton pulpitus gingivitus"
        assert len(candidate_terms(text, lexicon, limit=5)) <= 5

    def test_chairside_speech_produces_little_noise(self, lexicon: Lexicon) -> None:
        """Conversational operatory speech should not fill the candidate list.

        Orthographic matching cannot be perfectly quiet -- some ordinary word
        will always resemble some clinical term -- so this asserts the noise is
        bounded rather than absent. Common chairside instructions are stopworded
        because they dominate operatory audio and are never themselves the
        misrecognition: "open" was matching "openbite", "wider" matching "oxide".
        """
        assert len(candidate_terms("okay please open a little wider", lexicon)) <= 2


class TestPromptParsing:
    def test_parses_plain_json(self) -> None:
        assert parse_response('{"corrected": "tooth number three"}') == "tooth number three"

    def test_strips_reasoning_block(self) -> None:
        """Reasoning models emit <think> before the answer. Left in place, a
        naive fallback would write the model's reasoning into the record."""
        raw = '<think>The word buckle is likely buccal.</think>{"corrected": "the buccal"}'
        assert parse_response(raw) == "the buccal"

    def test_strips_code_fence(self) -> None:
        assert parse_response('```json\n{"corrected": "tooth 3"}\n```') == "tooth 3"

    def test_returns_none_on_unparseable(self) -> None:
        assert parse_response("I think the answer is buccal.") is None

    def test_returns_none_on_missing_key(self) -> None:
        assert parse_response('{"result": "tooth 3"}') is None

    def test_returns_none_on_empty_correction(self) -> None:
        assert parse_response('{"corrected": "   "}') is None

    def test_strip_reasoning_is_case_insensitive(self) -> None:
        assert strip_reasoning("<THINK>x</THINK>hello") == "hello"


class TestPromptBuilding:
    def test_includes_the_utterance(self, lexicon: Lexicon) -> None:
        assert "buckle" in build_user_prompt("on the buckle", lexicon)

    def test_marks_context_as_uncorrectable(self, lexicon: Lexicon) -> None:
        prompt = build_user_prompt("and the distal", lexicon, context_before="Tooth 3 mesial.")
        assert "do NOT correct" in prompt
        assert "Tooth 3 mesial." in prompt

    def test_confusion_alternatives_are_separate_from_candidates(
        self, lexicon: Lexicon, confusions: Confusions
    ) -> None:
        prompt = build_user_prompt(
            "presents with reversible pulpitis", lexicon, confusions=confusions
        )
        assert "irreversible pulpitis" in prompt
        assert "only change it if" in prompt


class TestConfusions:
    def test_counterparts_are_symmetric(self, confusions: Confusions) -> None:
        assert "irreversible pulpitis" in confusions.counterparts("reversible pulpitis")
        assert "reversible pulpitis" in confusions.counterparts("irreversible pulpitis")

    def test_directional_opposites(self, confusions: Confusions) -> None:
        assert "distal" in confusions.counterparts("mesial")
        assert "lingual" in confusions.counterparts("buccal")

    def test_unknown_term_has_no_counterparts(self, confusions: Confusions) -> None:
        assert confusions.counterparts("stethoscope") == []

    def test_find_prefers_longest_phrase(self, confusions: Confusions) -> None:
        """ "reversible pulpitis" must resolve as the two-word diagnosis, not as
        the bare word "reversible"."""
        found = confusions.find("presents with reversible pulpitis today")
        assert "irreversible pulpitis" in found


class TestFlagging:
    def test_flags_confusable_diagnosis(self, confusions: Confusions) -> None:
        transcript = Transcript(
            utterances=[
                Utterance(
                    id="u0", start=0, end=1, text="Tooth 19 presents with reversible pulpitis"
                )
            ]
        )
        flags = flag_confusions(transcript, confusions)
        assert any("irreversible pulpitis" in f.alternatives for f in flags)

    def test_flags_reference_their_utterance(self, confusions: Confusions) -> None:
        transcript = Transcript(
            utterances=[
                Utterance(id="u0", start=0, end=1, text="nothing notable"),
                Utterance(id="u1", start=1, end=2, text="presents with reversible pulpitis"),
            ]
        )
        flags = flag_confusions(transcript, confusions)
        assert flags and all(f.utterance_id == "u1" for f in flags)

    def test_low_severity_terms_are_not_flagged(self, confusions: Confusions) -> None:
        """mesial/distal are genuinely confusable but appear in nearly every
        dental note. Flagging them every time buries the rare flag that matters,
        so they are recorded as low severity and left out of review."""
        transcript = Transcript(
            utterances=[Utterance(id="u0", start=0, end=1, text="caries on the mesial surface")]
        )
        assert flag_confusions(transcript, confusions) == []
        assert flag_confusions(transcript, confusions, severities=("high", "low"))

    def test_no_flags_for_general_speech(self, confusions: Confusions) -> None:
        transcript = Transcript(
            utterances=[Utterance(id="u0", start=0, end=1, text="please open a little wider")]
        )
        assert flag_confusions(transcript, confusions) == []

    def test_flagging_changes_nothing(self, confusions: Confusions) -> None:
        """Flags point; they never edit. Choosing between two valid clinical
        terms is a human judgement."""
        original = "reversible pulpitis noted"
        transcript = Transcript(utterances=[Utterance(id="u0", start=0, end=1, text=original)])
        flag_confusions(transcript, confusions)
        assert transcript.utterances[0].text == original
