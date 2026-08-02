"""Validation for LLM-proposed corrections.

An LLM asked to fix terminology will sometimes paraphrase, expand shorthand into
prose, or add plausible clinical detail that was never said. In a patient record
that is worse than the original error, because it reads as authoritative.

Every proposed correction passes through :func:`check` before being accepted.
Rejection is silent and safe: the verbatim text is kept. The guards are
deliberately blunt -- they cannot tell a good correction from a bad one, only a
*constrained edit* from a *rewrite*, which is the distinction that matters.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from ..dental.lexicon import Lexicon
from ..dental.teeth import extract_tooth_numbers
from ..normalize import normalize_text, tokenize

__all__ = ["GuardResult", "check", "classify", "candidate_terms"]

_NUMERIC = re.compile(r"^\d+$")

# Common words that generate spurious orthographic matches against clinical
# vocabulary ("patient" -> "papilla", "well" -> "wedge") and waste candidate
# slots. Not a general stopword list -- only words frequent in clinical
# dictation that are never themselves the misrecognition being repaired.
_STOPWORDS = frozenset(
    {
        "have",
        "has",
        "had",
        "with",
        "without",
        "that",
        "this",
        "these",
        "those",
        "there",
        "their",
        "they",
        "them",
        "then",
        "than",
        "from",
        "into",
        "onto",
        "will",
        "would",
        "should",
        "could",
        "been",
        "being",
        "were",
        "was",
        "patient",
        "please",
        "thank",
        "thanks",
        "okay",
        "right",
        "left",
        "well",
        "good",
        "some",
        "very",
        "just",
        "like",
        "what",
        "when",
        "where",
        "which",
        "about",
        "after",
        "before",
        "again",
        "also",
        "here",
        "over",
        "under",
        "your",
        "yours",
        "mine",
        "ours",
        "going",
        "doing",
        "does",
        "done",
        "little",
        "more",
        "most",
        "much",
        "many",
        "next",
        "last",
        "first",
        "today",
        "tomorrow",
        "yesterday",
        "week",
        "month",
        "year",
        "time",
        # Chairside instructions and small talk. These dominate operatory audio
        # and are never the misrecognition being repaired, but they match
        # clinical vocabulary orthographically ("open" -> "openbite",
        # "wider" -> "oxide") and fill the candidate list with noise.
        "open",
        "wider",
        "close",
        "closed",
        "rinse",
        "spit",
        "swallow",
        "breathe",
        "relax",
        "turn",
        "head",
        "chin",
        "tongue",
        "lips",
        "hurt",
        "hurts",
        "sore",
        "feel",
        "feeling",
        "comfortable",
        "ready",
        "almost",
        "finished",
        "great",
        "perfect",
        "sorry",
        "moment",
        "second",
        "minute",
    }
)

# Drug names get their own edit class: a wrong anaesthetic or antibiotic in a
# record is a different order of problem from a wrong surface abbreviation.
_DRUGS = frozenset(
    {
        "lidocaine",
        "articaine",
        "mepivacaine",
        "prilocaine",
        "bupivacaine",
        "epinephrine",
        "levonordefrin",
        "benzocaine",
        "chlorhexidine",
        "amoxicillin",
        "clindamycin",
        "penicillin",
        "metronidazole",
        "azithromycin",
        "ibuprofen",
        "acetaminophen",
    }
)


@dataclass
class GuardResult:
    """Outcome of validating one proposed correction."""

    accepted: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.accepted


def _numeric_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if _NUMERIC.match(t)]


def check(before: str, after: str, *, max_edit_ratio: float = 0.25) -> GuardResult:
    """Decide whether a proposed correction is a constrained edit.

    Three independent checks, any of which rejects:

    1. **Edit ratio** -- how much of the utterance changed. A corrector fixing
       terms touches a few words; one rewriting a quarter of them is
       paraphrasing.
    2. **Numeric tokens preserved** -- the count of numbers must not change.
       Tooth numbers and probing depths are the highest-stakes content in dental
       dictation, and a corrector that invents or drops one is dangerous in a way
       a misspelt term is not. Values may change (that is the point); the count
       may not.
    3. **Tooth-reference count preserved** -- same reasoning, one level up:
       correcting "tooth 14" to "tooth 15" is legitimate, adding a tooth
       reference that was never dictated is not.
    """
    if not after.strip():
        return GuardResult(False, "empty correction")

    before_tokens = tokenize(before)
    after_tokens = tokenize(after)

    if not before_tokens:
        return GuardResult(True)

    matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens, autojunk=False)
    ratio = 1.0 - matcher.ratio()
    if ratio > max_edit_ratio:
        return GuardResult(False, f"edit ratio {ratio:.0%} exceeds {max_edit_ratio:.0%}")

    before_nums, after_nums = _numeric_tokens(before), _numeric_tokens(after)
    if len(before_nums) != len(after_nums):
        return GuardResult(
            False,
            f"numeric token count changed ({len(before_nums)} -> {len(after_nums)})",
        )

    before_teeth, after_teeth = extract_tooth_numbers(before), extract_tooth_numbers(after)
    if len(before_teeth) != len(after_teeth):
        return GuardResult(
            False,
            f"tooth reference count changed ({len(before_teeth)} -> {len(after_teeth)})",
        )

    return GuardResult(True)


def classify(before: str, after: str, lexicon: Lexicon) -> str:
    """Label what kind of change a correction made, for the audit trail.

    Determined from what the edit *introduced*, since that is what a reviewer is
    being asked to accept.
    """
    if normalize_text(before) == normalize_text(after):
        return "punctuation"

    if extract_tooth_numbers(before) != extract_tooth_numbers(after):
        return "tooth_number"

    before_tokens, after_tokens = tokenize(before), tokenize(after)
    matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens, autojunk=False)
    introduced = [
        token
        for tag, _, _, j1, j2 in matcher.get_opcodes()
        if tag in ("replace", "insert")
        for token in after_tokens[j1:j2]
    ]

    if any(token in _DRUGS for token in introduced):
        return "drug"

    introduced_domain = lexicon.domain_indices(introduced)
    if introduced_domain:
        surfaces = {"mesial", "distal", "occlusal", "buccal", "lingual", "incisal", "facial"}
        if any(introduced[i] in surfaces for i in introduced_domain):
            return "surface"
        return "term"

    return "other"


def candidate_terms(
    text: str, lexicon: Lexicon, *, limit: int = 12, cutoff: float = 0.5
) -> list[str]:
    """Domain terms that plausibly match what was said, by orthographic similarity.

    This is what keeps the correction prompt small and specific rather than a
    dump of the whole lexicon: the model chooses among real candidates instead of
    recalling dental vocabulary unaided, which is both cheaper and markedly less
    prone to invention.

    Three properties matter, each learned from a miss:

    * **Component words of phrases are in the vocabulary.** "irreversible" only
      appears in the lexicon inside "irreversible pulpitis". Indexing whole
      phrases alone meant a misrecognized "reversible pulpitis" -- which inverts
      the diagnosis -- had no candidate to correct toward, despite the two words
      being a 0.91 match.
    * **Ranked by similarity, globally.** Slots are scarce, and insertion order
      let unrelated terms crowd out the actual candidate.
    * **Words already present are excluded.** Telling the model that "occlusal"
      is a dental term when it already transcribed "occlusal" correctly spends a
      slot to say nothing.
    * **Only non-domain words get candidates.** A word already in the lexicon was
      recognized fine and needs no suggestion -- and querying it actively hurts,
      because a near-miss between two real terms scores high ("restoration" vs
      "restorative", 0.90) and outranks the moderate-similarity match that is
      actually wanted ("mutual" vs "mesial", 0.50). Skipping known-good tokens
      removes the noise floor rather than trying to out-rank it.
    """
    vocabulary: set[str] = set()
    for term in lexicon.terms:
        parts = term.split()
        vocabulary.update(parts)
        if len(parts) > 1:
            vocabulary.add(term)

    tokens = tokenize(text)
    present = set(tokens)
    pool = sorted(vocabulary - present)
    known_good = lexicon.domain_indices(tokens)

    # Best matches for each suspicious word, kept separate rather than pooled.
    per_token: list[list[str]] = []
    for position, token in enumerate(tokens):
        if len(token) < 4 or _NUMERIC.match(token) or position in known_good or token in _STOPWORDS:
            continue
        matches = difflib.get_close_matches(token, pool, n=3, cutoff=cutoff)
        if matches:
            per_token.append(matches)

    # Round-robin rather than a global ranking. Score ordering across the whole
    # utterance lets one token monopolize every slot -- "tooth" matches "root" at
    # 0.67 and buries "mutual" -> "mesial" at 0.50, which is the one that
    # matters. Interleaving guarantees each suspicious word contributes its best
    # candidate before any word contributes a second.
    selected: dict[str, None] = {}
    for rank in range(3):
        for matches in per_token:
            if rank < len(matches):
                selected.setdefault(matches[rank], None)
            if len(selected) >= limit:
                return list(selected)

    return list(selected)[:limit]
