"""Dental lexicon.

Serves two consumers with different needs:

* The **evaluation harness** asks which reference words are domain terms, to
  compute domain-WER. That is the metric that matters, because published dental
  ASR work found overall WER can look acceptable while clinical vocabulary
  fails -- an average dominated by ordinary words hides it.
* The **correction stage** asks for candidate terms to put in front of an LLM,
  and for a small hot-word set to bias the recognizer.

Multi-word terms ("root planing", "bleeding on probing") are indexed by length
so phrase matching can be greedy-longest, which keeps "root planing" from being
scored as the two unrelated unigrams "root" and "planing".
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from ..normalize import normalize_text

__all__ = [
    "Lexicon",
    "load_seed_lexicon",
    "load_confusions",
    "SEED_PATH",
    "BUILT_PATH",
    "CONFUSIONS_PATH",
]

SEED_PATH = Path(__file__).parent / "data" / "seed_lexicon.txt"
BUILT_PATH = Path(__file__).parent / "data" / "lexicon.txt"
CONFUSIONS_PATH = Path(__file__).parent / "data" / "confusions.txt"


def _parse(lines: Iterable[str]) -> list[str]:
    terms = []
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if text:
            terms.append(text)
    return terms


class Lexicon:
    """A set of domain terms with phrase-aware lookup."""

    def __init__(self, terms: Iterable[str]) -> None:
        self._terms: set[str] = set()
        self._by_length: dict[int, set[tuple[str, ...]]] = {}

        for term in terms:
            normalized = normalize_text(term, numbers_to_digits=False)
            if not normalized:
                continue
            self._terms.add(normalized)
            tokens = tuple(normalized.split())
            self._by_length.setdefault(len(tokens), set()).add(tokens)

        self._max_phrase = max(self._by_length, default=1)

    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, term: object) -> bool:
        if not isinstance(term, str):
            return False
        return normalize_text(term, numbers_to_digits=False) in self._terms

    @property
    def terms(self) -> frozenset[str]:
        return frozenset(self._terms)

    @classmethod
    def from_file(cls, path: str | Path) -> Lexicon:
        return cls(_parse(Path(path).read_text(encoding="utf-8").splitlines()))

    def extend(self, terms: Iterable[str]) -> Lexicon:
        """Return a new lexicon with additional terms.

        Used to fold in a practice-supplied procedure code list without mutating
        the shared bundled lexicon.
        """
        return Lexicon(list(self._terms) + list(terms))

    def domain_indices(self, tokens: list[str]) -> set[int]:
        """Indices of tokens that participate in a domain term.

        Matches longest-phrase-first and does not allow overlap, so each token is
        attributed to at most one term.
        """
        claimed: set[int] = set()
        i = 0
        while i < len(tokens):
            for length in range(min(self._max_phrase, len(tokens) - i), 0, -1):
                candidate = tuple(tokens[i : i + length])
                if candidate in self._by_length.get(length, ()):
                    claimed.update(range(i, i + length))
                    i += length
                    break
            else:
                i += 1
        return claimed

    def find(self, text: str) -> Iterator[str]:
        """Yield domain terms occurring in ``text``, in order."""
        tokens = normalize_text(text, numbers_to_digits=False).split()
        i = 0
        while i < len(tokens):
            for length in range(min(self._max_phrase, len(tokens) - i), 0, -1):
                candidate = tuple(tokens[i : i + length])
                if candidate in self._by_length.get(length, ()):
                    yield " ".join(candidate)
                    i += length
                    break
            else:
                i += 1

    def hotwords(self, limit: int = 40) -> list[str]:
        """A small biasing set for the recognizer.

        Intentionally truncated. Whisper's prompt channel reads only its last 224
        tokens and grows hallucination-prone when packed, so this cannot be the
        delivery mechanism for a lexicon of hundreds of terms -- that is the
        correction stage's job. Single words only, since phrase bias is unreliable.
        """
        singles = sorted(t for t in self._terms if " " not in t)
        return singles[:limit]


def load_seed_lexicon() -> Lexicon:
    """Load the bundled lexicon.

    Prefers the generated file from ``scripts/build_lexicon.py`` (MeSH + RxNorm
    merged with the seed list) and falls back to the seed list alone when it has
    not been generated.

    Caution when comparing runs: domain WER is computed over whichever terms this
    returns, so a larger lexicon changes the denominator. DWER figures are only
    comparable within one lexicon version.
    """
    return Lexicon.from_file(BUILT_PATH if BUILT_PATH.exists() else SEED_PATH)


class Confusions:
    """Clinically significant confusion sets.

    Handles the error class that similarity search structurally cannot: a
    misrecognition that lands on *another valid clinical term*. "irreversible
    pulpitis" heard as "reversible pulpitis" is correctly spelled, present in the
    lexicon, and names the opposite treatment decision. Nothing about the string
    looks wrong, so the only way to surface it is to know in advance which terms
    get confused with which.
    """

    def __init__(self, groups: Iterable[Iterable[str]]) -> None:
        self._counterparts: dict[str, list[str]] = {}
        self._max_phrase = 1

        for group in groups:
            members = [normalize_text(m, numbers_to_digits=False) for m in group]
            members = [m for m in members if m]
            if len(members) < 2:
                continue
            for member in members:
                others = [m for m in members if m != member]
                self._counterparts.setdefault(member, []).extend(others)
                self._max_phrase = max(self._max_phrase, len(member.split()))

    def __len__(self) -> int:
        return len(self._counterparts)

    def __contains__(self, term: object) -> bool:
        if not isinstance(term, str):
            return False
        return normalize_text(term, numbers_to_digits=False) in self._counterparts

    @classmethod
    def from_file(cls, path: str | Path) -> Confusions:
        groups = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            members = [part.strip() for part in text.split(",")]
            if len(members) >= 2:
                groups.append(members)
        return cls(groups)

    def counterparts(self, term: str) -> list[str]:
        """Terms confusable with ``term``."""
        key = normalize_text(term, numbers_to_digits=False)
        return list(dict.fromkeys(self._counterparts.get(key, [])))

    def find(self, text: str) -> list[str]:
        """Counterparts of every confusable term appearing in ``text``.

        Matches longest phrase first so "reversible pulpitis" resolves as the
        two-word diagnosis rather than the bare word "reversible".
        """
        tokens = normalize_text(text, numbers_to_digits=False).split()
        found: dict[str, None] = {}
        i = 0
        while i < len(tokens):
            for length in range(min(self._max_phrase, len(tokens) - i), 0, -1):
                phrase = " ".join(tokens[i : i + length])
                if phrase in self._counterparts:
                    for other in self.counterparts(phrase):
                        found.setdefault(other, None)
                    i += length
                    break
            else:
                i += 1
        return list(found)


def load_confusions() -> Confusions:
    """Load the bundled confusion sets."""
    return Confusions.from_file(CONFUSIONS_PATH)
