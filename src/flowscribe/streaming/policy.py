"""LocalAgreement-N confirmation.

The problem this solves: transcribing a growing audio buffer means re-decoding
overlapping audio, and Whisper-family models revise their output as more context
arrives. Emitting every hypothesis directly produces text that visibly rewrites
itself -- unusable for a clinician reading along, and worse, it can show a wrong
drug name or tooth number that later corrects itself.

LocalAgreement-N holds text back until N successive decodes agree on it. Agreed
text is stable and can be committed; the rest stays provisional. N=2 is the
published setting and the usual quality/latency knee: it costs roughly one decode
interval of delay and removes nearly all flicker.

Confirmation is word-level rather than utterance-level. Segment boundaries move
around between decodes as the model reconsiders sentence breaks, so comparing
whole utterances would find disagreement where the words in fact matched.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Utterance, Word, utterance_id

__all__ = ["LocalAgreement", "Token"]


@dataclass(slots=True)
class Token:
    """A word with timing, detached from any utterance grouping."""

    text: str
    start: float
    end: float
    confidence: float | None = None
    speaker: str | None = None
    language: str | None = None

    def key(self) -> str:
        """What equality is judged on.

        Case and trailing punctuation are ignored: models flip capitalization and
        commas freely between decodes as context shifts, and treating that as
        disagreement would stall confirmation indefinitely on stable words.
        """
        return self.text.strip().lower().strip(".,;:!?\"'")


def _to_tokens(utterances: list[Utterance]) -> list[Token]:
    """Flatten utterances to a word sequence.

    Backends without word-level timestamps get evenly interpolated timings.
    Those are approximate by construction, but the policy only needs them
    ordered, and an approximate timestamp is more useful than none -- the final
    tier re-derives exact timings anyway.
    """
    tokens: list[Token] = []
    for utterance in utterances:
        if utterance.words:
            tokens.extend(
                Token(
                    text=w.text,
                    start=w.start,
                    end=w.end,
                    confidence=w.confidence,
                    speaker=utterance.speaker,
                    language=utterance.language,
                )
                for w in utterance.words
                if w.text.strip()
            )
            continue

        pieces = utterance.text.split()
        if not pieces:
            continue
        span = max(utterance.end - utterance.start, 1e-6) / len(pieces)
        tokens.extend(
            Token(
                text=piece,
                start=utterance.start + i * span,
                end=utterance.start + (i + 1) * span,
                confidence=utterance.confidence,
                speaker=utterance.speaker,
                language=utterance.language,
            )
            for i, piece in enumerate(pieces)
        )
    return tokens


def _common_prefix(sequences: list[list[Token]]) -> int:
    """Length of the longest prefix on which every sequence agrees."""
    if not sequences:
        return 0
    limit = min(len(s) for s in sequences)
    length = 0
    while length < limit:
        key = sequences[0][length].key()
        if any(s[length].key() != key for s in sequences[1:]):
            break
        length += 1
    return length


class LocalAgreement:
    """Confirms text once ``n`` successive hypotheses agree on it."""

    def __init__(self, n: int = 2) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self._committed: list[Token] = []
        self._history: list[list[Token]] = []
        self._emitted = 0

    # -- state -------------------------------------------------------------- #

    @property
    def committed_until(self) -> float:
        """End time of the last confirmed word.

        The streaming loop trims its audio buffer here, which is what keeps
        decode cost bounded on a long appointment instead of growing with the
        recording.
        """
        return self._committed[-1].end if self._committed else 0.0

    @property
    def pending(self) -> bool:
        """Whether any hypothesis is still awaiting confirmation.

        The streaming loop checks this before skipping a silent block: unflushed
        text needs another decode to be confirmed, and silence is precisely what
        follows the last word of an utterance.
        """
        return bool(self._history and self._history[-1])

    def reset(self) -> None:
        self._committed = []
        self._history = []
        self._emitted = 0

    # -- StreamingPolicy protocol ------------------------------------------- #

    def update(self, hypothesis: list[Utterance]) -> tuple[list[Utterance], list[Utterance]]:
        """Feed a new hypothesis; return ``(newly_confirmed, still_pending)``."""
        tokens = _to_tokens(hypothesis)

        # The hypothesis covers only the untrimmed part of the buffer, so it is
        # already relative to what has been committed -- but a decode may still
        # repeat committed words when the buffer was not trimmed exactly at a
        # word boundary. Drop any leading run that matches the tail already held.
        fresh = self._drop_committed_prefix(tokens)

        self._history.append(fresh)
        if len(self._history) > self.n:
            self._history.pop(0)

        confirmed_count = _common_prefix(self._history) if len(self._history) >= self.n else 0

        newly = fresh[:confirmed_count]
        self._committed.extend(newly)
        # Confirmed text must not be re-confirmed on the next update.
        self._history = [seq[confirmed_count:] for seq in self._history]

        pending = fresh[confirmed_count:]
        return self._as_utterances(newly, final=True), self._as_utterances(pending, final=False)

    def flush(self) -> list[Utterance]:
        """Confirm whatever is still pending, at end of stream.

        Nothing further will arrive to agree with, so the most recent hypothesis
        is the best available answer. Its text is provisional in the sense that
        the final tier may re-transcribe it with full context.
        """
        remaining = self._history[-1] if self._history else []
        self._committed.extend(remaining)
        self._history = []
        return self._as_utterances(remaining, final=True)

    # -- helpers ------------------------------------------------------------- #

    def _drop_committed_prefix(self, tokens: list[Token]) -> list[Token]:
        if not self._committed or not tokens:
            return tokens
        tail = self._committed[-min(len(self._committed), 16) :]
        for overlap in range(min(len(tail), len(tokens)), 0, -1):
            if [t.key() for t in tail[-overlap:]] == [t.key() for t in tokens[:overlap]]:
                return tokens[overlap:]
        return tokens

    def _as_utterances(self, tokens: list[Token], *, final: bool) -> list[Utterance]:
        """Group a token run into one utterance.

        One utterance per emission rather than re-deriving sentence boundaries:
        the live tier's job is low-latency text, and the final tier produces the
        authoritative segmentation with full context.
        """
        if not tokens:
            return []

        confidences = [t.confidence for t in tokens if t.confidence is not None]
        utterance = Utterance(
            id=utterance_id(self._emitted),
            start=tokens[0].start,
            end=tokens[-1].end,
            text=" ".join(t.text.strip() for t in tokens),
            words=[
                Word(text=t.text.strip(), start=t.start, end=t.end, confidence=t.confidence)
                for t in tokens
            ],
            speaker=tokens[0].speaker,
            language=tokens[0].language,
            confidence=sum(confidences) / len(confidences) if confidences else None,
            is_final=final,
        )
        if final:
            self._emitted += 1
        return [utterance]
