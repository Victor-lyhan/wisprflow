"""Review flagging.

Marks spans a clinician should verify, without changing them. Runs as a pure
lexicon lookup, so it costs nothing and works with the LLM corrector disabled
entirely -- which matters, because this addresses a failure mode the corrector
structurally cannot.

The correction stage repairs words that are *wrong*: "buckle" is not a dental
term, so a candidate list and a model can fix it. Flagging handles words that are
*right but possibly the wrong right word*: "reversible pulpitis" is correctly
spelled, in the lexicon, and names a real diagnosis -- just not necessarily the
one that was said. No amount of string analysis finds that, and asking an LLM to
adjudicate risks it inverting a diagnosis on inference. So the system points and
a human decides.
"""

from __future__ import annotations

from ..contracts import Flag, Transcript
from ..normalize import normalize_text
from .lexicon import Confusions

__all__ = ["flag_confusions"]


def flag_confusions(transcript: Transcript, confusions: Confusions) -> list[Flag]:
    """Flag every clinically confusable term in the transcript."""
    flags: list[Flag] = []

    for utterance in transcript.utterances:
        text = utterance.text.strip()
        if not text:
            continue

        tokens = normalize_text(text, numbers_to_digits=False).split()
        index = 0
        while index < len(tokens):
            for length in range(min(3, len(tokens) - index), 0, -1):
                phrase = " ".join(tokens[index : index + length])
                alternatives = confusions.counterparts(phrase)
                if alternatives:
                    flags.append(
                        Flag(
                            utterance_id=utterance.id,
                            term=phrase,
                            alternatives=alternatives,
                        )
                    )
                    index += length
                    break
            else:
                index += 1

    return flags
