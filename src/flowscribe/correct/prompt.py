"""Prompt construction for the correction stage.

The prompt is deliberately narrow. The model is not asked to improve, summarize,
or clinically interpret the transcript -- only to repair words the recognizer got
wrong. Every widening of that instruction increases the chance of invented
clinical content, which the guards then reject, which wastes a decode.

Candidate terms are supplied per utterance rather than shipping the whole
lexicon: given "buckle" the candidate list already contains "buccal", so the
model picks from real options instead of recalling dental vocabulary unaided.
"""

from __future__ import annotations

import re

from ..dental.lexicon import Confusions, Lexicon
from .guard import candidate_terms

__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "strip_reasoning", "parse_response"]

SYSTEM_PROMPT = """\
You correct speech-recognition errors in dental clinical transcripts.

You will be given one utterance transcribed from audio in a dental operatory. \
The recognizer often mishears dental terminology, tooth numbers, and drug names.

Rules:
1. Fix ONLY misrecognized words: dental terms, anatomy, materials, drug names, \
tooth numbers, and surface abbreviations.
2. Do NOT add clinical content. Do NOT expand abbreviations into prose. \
Do NOT paraphrase, reword, or improve style.
3. Do NOT change the number of numeric values. Probing depths like "3 2 3" are \
three separate measurements and must stay three separate values.
4. Keep filler words, false starts, and grammatical errors exactly as they are. \
This is a verbatim clinical record, not prose.
5. If nothing is misrecognized, return the text completely unchanged.

Reply with JSON only: {"corrected": "<the corrected utterance>"}\
"""

_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)
_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)


def build_user_prompt(
    text: str,
    lexicon: Lexicon,
    *,
    confusions: Confusions | None = None,
    context_before: str = "",
    context_after: str = "",
) -> str:
    """Build the per-utterance prompt.

    Neighbouring utterances are included as read-only context because dental
    dictation is highly elliptical -- "and the distal on that one" is only
    resolvable against what came before. They are explicitly marked not to be
    corrected, so their content cannot leak into the output.
    """
    parts: list[str] = []

    candidates = candidate_terms(text, lexicon)
    if candidates:
        parts.append(
            "Dental terms that sound similar to words in this utterance "
            "(use only if they fit what was said):\n" + ", ".join(candidates)
        )

    # Kept separate from the similarity candidates because the instruction is
    # different. A similarity candidate says "this word may be wrong"; a
    # confusion alternative says "this term is valid, and so is its opposite --
    # decide from context". These are the errors that read as correct.
    if confusions is not None:
        alternatives = confusions.find(text)
        if alternatives:
            parts.append(
                "This utterance contains terms that are commonly confused with "
                "clinically different ones. The transcript may already be right -- "
                "only change it if the surrounding context clearly indicates the "
                "alternative was said:\n" + ", ".join(alternatives)
            )

    if context_before or context_after:
        lines = ["Surrounding context (for reference only -- do NOT correct or return this):"]
        if context_before:
            lines.append(f"  previous: {context_before}")
        if context_after:
            lines.append(f"  next: {context_after}")
        parts.append("\n".join(lines))

    parts.append(f"Utterance to correct:\n{text}")
    return "\n\n".join(parts)


def strip_reasoning(raw: str) -> str:
    """Remove chain-of-thought blocks and code fences.

    Reasoning models (Qwen3 among them) emit ``<think>`` blocks ahead of the
    answer. Left in place they defeat JSON parsing, and any fallback that treated
    the whole response as the correction would write the model's reasoning into
    the patient record.
    """
    return _FENCE.sub("", _THINK.sub("", raw)).strip()


def parse_response(raw: str) -> str | None:
    """Extract the corrected text from a model response.

    Returns ``None`` when the response cannot be parsed. The caller keeps the
    verbatim text in that case -- guessing at a malformed response is exactly how
    reasoning traces end up in a clinical transcript.
    """
    import json

    cleaned = strip_reasoning(raw)
    if not cleaned:
        return None

    match = _JSON_OBJECT.search(cleaned)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    corrected = payload.get("corrected")
    if not isinstance(corrected, str) or not corrected.strip():
        return None

    return corrected.strip()
