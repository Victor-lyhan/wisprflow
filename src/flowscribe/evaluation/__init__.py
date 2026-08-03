"""Evaluation harness.

Requires ``pip install 'flowscribe[eval]'``.
"""

from ..normalize import normalize_text, tokenize, words_to_digits
from .dataset import Sample, iter_manifest, load_manifest
from .metrics import (
    ScoreCard,
    domain_word_error_rate,
    insertion_rate,
    score,
    tooth_accuracy,
    word_error_rate,
)
from .runner import EvalResult, SampleResult, aggregate, evaluate

__all__ = [
    "Sample",
    "load_manifest",
    "iter_manifest",
    "ScoreCard",
    "score",
    "word_error_rate",
    "domain_word_error_rate",
    "tooth_accuracy",
    "insertion_rate",
    "normalize_text",
    "tokenize",
    "words_to_digits",
    "evaluate",
    "aggregate",
    "EvalResult",
    "SampleResult",
]
