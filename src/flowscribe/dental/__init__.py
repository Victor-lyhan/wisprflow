"""Dental domain layer: lexicon, tooth notation, and terminology normalization.

Licensing note: this package never bundles CDT procedure codes or SNODENT. Both
are American Dental Association copyright and require a paid commercial license
to redistribute, even though a practice may use CDT freely in its own records. A
licensed practice supplies its own file via ``CorrectionConfig.user_codes_path``.
The bundled lexicon is built only from freely redistributable sources.
"""

from .lexicon import Confusions, Lexicon, load_confusions, load_seed_lexicon
from .review import flag_confusions
from .teeth import extract_tooth_numbers

__all__ = [
    "Lexicon",
    "Confusions",
    "load_seed_lexicon",
    "load_confusions",
    "extract_tooth_numbers",
    "flag_confusions",
]
