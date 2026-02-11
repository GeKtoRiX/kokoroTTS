"""Domain logic for text processing and voice handling."""

from .normalization import (
    TextNormalizer,
    digits_to_words,
    normalize_numbers,
    normalize_times,
    number_to_words_0_59,
    number_to_words_0_99,
    number_to_words_0_999,
    number_to_words_0_9999,
    ordinal_to_words,
    ordinalize_words,
    time_to_words,
)
from .splitting import smart_split, split_parts, split_sentences
from .voice import (
    CHOICES,
    limit_dialogue_parts,
    normalize_voice_input,
    normalize_voice_tag,
    parse_voice_segments,
    resolve_voice,
    summarize_voice,
)

__all__ = [
    "CHOICES",
    "TextNormalizer",
    "digits_to_words",
    "limit_dialogue_parts",
    "normalize_numbers",
    "normalize_times",
    "normalize_voice_input",
    "normalize_voice_tag",
    "number_to_words_0_59",
    "number_to_words_0_99",
    "number_to_words_0_999",
    "number_to_words_0_9999",
    "ordinal_to_words",
    "ordinalize_words",
    "parse_voice_segments",
    "resolve_voice",
    "smart_split",
    "split_parts",
    "split_sentences",
    "summarize_voice",
    "time_to_words",
]
