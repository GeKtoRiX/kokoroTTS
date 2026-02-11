"""Text normalization for times and numbers."""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from .text_utils import _apply_outside_spans, _find_skip_spans, _merge_spans

logger = logging.getLogger("kokoro_app")

TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*([AaPp])\.?\s*[Mm]\.?)?\b")
MULTI_DOT_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+){2,}\b")
ORDINAL_RE = re.compile(r"\b(\d{1,4})(st|nd|rd|th)\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?%")
DECIMAL_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)\.(\d+)\b")
INT_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)\b")
DIGIT_RE = re.compile(r"\d")

ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
TEENS = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}

ORDINAL_WORDS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "four": "fourth",
    "five": "fifth",
    "six": "sixth",
    "seven": "seventh",
    "eight": "eighth",
    "nine": "ninth",
    "ten": "tenth",
    "eleven": "eleventh",
    "twelve": "twelfth",
    "thirteen": "thirteenth",
    "fourteen": "fourteenth",
    "fifteen": "fifteenth",
    "sixteen": "sixteenth",
    "seventeen": "seventeenth",
    "eighteen": "eighteenth",
    "nineteen": "nineteenth",
    "twenty": "twentieth",
    "thirty": "thirtieth",
    "forty": "fortieth",
    "fifty": "fiftieth",
    "sixty": "sixtieth",
    "seventy": "seventieth",
    "eighty": "eightieth",
    "ninety": "ninetieth",
    "hundred": "hundredth",
    "thousand": "thousandth",
}


def number_to_words_0_59(value: int) -> str:
    if value < 10:
        return ONES[value]
    if value < 20:
        return TEENS[value - 10]
    tens = value // 10
    ones = value % 10
    tens_word = TENS[tens]
    return tens_word if ones == 0 else f"{tens_word} {ONES[ones]}"


def number_to_words_0_99(value: int) -> str:
    if value < 60:
        return number_to_words_0_59(value)
    tens = value // 10
    ones = value % 10
    tens_word = TENS[tens]
    return tens_word if ones == 0 else f"{tens_word} {ONES[ones]}"


def number_to_words_0_999(value: int) -> str:
    if value < 100:
        return number_to_words_0_99(value)
    hundreds = value // 100
    remainder = value % 100
    words = f"{ONES[hundreds]} hundred"
    if remainder:
        words = f"{words} {number_to_words_0_99(remainder)}"
    return words


@lru_cache(maxsize=10000)
def number_to_words_0_9999(value: int) -> str:
    if value < 1000:
        return number_to_words_0_999(value)
    thousands = value // 1000
    remainder = value % 1000
    words = f"{ONES[thousands]} thousand"
    if remainder:
        words = f"{words} {number_to_words_0_999(remainder)}"
    return words


def ordinalize_words(words: str) -> str:
    parts = words.split()
    if not parts:
        return words
    last = parts[-1]
    if last in ORDINAL_WORDS:
        parts[-1] = ORDINAL_WORDS[last]
    return " ".join(parts)


@lru_cache(maxsize=10000)
def ordinal_to_words(value: int) -> str:
    words = number_to_words_0_9999(value)
    return ordinalize_words(words)


@lru_cache(maxsize=2048)
def digits_to_words(digits: str) -> str:
    return " ".join(ONES[int(digit)] for digit in digits)


@lru_cache(maxsize=2048)
def time_to_words(hour: int, minute: int, ampm_letter: str | None) -> str:
    if ampm_letter:
        hour = hour % 12
        if hour == 0:
            hour = 12
    hour_word = number_to_words_0_59(hour)
    if minute == 0:
        if ampm_letter:
            suffix = "a.m." if ampm_letter.lower() == "a" else "p.m."
            return f"{hour_word} {suffix}"
        if hour == 0:
            return "midnight"
        if hour == 12:
            return "noon"
        return f"{hour_word} o'clock"
    if minute < 10:
        minute_word = f"oh {ONES[minute]}"
    else:
        minute_word = number_to_words_0_59(minute)
    if ampm_letter:
        suffix = "a.m." if ampm_letter.lower() == "a" else "p.m."
        return f"{hour_word} {minute_word} {suffix}"
    return f"{hour_word} {minute_word}"


def normalize_times(text: str) -> str:
    if ":" not in text or not DIGIT_RE.search(text):
        return text
    replacements = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal replacements
        hour = int(match.group(1))
        minute = int(match.group(2))
        ampm = match.group(3)
        replacements += 1
        return time_to_words(hour, minute, ampm)

    skip_spans = _find_skip_spans(text)
    if not skip_spans:
        updated = TIME_RE.sub(repl, text)
    else:
        parts: list[str] = []
        last = 0
        for start, end in skip_spans:
            if start > last:
                parts.append(TIME_RE.sub(repl, text[last:start]))
            parts.append(text[start:end])
            last = end
        if last < len(text):
            parts.append(TIME_RE.sub(repl, text[last:]))
        updated = "".join(parts)
    if replacements:
        logger.debug("Normalized times: %s", replacements)
    return updated


def normalize_numbers(text: str) -> str:
    if not DIGIT_RE.search(text):
        return text
    replacements = 0

    def int_to_words(value_str: str) -> str | None:
        try:
            value = int(value_str.replace(",", ""))
        except ValueError:
            return None
        if value > 9999:
            return None
        return number_to_words_0_9999(value)

    def repl_percent(match: re.Match[str]) -> str:
        nonlocal replacements
        int_part = match.group(1)
        dec_part = match.group(2)
        int_words = int_to_words(int_part)
        if int_words is None:
            return match.group(0)
        if dec_part:
            int_words = f"{int_words} point {digits_to_words(dec_part)}"
        replacements += 1
        return f"{int_words} percent"

    def repl_ordinal(match: re.Match[str]) -> str:
        nonlocal replacements
        value = int(match.group(1))
        if value > 9999:
            return match.group(0)
        replacements += 1
        return ordinal_to_words(value)

    def repl_decimal(match: re.Match[str]) -> str:
        nonlocal replacements
        int_part = match.group(1)
        dec_part = match.group(2)
        int_words = int_to_words(int_part)
        if int_words is None:
            return match.group(0)
        replacements += 1
        return f"{int_words} point {digits_to_words(dec_part)}"

    def repl_int(match: re.Match[str]) -> str:
        nonlocal replacements
        int_words = int_to_words(match.group(1))
        if int_words is None:
            return match.group(0)
        replacements += 1
        return int_words

    skip_spans = _find_skip_spans(text)
    if "." in text:
        skip_spans.extend(match.span() for match in MULTI_DOT_NUMBER_RE.finditer(text))
    skip_spans = _merge_spans(skip_spans)

    def apply_all(segment: str) -> str:
        segment = PERCENT_RE.sub(repl_percent, segment)
        segment = ORDINAL_RE.sub(repl_ordinal, segment)
        segment = DECIMAL_RE.sub(repl_decimal, segment)
        segment = INT_RE.sub(repl_int, segment)
        return segment

    updated = _apply_outside_spans(text, skip_spans, apply_all)
    if replacements:
        logger.debug("Normalized numbers: %s", replacements)
    return updated


class TextNormalizer:
    """Applies character limits and number/time normalization."""

    def __init__(self, char_limit: int | None, normalize_times: bool, normalize_numbers: bool) -> None:
        self.char_limit = char_limit
        self.normalize_times = normalize_times
        self.normalize_numbers = normalize_numbers

    def preprocess(
        self,
        text: str,
        normalize_times_enabled: bool | None = None,
        normalize_numbers_enabled: bool | None = None,
        apply_char_limit: bool = True,
    ) -> str:
        if apply_char_limit:
            text = text if self.char_limit is None else text.strip()[: self.char_limit]
        if normalize_times_enabled is None:
            normalize_times_enabled = self.normalize_times
        if normalize_numbers_enabled is None:
            normalize_numbers_enabled = self.normalize_numbers
        if normalize_times_enabled:
            text = normalize_times(text)
        if normalize_numbers_enabled:
            text = normalize_numbers(text)
        return text
