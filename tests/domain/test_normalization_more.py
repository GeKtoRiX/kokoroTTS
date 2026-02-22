from kokoro_tts.domain.normalization import (
    TextNormalizer,
    digit_sequence_to_words,
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
    year_to_words,
)


def test_number_and_ordinal_converters_cover_boundaries():
    assert number_to_words_0_59(0) == "zero"
    assert number_to_words_0_59(59) == "fifty nine"
    assert number_to_words_0_99(99) == "ninety nine"
    assert number_to_words_0_999(105) == "one hundred five"
    assert number_to_words_0_9999(9999) == "nine thousand nine hundred ninety nine"
    assert ordinalize_words("twenty one") == "twenty first"
    assert ordinal_to_words(12) == "twelfth"
    assert digits_to_words("507") == "five zero seven"
    assert digit_sequence_to_words("0123411") == "oh one two three four double one"
    assert digit_sequence_to_words("3333") == "quadruple three"
    assert year_to_words(1993) == "nineteen ninety three"
    assert year_to_words(2006) == "two thousand and six"
    assert year_to_words(2024) == "twenty twenty four"


def test_time_to_words_and_normalize_times_edge_cases():
    assert time_to_words(0, 0, None) == "midnight"
    assert time_to_words(12, 0, None) == "noon"
    assert time_to_words(8, 5, None) == "eight oh five"
    assert time_to_words(14, 30, "p") == "two thirty p.m."
    assert normalize_times("Meet at 12:00 and 09:05.") == "Meet at noon and nine oh five."
    assert normalize_times("No digits here") == "No digits here"


def test_normalize_numbers_edge_cases():
    assert normalize_numbers("10000 should stay 10000") == "10000 should stay 10000"
    assert normalize_numbers("1,234 is valid") == "one thousand two hundred thirty four is valid"
    assert (
        normalize_numbers("3.14 and 20% and 2nd")
        == "three point one four and twenty percent and second"
    )
    assert normalize_numbers("No digits") == "No digits"
    assert normalize_numbers("Call 0123411 now") == "Call 0123411 now"
    assert normalize_numbers("Phone 1234567") == "Phone 1234567"
    assert normalize_numbers("/0123411/ should stay") == "/0123411/ should stay"
    assert (
        normalize_numbers("Born in 1993 and moved in 2006")
        == "Born in one thousand nine hundred ninety three and moved in two thousand six"
    )
    assert (
        normalize_numbers("Born in [date]1993 and moved in [date=2006]")
        == "Born in nineteen ninety three and moved in two thousand and six"
    )
    assert (
        normalize_numbers("Call [tnumber]0123411 now")
        == "Call oh one two three four double one now"
    )
    assert normalize_numbers("Code [tnumber=1002]") == "Code one double oh two"


def test_text_normalizer_apply_char_limit_toggle():
    normalizer = TextNormalizer(char_limit=5, normalize_times=True, normalize_numbers=True)
    assert normalizer.preprocess(" 123456 ", apply_char_limit=True) == "12345"
    assert normalizer.preprocess("12:30", apply_char_limit=False) == "twelve thirty"
