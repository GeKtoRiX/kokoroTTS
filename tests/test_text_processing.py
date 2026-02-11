import os

os.environ.setdefault("KOKORO_SKIP_APP_INIT", "1")

import app


def test_normalize_times_basic():
    assert app.normalize_times("Meet me at 12:30.") == "Meet me at twelve thirty."


def test_normalize_times_skip_markdown():
    text = "[link](/12:30/) at 12:30"
    assert app.normalize_times(text) == "[link](/12:30/) at twelve thirty"


def test_normalize_numbers_percent_decimal_ordinal():
    text = "I got 42% on the 3rd try in 1.5 hours."
    expected = "I got forty two percent on the third try in one point five hours."
    assert app.normalize_numbers(text) == expected


def test_normalize_numbers_skip_slashed_and_multi_dot():
    text = "Version 1.2.3 and /123/ and 100."
    expected = "Version 1.2.3 and /123/ and one hundred."
    assert app.normalize_numbers(text) == expected


def test_split_sentences_abbrev():
    text = "Dr. Smith went home. Then he slept."
    assert app.split_sentences(text) == ["Dr. Smith went home.", "Then he slept."]


def test_smart_split_max_chars():
    text = "Hello world. Hi there."
    assert app.smart_split(text, max_chars=12, keep_sentences=False) == ["Hello world.", "Hi there."]


def test_split_parts():
    assert app.split_parts("a| b | |c|") == ["a", "b", "c"]


def test_normalize_voice_input_mix():
    assert app.normalize_voice_input(["af_heart", "af_bella"]) == "af_heart,af_bella"


def test_normalize_voice_tag_unknown():
    assert app.normalize_voice_tag("unknown", "af_heart") == "af_heart"


def test_limit_dialogue_parts_truncates():
    parts = [[("af_heart", "hello"), ("af_heart", "world")], [("af_heart", "bye")]]
    limited, truncated = app.limit_dialogue_parts(parts, char_limit=7)
    assert truncated is True
    assert limited == [[("af_heart", "hello"), ("af_heart", "wo")]]
