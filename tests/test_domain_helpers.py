from kokoro_tts.domain.splitting import _is_abbrev, _split_long_piece, smart_split, split_parts, split_sentences
from kokoro_tts.domain.style import normalize_style_preset, resolve_style_runtime
from kokoro_tts.domain.text_utils import (
    _apply_outside_spans,
    _find_skip_spans,
    _is_within_spans,
    _merge_spans,
)
from kokoro_tts.domain.voice import (
    DEFAULT_VOICE,
    default_voice_for_lang,
    get_voice_choices,
    limit_dialogue_parts,
    normalize_lang_code,
    normalize_voice_input,
    normalize_voice_tag,
    parse_voice_segments,
    resolve_voice,
    summarize_voice,
    voice_language,
)


def test_split_sentences_and_smart_split_cover_edge_cases():
    assert split_sentences("") == []
    assert _is_abbrev("e.g. example", 3) is True
    assert _is_abbrev("Dr. Smith", 2) is True
    assert split_sentences("No punctuation here") == ["No punctuation here"]
    assert split_sentences("[doc](/v1.2.3/) then stop. Next one.") == [
        "[doc](/v1.2.3/) then stop.",
        "Next one.",
    ]

    long_piece = "one two three four five six seven eight nine ten"
    chunks = _split_long_piece(long_piece, max_chars=10)
    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks)

    keep_chunks = smart_split("alpha beta gamma delta", max_chars=8, keep_sentences=True)
    assert keep_chunks
    assert smart_split("   ", max_chars=8) == []
    assert _split_long_piece("short", max_chars=10) == ["short"]
    assert _split_long_piece("alpha, beta, gamma, delta", max_chars=12)
    assert smart_split("one. two three four five six.", max_chars=10, keep_sentences=False)


def test_split_parts_handles_empty_pipe_input():
    assert split_parts("|||") == [""]
    assert split_parts("plain text") == ["plain text"]


def test_text_utils_span_helpers():
    text = "[link](/abc/) and /def/ tail"
    spans = _find_skip_spans(text)
    assert spans
    assert _is_within_spans(spans[0][0], spans)
    assert _is_within_spans(len(text) - 1, spans) is False

    merged = _merge_spans([(0, 2), (2, 5), (8, 10)])
    assert merged == [(0, 5), (8, 10)]

    transformed = _apply_outside_spans(
        "abc[keep]def",
        [(3, 9)],
        lambda part: part.upper(),
    )
    assert transformed == "ABC[keep]DEF"


def test_voice_helpers_cover_language_mix_and_tags():
    assert normalize_lang_code("en-us") == "a"
    assert normalize_lang_code("unknown", default="b") == "b"
    assert voice_language("", default="b") == "b"
    assert default_voice_for_lang("a").startswith("a")

    choices = get_voice_choices("unknown")
    assert choices

    fallback = normalize_voice_input("af_heart", voice_mix="unknown_voice")
    assert fallback == DEFAULT_VOICE

    mixed = normalize_voice_input("af_heart", voice_mix=["af_heart", "af_bella", "bf_emma"])
    assert mixed == "af_heart,af_bella"
    assert resolve_voice("af_heart", ["af_bella"], mix_enabled=True) == "af_bella"
    assert resolve_voice("af_heart", ["af_bella"], mix_enabled=False) == "af_heart"

    assert normalize_voice_tag("default", "af_bella") == "af_bella"
    assert normalize_voice_tag("af_heart+af_bella", "af_heart") == "af_heart,af_bella"

    parts = parse_voice_segments("[voice=af_heart]Hello [voice=am_michael]World", "af_bella")
    assert parts == [("af_heart", "Hello"), ("am_michael", "World")]
    assert summarize_voice([parts], "af_bella") == "multi"
    assert summarize_voice([], "af_bella") == "af_bella"


def test_limit_dialogue_parts_without_limit_returns_original():
    parts = [[("af_heart", "hello"), ("af_heart", "world")]]
    limited, truncated = limit_dialogue_parts(parts, char_limit=None)
    assert limited == parts
    assert truncated is False


def test_style_presets_normalize_and_adjust_runtime():
    assert normalize_style_preset("Narrator") == "narrator"
    assert normalize_style_preset("unknown") == "neutral"

    preset, speed, pause = resolve_style_runtime("energetic", speed=1.0, pause_seconds=1.0)
    assert preset == "energetic"
    assert round(speed, 2) == 1.12
    assert round(pause, 2) == 0.8
