import pytest

from kokoro_tts.domain.lexeme_checks import (
    analyze_and_validate_english_lexemes,
)


def test_analyze_and_validate_english_lexemes_accepts_valid_payload():
    def analyzer(_: str):
        return {
            "language": "en",
            "items": [
                {
                    "token": "Cats",
                    "lemma": "cat",
                    "upos": "NOUN",
                    "feats": {"Number": "Plur"},
                    "start": 0,
                    "end": 4,
                    "key": "cat|noun",
                }
            ],
        }

    payload = analyze_and_validate_english_lexemes("Cats run", analyzer=analyzer)
    assert payload["language"] == "en"
    assert payload["items"][0]["key"] == "cat|noun"


def test_analyze_and_validate_english_lexemes_rejects_invalid_offsets():
    def analyzer(_: str):
        return {
            "language": "en",
            "items": [
                {
                    "token": "Cats",
                    "lemma": "cat",
                    "upos": "NOUN",
                    "feats": {"Number": "Plur"},
                    "start": 0,
                    "end": 3,
                    "key": "cat|noun",
                }
            ],
        }

    with pytest.raises(ValueError, match="token/offset mismatch"):
        analyze_and_validate_english_lexemes("Cats run", analyzer=analyzer)


def test_analyze_and_validate_english_lexemes_rejects_invalid_key():
    def analyzer(_: str):
        return {
            "language": "en",
            "items": [
                {
                    "token": "run",
                    "lemma": "run",
                    "upos": "VERB",
                    "feats": {"VerbForm": "Inf"},
                    "start": 0,
                    "end": 3,
                    "key": "run|noun",
                }
            ],
        }

    with pytest.raises(ValueError, match="invalid key"):
        analyze_and_validate_english_lexemes("run", analyzer=analyzer)
