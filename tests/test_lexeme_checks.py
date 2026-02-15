import pytest

from kokoro_tts.domain.lexeme_checks import (
    LmVerifySettings,
    analyze_and_validate_english_lexemes,
    load_lm_verify_settings_from_env,
    verify_english_lexemes_with_lm,
)
from kokoro_tts.integrations.lm_studio import PosVerifyRequest


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


def test_verify_english_lexemes_with_lm_reports_mismatches():
    analysis = {
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
            },
            {
                "token": "run",
                "lemma": "run",
                "upos": "VERB",
                "feats": {"VerbForm": "Inf"},
                "start": 5,
                "end": 8,
                "key": "run|verb",
            },
        ],
    }
    settings = LmVerifySettings(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="test-model",
        timeout_seconds=30,
        temperature=0.0,
        max_tokens=512,
    )

    def verifier(request: PosVerifyRequest):
        assert request.model == "test-model"
        return {
            "token_checks": [
                {"token_index": 0, "lemma": "cat", "upos": "NOUN", "feats": {}},
                {"token_index": 1, "lemma": "running", "upos": "NOUN", "feats": {}},
            ],
            "new_expressions": [],
        }

    summary = verify_english_lexemes_with_lm(
        "Cats run",
        analysis,
        settings=settings,
        verifier=verifier,
    )
    assert summary["token_count"] == 2
    assert summary["mismatch_count"] == 1
    assert summary["mismatches"][0]["token_index"] == 1


def test_verify_english_lexemes_with_lm_allows_zero_timeout():
    analysis = {
        "language": "en",
        "items": [
            {
                "token": "look",
                "lemma": "look",
                "upos": "VERB",
                "feats": {},
                "start": 0,
                "end": 4,
                "key": "look|verb",
            },
            {
                "token": "up",
                "lemma": "up",
                "upos": "ADP",
                "feats": {},
                "start": 5,
                "end": 7,
                "key": "up|adp",
            },
        ],
    }
    settings = LmVerifySettings(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="test-model",
        timeout_seconds=0,
        temperature=0.0,
        max_tokens=512,
    )

    captured = {"timeout": None}

    def verifier(request: PosVerifyRequest):
        captured["timeout"] = request.timeout_seconds
        return {
            "token_checks": [
                {"token_index": 0, "lemma": "look", "upos": "VERB", "feats": {}},
                {"token_index": 1, "lemma": "up", "upos": "ADP", "feats": {}},
            ],
            "new_expressions": [],
        }

    summary = verify_english_lexemes_with_lm(
        "look up",
        analysis,
        settings=settings,
        verifier=verifier,
    )
    assert summary["mismatch_count"] == 0
    assert captured["timeout"] == 0


def test_load_lm_verify_settings_from_env_uses_defaults(monkeypatch):
    monkeypatch.delenv("LM_VERIFY_BASE_URL", raising=False)
    monkeypatch.delenv("LM_VERIFY_API_KEY", raising=False)
    monkeypatch.delenv("LM_VERIFY_MODEL", raising=False)
    monkeypatch.delenv("LM_VERIFY_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LM_VERIFY_TEMPERATURE", raising=False)
    monkeypatch.delenv("LM_VERIFY_MAX_TOKENS", raising=False)

    settings = load_lm_verify_settings_from_env()
    assert settings.base_url == "http://127.0.0.1:1234/v1"
    assert settings.api_key == "lm-studio"
    assert settings.model == ""
    assert settings.timeout_seconds == 30
    assert settings.temperature == 0.0
    assert settings.max_tokens == 512
