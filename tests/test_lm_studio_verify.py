import json

import pytest

from kokoro_tts.integrations import lm_studio as lm


def _request(tokens):
    return lm.PosVerifyRequest(
        segment_text="look up the word",
        tokens=tokens,
        locked_expressions=[{"text": "look up", "lemma": "look up", "kind": "phrasal_verb", "start": 0, "end": 7}],
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="test-model",
        timeout_seconds=30,
        temperature=0.0,
        max_tokens=512,
    )


def test_verify_pos_with_context_parses_valid_payload(monkeypatch):
    tokens = [
        {"token_index": 0, "token": "look", "lemma": "look", "upos": "VERB", "start": 0, "end": 4},
        {"token_index": 1, "token": "up", "lemma": "up", "upos": "ADP", "start": 5, "end": 7},
    ]
    payload = {
        "token_checks": [
            {"token_index": 0, "lemma": "look", "upos": "VERB", "feats": {"VerbForm": "Inf"}},
            {"token_index": 1, "lemma": "up", "upos": "ADP", "feats": {}},
        ],
        "new_expressions": [
            {"text": "look up", "lemma": "look up", "kind": "phrasal_verb", "start": 0, "end": 7}
        ],
    }

    monkeypatch.setattr(
        lm,
        "_post_chat_completion",
        lambda **_: {"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    result = lm.verify_pos_with_context(_request(tokens))

    assert [item["token_index"] for item in result["token_checks"]] == [0, 1]
    assert result["token_checks"][0]["upos"] == "VERB"
    assert result["new_expressions"][0]["kind"] == "phrasal_verb"


def test_verify_pos_with_context_rejects_non_json_response(monkeypatch):
    tokens = [
        {"token_index": 0, "token": "look", "lemma": "look", "upos": "VERB", "start": 0, "end": 4},
    ]
    monkeypatch.setattr(
        lm,
        "_post_chat_completion",
        lambda **_: {"choices": [{"message": {"content": "not-json"}}]},
    )

    with pytest.raises(lm.LmStudioError, match="not valid JSON"):
        lm.verify_pos_with_context(_request(tokens))


def test_verify_pos_with_context_rejects_invalid_token_checks(monkeypatch):
    tokens = [
        {"token_index": 0, "token": "look", "lemma": "look", "upos": "VERB", "start": 0, "end": 4},
        {"token_index": 1, "token": "up", "lemma": "up", "upos": "ADP", "start": 5, "end": 7},
    ]
    payload = {
        "token_checks": [
            {"token_index": 0, "lemma": "look", "upos": "BAD"},
            {"token_index": 1, "lemma": "up", "upos": "ADP"},
        ],
        "new_expressions": [],
    }
    monkeypatch.setattr(
        lm,
        "_post_chat_completion",
        lambda **_: {"choices": [{"message": {"content": json.dumps(payload)}}]},
    )

    with pytest.raises(lm.LmStudioError, match="upos is invalid"):
        lm.verify_pos_with_context(_request(tokens))


def test_verify_pos_with_context_accepts_text_new_expressions(monkeypatch):
    tokens = [
        {"token_index": 0, "token": "stand", "lemma": "stand", "upos": "VERB", "start": 0, "end": 5},
        {"token_index": 1, "token": "up", "lemma": "up", "upos": "ADP", "start": 6, "end": 8},
    ]
    payload = {
        "token_checks": [
            {"token_index": 0, "lemma": "stand", "upos": "VERB", "feats": {}},
            {"token_index": 1, "lemma": "up", "upos": "ADP", "feats": {}},
        ],
        "new_expressions": [
            "stand up for himself, rock the boat",
            {"text": "walk on eggshells"},
        ],
    }
    monkeypatch.setattr(
        lm,
        "_post_chat_completion",
        lambda **_: {"choices": [{"message": {"content": json.dumps(payload)}}]},
    )

    result = lm.verify_pos_with_context(_request(tokens))
    assert result["new_expressions"][0]["text"] == "stand up for himself, rock the boat"
    assert result["new_expressions"][1]["text"] == "walk on eggshells"


def test_verify_pos_with_context_parses_expression_table(monkeypatch):
    tokens = [
        {"token_index": 0, "token": "stood", "lemma": "stand", "upos": "VERB", "start": 0, "end": 5},
        {"token_index": 1, "token": "up", "lemma": "up", "upos": "ADP", "start": 6, "end": 8},
    ]
    payload = {
        "token_checks": [
            {"token_index": 0, "lemma": "stand", "upos": "VERB", "feats": {}},
            {"token_index": 1, "lemma": "up", "upos": "ADP", "feats": {}},
        ],
        "expression_table": (
            "| phrasal_verbs | idioms |\n"
            "|---|---|\n"
            "| stand up for himself | rock the boat |\n"
            "| carry on | |\n"
        ),
        "new_expressions": [],
    }
    monkeypatch.setattr(
        lm,
        "_post_chat_completion",
        lambda **_: {"choices": [{"message": {"content": json.dumps(payload)}}]},
    )

    result = lm.verify_pos_with_context(_request(tokens))
    assert {"text": "stand up for himself", "kind": "phrasal_verb"} in result["new_expressions"]
    assert {"text": "carry on", "kind": "phrasal_verb"} in result["new_expressions"]
    assert {"text": "rock the boat", "kind": "idiom"} in result["new_expressions"]


def test_post_chat_completion_disables_timeout_when_zero(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    captured = {"kwargs": None}

    def fake_urlopen(_request, **kwargs):
        captured["kwargs"] = kwargs
        return _Response()

    monkeypatch.setattr(lm.urllib.request, "urlopen", fake_urlopen)
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    response = lm._post_chat_completion(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        timeout_seconds=0,
        payload=payload,
    )

    assert captured["kwargs"] == {}
    assert response["choices"][0]["message"]["content"] == "ok"
