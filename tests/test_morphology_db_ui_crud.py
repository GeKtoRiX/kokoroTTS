import pytest

from kokoro_tts.ui.common import (
    build_morph_update_payload,
    extract_morph_headers,
    normalize_morph_dataset,
    resolve_morph_delete_confirmation,
)


def test_build_update_payload_for_occurrences_excludes_id_and_created_at():
    headers = [
        "id",
        "source",
        "part_index",
        "segment_index",
        "token_index",
        "voice",
        "token_text",
        "lemma",
        "upos",
        "feats_json",
        "start_offset",
        "end_offset",
        "dedup_key",
        "text_sha1",
        "created_at",
    ]
    row = [
        "15",
        "manual",
        "1",
        "2",
        "3",
        "af_heart",
        "Cats",
        "cat",
        "NOUN",
        "{}",
        "0",
        "4",
        "cat|noun",
        "abc123",
        "2026-02-15 00:00:00",
    ]

    selected_row_id, payload = build_morph_update_payload("occurrence", headers, row)

    assert selected_row_id == "15"
    assert "id" not in payload
    assert "created_at" not in payload
    assert payload["token_text"] == "Cats"
    assert payload["lemma"] == "cat"
    assert payload["part_index"] == 1
    assert payload["segment_index"] == 2
    assert payload["token_index"] == 3
    assert payload["start_offset"] == 0
    assert payload["end_offset"] == 4


def test_build_update_payload_for_lexemes_uses_dedup_key_and_skips_it_in_payload():
    headers = ["dedup_key", "lemma", "upos", "feats_json", "created_at"]
    row = ["run|verb", "run", "VERB", "{}", "2026-02-15 00:00:00"]

    selected_row_id, payload = build_morph_update_payload("lexeme", headers, row)

    assert selected_row_id == "run|verb"
    assert "dedup_key" not in payload
    assert payload == {"lemma": "run", "upos": "VERB", "feats_json": "{}"}


def test_build_update_payload_for_expressions_coerces_wordnet_hit_safely():
    headers = [
        "id",
        "expression_text",
        "expression_lemma",
        "expression_type",
        "part_index",
        "segment_index",
        "expression_index",
        "start_offset",
        "end_offset",
        "expression_key",
        "match_source",
        "wordnet_hit",
        "text_sha1",
        "created_at",
    ]
    row = [
        "7",
        "look up",
        "look up",
        "phrasal_verb",
        "0",
        "1",
        "2",
        "5",
        "12",
        "look up|phrasal_verb",
        "dependency_matcher",
        "0",
        "abc",
        "2026-02-15 00:00:00",
    ]

    selected_row_id, payload = build_morph_update_payload("expressions", headers, row)

    assert selected_row_id == "7"
    assert payload["wordnet_hit"] == 0
    assert payload["part_index"] == 0
    assert payload["segment_index"] == 1
    assert payload["expression_index"] == 2


def test_build_update_payload_rejects_invalid_selection_data():
    headers = ["id", "token_text", "created_at"]
    with pytest.raises(ValueError, match="No row selected"):
        build_morph_update_payload("occurrences", headers, None)

    with pytest.raises(ValueError, match="Primary key column 'dedup_key' is missing"):
        build_morph_update_payload("lexemes", headers, ["1", "Cats", "2026-02-15 00:00:00"])


def test_delete_confirmation_state_requires_second_click_and_resets_on_selection_change():
    should_delete, armed, message = resolve_morph_delete_confirmation("15", "")
    assert should_delete is False
    assert armed == "15"
    assert "again to confirm" in message

    should_delete, armed, message = resolve_morph_delete_confirmation("15", "15")
    assert should_delete is True
    assert armed == ""
    assert message == ""

    should_delete, armed, message = resolve_morph_delete_confirmation("16", "15")
    assert should_delete is False
    assert armed == "16"
    assert "again to confirm" in message


def test_extract_headers_from_table_update_and_missing_headers_fallback():
    update = {"value": [], "headers": ["id", "token_text"]}
    assert extract_morph_headers(update) == ["id", "token_text"]
    assert extract_morph_headers({"value": []}) == []


def test_normalize_dataset_aliases():
    assert normalize_morph_dataset("occurrence") == "occurrences"
    assert normalize_morph_dataset("lexeme") == "lexemes"
    assert normalize_morph_dataset("mwes") == "expressions"
    assert normalize_morph_dataset("review") == "reviews"
    assert normalize_morph_dataset("unknown") == "occurrences"
