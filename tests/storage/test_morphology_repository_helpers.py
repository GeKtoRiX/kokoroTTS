import pytest

from kokoro_tts.storage.morphology_repository import (
    ExpressionRow,
    MorphRow,
    MorphologyRepository,
    _build_key,
    _build_pos_table_from_entries,
    _normalize_prefix,
    _stringify_cell,
    _to_int,
    _unique_expression_rows,
    _unique_lexeme_rows,
    _unique_occurrence_rows,
)


def test_helper_normalization_and_casting():
    assert _normalize_prefix("!!!") == "morph_"
    assert _normalize_prefix("abc-123") == "abc123"
    assert _to_int("10", 0) == 10
    assert _to_int("bad", 5) == 5
    assert _build_key(" Lemma ", " NOUN ") == "lemma|noun"
    assert _stringify_cell(None) == ""
    assert _stringify_cell(42) == "42"


def test_build_pos_table_from_entries_handles_duplicates_and_extra_tags():
    headers, rows = _build_pos_table_from_entries(
        [
            ("NOUN", "cat"),
            ("NOUN", "cat"),
            ("VERB", "run"),
            ("ZZZ", "rare"),
        ]
    )
    assert "Noun" in headers
    assert "Verb" in headers
    assert "ZZZ" in headers
    assert rows


def test_unique_row_helpers_deduplicate():
    row = MorphRow(
        source="s",
        part_index=0,
        segment_index=0,
        token_index=0,
        voice="af_heart",
        token="Cats",
        lemma="cat",
        upos="NOUN",
        feats_json="{}",
        start=0,
        end=4,
        key="cat|noun",
        text_sha1="abc",
    )
    expr = ExpressionRow(
        source="s",
        part_index=0,
        segment_index=0,
        expression_index=0,
        voice="af_heart",
        expression_text="look up",
        expression_lemma="look up",
        expression_type="phrasal_verb",
        start=0,
        end=7,
        expression_key="look up|phrasal_verb",
        match_source="dep",
        wordnet=1,
        text_sha1="abc",
    )

    assert len(_unique_lexeme_rows([row, row])) == 1
    assert len(_unique_occurrence_rows([row, row])) == 1
    assert len(_unique_expression_rows([expr, expr])) == 1


def test_serialize_feats_json_avoids_unnecessary_json_work():
    repository = MorphologyRepository(enabled=False, db_path=":memory:")
    assert repository._serialize_feats_json(None) == "{}"
    assert repository._serialize_feats_json({}) == "{}"
    assert repository._serialize_feats_json({"Number": "Sing"}) == '{"Number": "Sing"}'


def test_validate_table_name_rejects_unknown_table():
    repository = MorphologyRepository(enabled=False, db_path=":memory:")
    with pytest.raises(ValueError, match="Unsupported table"):
        repository._validate_table_name("malicious_table")
