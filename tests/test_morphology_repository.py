import logging
import sqlite3
import csv
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from kokoro_tts.storage.morphology_repository import MorphologyRepository


def test_morphology_repository_disabled_skips_db_creation(tmp_path: Path):
    db_path = tmp_path / "disabled.sqlite3"
    repo = MorphologyRepository(
        enabled=False,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {"language": "en", "items": []},
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "Cats run")]], source="generate_first")
    assert db_path.exists() is False


def test_morphology_repository_inserts_ignore_and_commits(tmp_path: Path):
    db_path = tmp_path / "morphology.sqlite3"

    def fake_analyzer(_: str):
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
                {
                    "token": "cats",
                    "lemma": "cat",
                    "upos": "NOUN",
                    "feats": {"Number": "Plur"},
                    "start": 9,
                    "end": 13,
                    "key": "cat|noun",
                },
            ],
        }

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=fake_analyzer,
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "Cats run cats")]], source="generate_first")
    repo.ingest_dialogue_parts([[("af_heart", "Cats run cats")]], source="generate_first")

    connection = sqlite3.connect(db_path)
    try:
        lexeme_count = connection.execute("SELECT COUNT(*) FROM morph_lexemes").fetchone()[0]
        occurrence_count = connection.execute(
            "SELECT COUNT(*) FROM morph_token_occurrences"
        ).fetchone()[0]
        lexeme_keys = {
            row[0]
            for row in connection.execute(
                "SELECT dedup_key FROM morph_lexemes ORDER BY dedup_key"
            ).fetchall()
        }
    finally:
        connection.close()

    assert lexeme_count == 2
    assert occurrence_count == 3
    assert lexeme_keys == {"cat|noun", "run|verb"}


def test_morphology_repository_skips_when_no_items(tmp_path: Path):
    db_path = tmp_path / "empty.sqlite3"
    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {"language": "en", "items": []},
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "!!!")]], source="generate_first")
    assert db_path.exists() is False


def test_morphology_repository_exports_lexemes_csv(tmp_path: Path):
    db_path = tmp_path / "export.sqlite3"
    export_dir = tmp_path / "exports"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
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
        },
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "Cats")]], source="generate_first")

    csv_path = repo.export_csv(dataset="lexemes", output_dir=str(export_dir))
    assert csv_path is not None
    assert Path(csv_path).is_file()
    date_dir = export_dir / datetime.now().strftime("%Y-%m-%d")
    assert Path(csv_path).parent == date_dir / "vocabulary"
    assert (date_dir / "records").is_dir()
    assert (date_dir / "vocabulary").is_dir()

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["dedup_key", "lemma", "upos", "feats_json", "created_at"]
    assert rows[1][0] == "cat|noun"


def test_morphology_repository_export_returns_none_for_empty_or_disabled(tmp_path: Path):
    repo = MorphologyRepository(
        enabled=False,
        db_path=str(tmp_path / "nope.sqlite3"),
        logger_instance=logging.getLogger("test"),
        expression_extractor=lambda _: [],
    )
    assert repo.export_csv(dataset="lexemes", output_dir=str(tmp_path)) is None
    assert repo.export_txt(dataset="lexemes", output_dir=str(tmp_path)) is None
    assert repo.export_word_table(dataset="lexemes", output_dir=str(tmp_path)) is None
    assert repo.export_spreadsheet(dataset="lexemes", output_dir=str(tmp_path)) is None


def test_morphology_repository_exports_pos_table_csv(tmp_path: Path):
    db_path = tmp_path / "pos.sqlite3"
    export_dir = tmp_path / "exports"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {
                    "token": "Good",
                    "lemma": "good",
                    "upos": "ADJ",
                    "feats": {"Degree": "Pos"},
                    "start": 0,
                    "end": 4,
                    "key": "good|adj",
                },
                {
                    "token": "morning",
                    "lemma": "morning",
                    "upos": "NOUN",
                    "feats": {"Number": "Sing"},
                    "start": 5,
                    "end": 12,
                    "key": "morning|noun",
                },
                {
                    "token": "run",
                    "lemma": "run",
                    "upos": "VERB",
                    "feats": {"VerbForm": "Inf"},
                    "start": 13,
                    "end": 16,
                    "key": "run|verb",
                },
            ],
        },
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "Good morning run")]], source="generate_first")

    csv_path = repo.export_csv(dataset="pos_table", output_dir=str(export_dir))
    assert csv_path is not None

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    headers = rows[0]
    noun_index = headers.index("Noun")
    verb_index = headers.index("Verb")
    adj_index = headers.index("Adjective")
    first_row = rows[1]
    assert first_row[noun_index] == "morning"
    assert first_row[verb_index] == "run"
    assert first_row[adj_index] == "good"


def test_morphology_repository_exports_lexemes_ods(tmp_path: Path):
    db_path = tmp_path / "export_ods.sqlite3"
    export_dir = tmp_path / "exports"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
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
        },
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "Cats")]], source="generate_first")

    ods_path = repo.export_spreadsheet(dataset="lexemes", output_dir=str(export_dir))
    assert ods_path is not None
    assert Path(ods_path).suffix == ".ods"
    assert Path(ods_path).is_file()
    date_dir = export_dir / datetime.now().strftime("%Y-%m-%d")
    assert Path(ods_path).parent == date_dir / "vocabulary"
    assert (date_dir / "records").is_dir()
    assert (date_dir / "vocabulary").is_dir()
    with zipfile.ZipFile(ods_path, "r") as archive:
        names = set(archive.namelist())
    assert "content.xml" in names
    assert "styles.xml" in names


def test_morphology_repository_exports_lexemes_txt(tmp_path: Path):
    db_path = tmp_path / "export_txt.sqlite3"
    export_dir = tmp_path / "exports"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
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
        },
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "Cats")]], source="generate_first")

    txt_path = repo.export_txt(dataset="lexemes", output_dir=str(export_dir))
    assert txt_path is not None
    assert Path(txt_path).suffix == ".txt"
    assert Path(txt_path).is_file()
    content = Path(txt_path).read_text(encoding="utf-8")
    assert "dedup_key" in content
    assert "cat|noun" in content


def test_morphology_repository_exports_lexemes_xlsx(tmp_path: Path):
    openpyxl_module = pytest.importorskip("openpyxl")
    _ = openpyxl_module

    db_path = tmp_path / "export_xlsx.sqlite3"
    export_dir = tmp_path / "exports"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
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
        },
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "Cats")]], source="generate_first")

    xlsx_path = repo.export_excel(dataset="lexemes", output_dir=str(export_dir))
    assert xlsx_path is not None
    assert Path(xlsx_path).suffix == ".xlsx"
    assert Path(xlsx_path).is_file()

    from openpyxl import load_workbook

    workbook = load_workbook(filename=xlsx_path)
    worksheet = workbook.active
    header_cells = [str(cell.value or "") for cell in worksheet[1]]
    value_cells = [str(cell.value or "") for cell in worksheet[2]]
    assert "dedup_key" in header_cells
    assert "cat|noun" in value_cells


def test_morphology_repository_exports_pos_table_ods(tmp_path: Path):
    db_path = tmp_path / "pos_ods.sqlite3"
    export_dir = tmp_path / "exports"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {
                    "token": "Good",
                    "lemma": "good",
                    "upos": "ADJ",
                    "feats": {"Degree": "Pos"},
                    "start": 0,
                    "end": 4,
                    "key": "good|adj",
                },
                {
                    "token": "morning",
                    "lemma": "morning",
                    "upos": "NOUN",
                    "feats": {"Number": "Sing"},
                    "start": 5,
                    "end": 12,
                    "key": "morning|noun",
                },
            ],
        },
        expression_extractor=lambda _: [],
    )
    repo.ingest_dialogue_parts([[("af_heart", "Good morning")]], source="generate_first")
    ods_path = repo.export_spreadsheet(dataset="pos_table", output_dir=str(export_dir))
    assert ods_path is not None
    assert Path(ods_path).suffix == ".ods"
    with zipfile.ZipFile(ods_path, "r") as archive:
        content = archive.read("content.xml").decode("utf-8")
    assert "Noun" in content
    assert "Adjective" in content
    assert "morning" in content
    assert "good" in content


def test_morphology_repository_inserts_expressions_table(tmp_path: Path):
    db_path = tmp_path / "expr.sqlite3"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
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
        },
        expression_extractor=lambda _: [
            {
                "text": "look up",
                "lemma": "look up",
                "kind": "phrasal_verb",
                "start": 0,
                "end": 7,
                "key": "look up|phrasal_verb",
                "source": "dependency_matcher",
                "wordnet": True,
            }
        ],
    )
    repo.ingest_dialogue_parts([[("af_heart", "look up")]], source="generate_first")
    repo.ingest_dialogue_parts([[("af_heart", "look up")]], source="generate_first")

    connection = sqlite3.connect(db_path)
    try:
        expression_count = connection.execute("SELECT COUNT(*) FROM morph_expressions").fetchone()[0]
        row = connection.execute(
            "SELECT expression_text, expression_lemma, expression_type, match_source, wordnet_hit "
            "FROM morph_expressions LIMIT 1"
        ).fetchone()
    finally:
        connection.close()

    assert expression_count == 1
    assert row[0] == "look up"
    assert row[1] == "look up"
    assert row[2] == "phrasal_verb"
    assert row[3] == "dependency_matcher"
    assert row[4] == 1


def test_morphology_repository_exports_expressions_ods(tmp_path: Path):
    db_path = tmp_path / "expr_ods.sqlite3"
    export_dir = tmp_path / "exports"
    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {
                    "token": "kicked",
                    "lemma": "kick",
                    "upos": "VERB",
                    "feats": {},
                    "start": 0,
                    "end": 6,
                    "key": "kick|verb",
                }
            ],
        },
        expression_extractor=lambda _: [
            {
                "text": "kick the bucket",
                "lemma": "kick the bucket",
                "kind": "idiom",
                "start": 0,
                "end": 15,
                "key": "kick the bucket|idiom",
                "source": "wordnet_phrase_matcher",
                "wordnet": True,
            }
        ],
    )
    repo.ingest_dialogue_parts([[("af_heart", "kicked the bucket")]], source="generate_first")
    ods_path = repo.export_spreadsheet(dataset="expressions", output_dir=str(export_dir))
    assert ods_path is not None
    assert Path(ods_path).suffix == ".ods"
    with zipfile.ZipFile(ods_path, "r") as archive:
        content = archive.read("content.xml").decode("utf-8")
    assert "kick the bucket" in content
    assert "idiom" in content


def test_morphology_repository_crud_operations(tmp_path: Path):
    db_path = tmp_path / "crud.sqlite3"
    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {"language": "en", "items": []},
        expression_extractor=lambda _: [],
    )

    occurrence_id = repo.insert_row(
        dataset="occurrences",
        payload={
            "source": "manual",
            "token_text": "Cats",
            "lemma": "cat",
            "upos": "NOUN",
        },
    )
    headers, rows = repo.list_rows(dataset="occurrences", limit=20, offset=0)
    assert "id" in headers
    assert rows
    token_index = headers.index("token_text")
    assert rows[0][token_index] == "Cats"

    updated = repo.update_row(
        dataset="occurrences",
        row_id=occurrence_id,
        payload={"token_text": "Dogs", "lemma": "dog"},
    )
    assert updated == 1
    headers, rows = repo.list_rows(dataset="occurrences", limit=20, offset=0)
    assert rows[0][headers.index("token_text")] == "Dogs"

    deleted = repo.delete_row(dataset="occurrences", row_id=occurrence_id)
    assert deleted == 1

    lexeme_key = repo.insert_row(
        dataset="lexemes",
        payload={"lemma": "run", "upos": "VERB", "feats_json": "{}"},
    )
    assert lexeme_key == "run|verb"
    lex_updated = repo.update_row(
        dataset="lexemes",
        row_id=lexeme_key,
        payload={"lemma": "running"},
    )
    assert lex_updated == 1
    lex_deleted = repo.delete_row(dataset="lexemes", row_id=lexeme_key)
    assert lex_deleted == 1


def test_morphology_repository_writes_reviews_on_verify_success(tmp_path: Path):
    db_path = tmp_path / "reviews_success.sqlite3"

    def analyzer(_: str):
        return {
            "language": "en",
            "items": [
                {"token": "look", "lemma": "look", "upos": "VERB", "feats": {}, "start": 0, "end": 4, "key": "look|verb"},
                {"token": "up", "lemma": "up", "upos": "ADP", "feats": {}, "start": 5, "end": 7, "key": "up|adp"},
            ],
        }

    def verifier(payload):
        _ = payload
        return {
            "token_checks": [
                {"token_index": 0, "lemma": "look", "upos": "VERB", "feats": {}},
                {"token_index": 1, "lemma": "up", "upos": "ADP", "feats": {}},
            ],
            "new_expressions": [],
        }

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=analyzer,
        expression_extractor=lambda _: [],
        lm_verifier=verifier,
        lm_verify_enabled=True,
        lm_verify_model="verify-model",
    )
    repo.ingest_dialogue_parts([[("af_heart", "look up")]], source="generate_first")
    repo.wait_for_pending_reviews(3.0)

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT status, is_match, model FROM morph_reviews ORDER BY token_index"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 2
    assert rows[0][0] == "success"
    assert rows[0][1] == 1
    assert rows[0][2] == "verify-model"


def test_morphology_repository_writes_review_mismatch(tmp_path: Path):
    db_path = tmp_path / "reviews_mismatch.sqlite3"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {"token": "runs", "lemma": "run", "upos": "VERB", "feats": {}, "start": 0, "end": 4, "key": "run|verb"}
            ],
        },
        expression_extractor=lambda _: [],
        lm_verifier=lambda _payload: {
            "token_checks": [
                {"token_index": 0, "lemma": "running", "upos": "NOUN", "feats": {}}
            ],
            "new_expressions": [],
        },
        lm_verify_enabled=True,
        lm_verify_model="verify-model",
    )
    repo.ingest_dialogue_parts([[("af_heart", "runs")]], source="generate_first")
    repo.wait_for_pending_reviews(3.0)

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT is_match, mismatch_fields FROM morph_reviews LIMIT 1"
        ).fetchone()
    finally:
        connection.close()

    assert row[0] == 0
    assert "lemma" in row[1]
    assert "upos" in row[1]


def test_morphology_repository_review_failure_retries_and_does_not_crash(tmp_path: Path):
    db_path = tmp_path / "reviews_failure.sqlite3"
    attempts = {"count": 0}

    def failing_verifier(_payload):
        attempts["count"] += 1
        raise RuntimeError("offline")

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {"token": "test", "lemma": "test", "upos": "NOUN", "feats": {}, "start": 0, "end": 4, "key": "test|noun"}
            ],
        },
        expression_extractor=lambda _: [],
        lm_verifier=failing_verifier,
        lm_verify_enabled=True,
        lm_verify_retries=1,
    )
    repo.ingest_dialogue_parts([[("af_heart", "test")]], source="generate_first")
    repo.wait_for_pending_reviews(3.0)

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT status, attempt_count, error_text FROM morph_reviews LIMIT 1"
        ).fetchone()
    finally:
        connection.close()

    assert attempts["count"] == 2
    assert row[0] == "failed"
    assert row[1] == 2
    assert "offline" in row[2]


def test_morphology_repository_skips_non_english_segments_for_verify(tmp_path: Path):
    db_path = tmp_path / "reviews_non_en.sqlite3"
    called = {"count": 0}

    def verifier(_payload):
        called["count"] += 1
        return {"token_checks": [], "new_expressions": []}

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {"token": "Привет", "lemma": "Привет", "upos": "X", "feats": {}, "start": 0, "end": 6, "key": "привет|x"}
            ],
        },
        expression_extractor=lambda _: [],
        lm_verifier=verifier,
        lm_verify_enabled=True,
    )
    repo.ingest_dialogue_parts([[("af_heart", "Привет")]], source="generate_first")
    repo.wait_for_pending_reviews(3.0)

    connection = sqlite3.connect(db_path)
    try:
        count = connection.execute("SELECT COUNT(*) FROM morph_reviews").fetchone()[0]
    finally:
        connection.close()

    assert called["count"] == 0
    assert count == 0


def test_morphology_repository_auto_adds_valid_lm_expressions(tmp_path: Path):
    db_path = tmp_path / "reviews_expressions.sqlite3"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {"token": "look", "lemma": "look", "upos": "VERB", "feats": {}, "start": 0, "end": 4, "key": "look|verb"},
                {"token": "up", "lemma": "up", "upos": "ADP", "feats": {}, "start": 5, "end": 7, "key": "up|adp"},
                {"token": "now", "lemma": "now", "upos": "ADV", "feats": {}, "start": 8, "end": 11, "key": "now|adv"},
            ],
        },
        expression_extractor=lambda _: [],
        lm_verifier=lambda _payload: {
            "token_checks": [
                {"token_index": 0, "lemma": "look", "upos": "VERB", "feats": {}},
                {"token_index": 1, "lemma": "up", "upos": "ADP", "feats": {}},
                {"token_index": 2, "lemma": "now", "upos": "ADV", "feats": {}},
            ],
            "new_expressions": [
                {"text": "look up", "lemma": "look up", "kind": "phrasal_verb", "start": 0, "end": 7},
                {"text": "up now", "lemma": "up now", "kind": "idiom", "start": 5, "end": 11},
            ],
        },
        lm_verify_enabled=True,
    )
    repo.ingest_dialogue_parts([[("af_heart", "look up now")]], source="generate_first")
    repo.wait_for_pending_reviews(3.0)

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT expression_text, expression_type, match_source FROM morph_expressions ORDER BY start_offset"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 1
    assert rows[0][0].lower() == "look up"
    assert rows[0][1] == "phrasal_verb"
    assert rows[0][2] == "lm_verify_auto"


def test_morphology_repository_auto_adds_text_only_lm_expressions(tmp_path: Path):
    db_path = tmp_path / "reviews_expressions_text.sqlite3"
    text = "stand up for himself and rock the boat"

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {"token": "stand", "lemma": "stand", "upos": "VERB", "feats": {}, "start": 0, "end": 5, "key": "stand|verb"},
                {"token": "up", "lemma": "up", "upos": "ADP", "feats": {}, "start": 6, "end": 8, "key": "up|adp"},
                {"token": "for", "lemma": "for", "upos": "ADP", "feats": {}, "start": 9, "end": 12, "key": "for|adp"},
                {"token": "himself", "lemma": "himself", "upos": "PRON", "feats": {}, "start": 13, "end": 20, "key": "himself|pron"},
                {"token": "and", "lemma": "and", "upos": "CCONJ", "feats": {}, "start": 21, "end": 24, "key": "and|cconj"},
                {"token": "rock", "lemma": "rock", "upos": "VERB", "feats": {}, "start": 25, "end": 29, "key": "rock|verb"},
                {"token": "the", "lemma": "the", "upos": "DET", "feats": {}, "start": 30, "end": 33, "key": "the|det"},
                {"token": "boat", "lemma": "boat", "upos": "NOUN", "feats": {}, "start": 34, "end": 38, "key": "boat|noun"},
            ],
        },
        expression_extractor=lambda _: [],
        lm_verifier=lambda _payload: {
            "token_checks": [
                {"token_index": 0, "lemma": "stand", "upos": "VERB", "feats": {}},
                {"token_index": 1, "lemma": "up", "upos": "ADP", "feats": {}},
                {"token_index": 2, "lemma": "for", "upos": "ADP", "feats": {}},
                {"token_index": 3, "lemma": "himself", "upos": "PRON", "feats": {}},
                {"token_index": 4, "lemma": "and", "upos": "CCONJ", "feats": {}},
                {"token_index": 5, "lemma": "rock", "upos": "VERB", "feats": {}},
                {"token_index": 6, "lemma": "the", "upos": "DET", "feats": {}},
                {"token_index": 7, "lemma": "boat", "upos": "NOUN", "feats": {}},
            ],
            "new_expressions": ["stand up for himself, rock the boat"],
        },
        lm_verify_enabled=True,
    )
    repo.ingest_dialogue_parts([[("af_heart", text)]], source="generate_first")
    repo.wait_for_pending_reviews(3.0)

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT expression_text, expression_type, match_source FROM morph_expressions ORDER BY start_offset"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 2
    assert rows[0][0].lower() == "stand up for himself"
    assert rows[0][1] == "phrasal_verb"
    assert rows[0][2] == "lm_verify_auto"
    assert rows[1][0].lower() == "rock the boat"
    assert rows[1][1] == "idiom"
    assert rows[1][2] == "lm_verify_auto"


def test_morphology_repository_verifies_one_sentence_per_request(tmp_path: Path):
    db_path = tmp_path / "reviews_sentence_batches.sqlite3"
    text = "Jake stood up for himself. He refused to rock the boat."
    payloads: list[dict[str, object]] = []

    def verifier(payload: dict[str, object]):
        payloads.append(payload)
        segment_text = str(payload.get("segment_text", ""))
        tokens = payload.get("tokens", [])
        if not isinstance(tokens, list):
            tokens = []
        token_checks = []
        for item in tokens:
            token_payload = item if isinstance(item, dict) else {}
            token_checks.append(
                {
                    "token_index": int(token_payload.get("token_index", 0)),
                    "lemma": str(token_payload.get("lemma", "")),
                    "upos": str(token_payload.get("upos", "X")),
                    "feats": {},
                }
            )
        if segment_text == "Jake stood up for himself.":
            return {
                "token_checks": token_checks,
                "new_expressions": [
                    {
                        "text": "stood up for himself",
                        "lemma": "stand up for himself",
                        "kind": "phrasal_verb",
                        "start": 5,
                        "end": 25,
                    }
                ],
            }
        if segment_text == "He refused to rock the boat.":
            return {
                "token_checks": token_checks,
                "new_expressions": [
                    {
                        "text": "rock the boat",
                        "lemma": "rock the boat",
                        "kind": "idiom",
                        "start": 14,
                        "end": 27,
                    }
                ],
            }
        raise AssertionError(f"Unexpected segment_text: {segment_text}")

    repo = MorphologyRepository(
        enabled=True,
        db_path=str(db_path),
        logger_instance=logging.getLogger("test"),
        analyzer=lambda _: {
            "language": "en",
            "items": [
                {"token": "Jake", "lemma": "Jake", "upos": "PROPN", "feats": {}, "start": 0, "end": 4, "key": "jake|propn"},
                {"token": "stood", "lemma": "stand", "upos": "VERB", "feats": {}, "start": 5, "end": 10, "key": "stand|verb"},
                {"token": "up", "lemma": "up", "upos": "ADP", "feats": {}, "start": 11, "end": 13, "key": "up|adp"},
                {"token": "for", "lemma": "for", "upos": "ADP", "feats": {}, "start": 14, "end": 17, "key": "for|adp"},
                {"token": "himself", "lemma": "himself", "upos": "PRON", "feats": {}, "start": 18, "end": 25, "key": "himself|pron"},
                {"token": "He", "lemma": "he", "upos": "PRON", "feats": {}, "start": 27, "end": 29, "key": "he|pron"},
                {"token": "refused", "lemma": "refuse", "upos": "VERB", "feats": {}, "start": 30, "end": 37, "key": "refuse|verb"},
                {"token": "to", "lemma": "to", "upos": "PART", "feats": {}, "start": 38, "end": 40, "key": "to|part"},
                {"token": "rock", "lemma": "rock", "upos": "VERB", "feats": {}, "start": 41, "end": 45, "key": "rock|verb"},
                {"token": "the", "lemma": "the", "upos": "DET", "feats": {}, "start": 46, "end": 49, "key": "the|det"},
                {"token": "boat", "lemma": "boat", "upos": "NOUN", "feats": {}, "start": 50, "end": 54, "key": "boat|noun"},
            ],
        },
        expression_extractor=lambda _: [],
        lm_verifier=verifier,
        lm_verify_enabled=True,
    )
    repo.ingest_dialogue_parts([[("af_heart", text)]], source="generate_first")
    repo.wait_for_pending_reviews(3.0)

    assert len(payloads) == 2
    assert payloads[0]["segment_text"] == "Jake stood up for himself."
    assert payloads[1]["segment_text"] == "He refused to rock the boat."

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT expression_text, expression_type, start_offset, end_offset FROM morph_expressions ORDER BY start_offset"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 2
    assert rows[0][0].lower() == "stood up for himself"
    assert rows[0][1] == "phrasal_verb"
    assert rows[0][2] == 5
    assert rows[0][3] == 25
    assert rows[1][0].lower() == "rock the boat"
    assert rows[1][1] == "idiom"
    assert rows[1][2] == 41
    assert rows[1][3] == 54
