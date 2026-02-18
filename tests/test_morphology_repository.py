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


def test_morphology_repository_list_rows_read_only_view(tmp_path: Path):
    db_path = tmp_path / "list_rows.sqlite3"
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

    headers, rows = repo.list_rows(dataset="occurrences", limit=20, offset=0)
    assert "id" in headers
    assert "token_text" in headers
    assert rows
    assert rows[0][headers.index("token_text")] == "Cats"


def test_morphology_repository_keeps_legacy_extra_tables(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite3"
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
                }
            ],
        },
        expression_extractor=lambda _: [],
    )
    repo.ensure_schema()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute('CREATE TABLE IF NOT EXISTS "morph_reviews" ("id" INTEGER PRIMARY KEY, "status" TEXT)')
        connection.execute('INSERT INTO "morph_reviews" ("status") VALUES ("legacy")')
        connection.commit()
    finally:
        connection.close()

    repo.ingest_dialogue_parts([[("af_heart", "look")]], source="generate_first")
    headers, rows = repo.list_rows(dataset="occurrences", limit=20, offset=0)
    assert headers
    assert rows

    connection = sqlite3.connect(db_path)
    try:
        legacy_rows = connection.execute('SELECT COUNT(*) FROM "morph_reviews"').fetchone()[0]
    finally:
        connection.close()
    assert legacy_rows == 1


def test_collect_ingest_rows_reuses_analysis_for_identical_segments(tmp_path: Path):
    calls = {"analyzer": 0, "expressions": 0}

    def analyzer(text: str):
        calls["analyzer"] += 1
        return {
            "language": "en",
            "items": [
                {
                    "token": text,
                    "lemma": text.lower(),
                    "upos": "NOUN",
                    "feats": {},
                    "start": 0,
                    "end": len(text),
                    "key": f"{text.lower()}|noun",
                }
            ],
        }

    def expression_extractor(_text: str):
        calls["expressions"] += 1
        return []

    repo = MorphologyRepository(
        enabled=False,
        db_path=str(tmp_path / "cache.sqlite3"),
        logger_instance=logging.getLogger("test"),
        analyzer=analyzer,
        expression_extractor=expression_extractor,
    )
    token_rows, expression_rows = repo._collect_ingest_rows(
        [
            [("af_heart", "Same text"), ("af_heart", "Same text")],
            [("af_heart", "Different text")],
        ],
        source="generate_first",
    )

    assert len(token_rows) == 3
    assert expression_rows == []
    assert calls["analyzer"] == 2
    assert calls["expressions"] == 2


def test_collect_ingest_rows_reuses_cached_analysis_across_calls(tmp_path: Path):
    calls = {"analyzer": 0, "expressions": 0}

    def analyzer(text: str):
        calls["analyzer"] += 1
        return {
            "language": "en",
            "items": [
                {
                    "token": text,
                    "lemma": text.lower(),
                    "upos": "NOUN",
                    "feats": {},
                    "start": 0,
                    "end": len(text),
                    "key": f"{text.lower()}|noun",
                }
            ],
        }

    def expression_extractor(_text: str):
        calls["expressions"] += 1
        return []

    repo = MorphologyRepository(
        enabled=False,
        db_path=str(tmp_path / "cache_cross_call.sqlite3"),
        logger_instance=logging.getLogger("test"),
        analyzer=analyzer,
        expression_extractor=expression_extractor,
    )
    rows_a, expressions_a = repo._collect_ingest_rows(
        [[("af_heart", "Repeated text")]],
        source="generate_first",
    )
    rows_b, expressions_b = repo._collect_ingest_rows(
        [[("af_heart", "Repeated text")]],
        source="generate_first",
    )

    assert rows_a and rows_b
    assert expressions_a == []
    assert expressions_b == []
    assert calls["analyzer"] == 1
    assert calls["expressions"] == 1
