"""SQLite persistence for token morphology analysis."""

from __future__ import annotations

from collections import OrderedDict
import csv
from datetime import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

Analyzer = Callable[[str], dict[str, object]]
ExpressionExtractor = Callable[[str], list[dict[str, object]]]

POS_TABLE_COLUMNS: list[tuple[str, str]] = [
    ("Noun", "NOUN"),
    ("Verb", "VERB"),
    ("Adjective", "ADJ"),
    ("Adverb", "ADV"),
    ("Pronoun", "PRON"),
    ("ProperNoun", "PROPN"),
    ("Number", "NUM"),
    ("Determiner", "DET"),
    ("Adposition", "ADP"),
    ("CConj", "CCONJ"),
    ("SConj", "SCONJ"),
    ("Particle", "PART"),
    ("Interjection", "INTJ"),
    ("Symbol", "SYM"),
    ("Other", "X"),
]


@dataclass(frozen=True, slots=True)
class MorphRow:
    source: str
    part_index: int
    segment_index: int
    token_index: int
    voice: str
    token: str
    lemma: str
    upos: str
    feats_json: str
    start: int
    end: int
    key: str
    text_sha1: str


@dataclass(frozen=True, slots=True)
class ExpressionRow:
    source: str
    part_index: int
    segment_index: int
    expression_index: int
    voice: str
    expression_text: str
    expression_lemma: str
    expression_type: str
    start: int
    end: int
    expression_key: str
    match_source: str
    wordnet: int
    text_sha1: str


@dataclass(frozen=True, slots=True)
class _TokenTemplate:
    token_index: int
    token: str
    lemma: str
    upos: str
    feats_json: str
    start: int
    end: int
    key: str


@dataclass(frozen=True, slots=True)
class _ExpressionTemplate:
    expression_index: int
    expression_text: str
    expression_lemma: str
    expression_type: str
    start: int
    end: int
    expression_key: str
    match_source: str
    wordnet: int


def _default_analyzer(text: str) -> dict[str, object]:
    from ..domain.morphology import analyze_english_text

    return analyze_english_text(text)


def _default_expression_extractor(text: str) -> list[dict[str, object]]:
    from ..domain.expressions import extract_english_expressions

    return extract_english_expressions(text)


class MorphologyRepository:
    def __init__(
        self,
        *,
        enabled: bool,
        db_path: str,
        table_prefix: str = "morph_",
        logger_instance=None,
        analyzer: Analyzer | None = None,
        expression_extractor: ExpressionExtractor | None = None,
        segment_cache_size: int = 1024,
    ) -> None:
        self.logger = logger_instance or logger
        self.enabled = bool(enabled)
        self.db_path = db_path
        self.table_prefix = _normalize_prefix(table_prefix)
        self.lexemes_table = f"{self.table_prefix}lexemes"
        self.occurrences_table = f"{self.table_prefix}token_occurrences"
        self.expressions_table = f"{self.table_prefix}expressions"
        self._allowed_tables = {
            self.lexemes_table,
            self.occurrences_table,
            self.expressions_table,
        }
        self.analyzer = analyzer or _default_analyzer
        self.expression_extractor = expression_extractor or _default_expression_extractor
        self._db_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._segment_cache_size = max(0, int(segment_cache_size))
        self._segment_templates_cache: OrderedDict[
            str,
            tuple[str, tuple[_TokenTemplate, ...], tuple[_ExpressionTemplate, ...]],
        ] = OrderedDict()
        self._schema_ready = False
        if self.enabled and not self.db_path:
            self.logger.warning(
                "Morphology DB enabled but MORPH_DB_PATH is empty; disabling DB writes."
            )
            self.enabled = False

    def ensure_schema(self) -> None:
        """Create morphology tables and indexes when missing."""
        if not self.enabled:
            raise RuntimeError("Morphology DB is disabled.")
        if not self.db_path:
            raise RuntimeError("Morphology DB path is empty.")

        self._ensure_parent_dir()
        with self._db_lock:
            connection = self._open_connection()
            try:
                with connection:
                    self._ensure_schema_with_connection(connection)
            finally:
                connection.close()

    def _ensure_schema_with_connection(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        connection.execute(self._sql_create_lexemes_table())
        connection.execute(self._sql_create_occurrences_table())
        connection.execute(self._sql_create_expressions_table())
        for statement in self._sql_create_occurrence_indexes():
            connection.execute(statement)
        self._schema_ready = True

    def list_rows(
        self,
        *,
        dataset: str = "occurrences",
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[str], list[list[str]]]:
        if not self.enabled:
            raise RuntimeError("Morphology DB is disabled.")
        table_key, table_name = self._resolve_crud_table(dataset)
        table_name = self._validate_table_name(table_name)
        _ = table_key
        self.ensure_schema()
        safe_limit = max(1, min(int(limit), 1000))
        safe_offset = max(0, int(offset))

        with self._db_lock:
            connection = self._open_connection()
            try:
                headers = self._table_headers(connection, table_name)
                if not headers:
                    return [], []
                order_clause = '"id" DESC' if "id" in headers else "ROWID DESC"
                query = (
                    f'SELECT * FROM "{table_name}" '  # nosec
                    f"ORDER BY {order_clause} LIMIT ? OFFSET ?"
                )
                cursor = connection.execute(query, (safe_limit, safe_offset))
                rows = [[_stringify_cell(item) for item in row] for row in cursor.fetchall()]
                return headers, rows
            finally:
                connection.close()

    def ingest_dialogue_parts(
        self,
        parts: Sequence[Sequence[tuple[str, str]]],
        *,
        source: str,
    ) -> None:
        if not self.enabled:
            return
        rows, expression_rows = self._collect_ingest_rows(parts, source)
        if not rows and not expression_rows:
            return
        lexeme_unique: dict[tuple[str, str, str, str], None] = {}
        occurrence_rows: list[tuple[object, ...]] = []
        for row in rows:
            occurrence_rows.append(
                (
                    row.source,
                    row.part_index,
                    row.segment_index,
                    row.token_index,
                    row.voice,
                    row.token,
                    row.lemma,
                    row.upos,
                    row.feats_json,
                    row.start,
                    row.end,
                    row.key,
                    row.text_sha1,
                )
            )
            lexeme_unique[(row.key, row.lemma, row.upos, row.feats_json)] = None
        lexeme_rows = list(lexeme_unique.keys())
        unique_expression_rows = _expression_rows(expression_rows)
        if not lexeme_rows and not occurrence_rows and not unique_expression_rows:
            return
        try:
            self._ensure_parent_dir()
            with self._db_lock:
                connection = self._open_connection()
                try:
                    with connection:
                        self._ensure_schema_with_connection(connection)
                        if lexeme_rows:
                            connection.executemany(self._sql_insert_lexeme(), lexeme_rows)
                        if occurrence_rows:
                            connection.executemany(self._sql_insert_occurrence(), occurrence_rows)
                        if unique_expression_rows:
                            connection.executemany(
                                self._sql_insert_expression(),
                                unique_expression_rows,
                            )
                finally:
                    connection.close()
            self.logger.debug(
                "Morphology DB ingest complete: lexemes=%s occurrences=%s expressions=%s source=%s path=%s",
                len(lexeme_rows),
                len(occurrence_rows),
                len(unique_expression_rows),
                source,
                self.db_path,
            )
        except Exception:
            self.logger.exception("Morphology DB ingest failed")
            return

    def export_csv(
        self,
        *,
        dataset: str = "lexemes",
        output_dir: str = "outputs",
    ) -> str | None:
        return self._export_dataset_file(
            dataset=dataset,
            output_dir=output_dir,
            file_format="csv",
        )

    def export_txt(
        self,
        *,
        dataset: str = "lexemes",
        output_dir: str = "outputs",
    ) -> str | None:
        return self._export_dataset_file(
            dataset=dataset,
            output_dir=output_dir,
            file_format="txt",
        )

    def export_word_table(
        self,
        *,
        dataset: str = "lexemes",
        output_dir: str = "outputs",
    ) -> str | None:
        """Backward-compatible alias for Excel export."""
        return self.export_excel(
            dataset=dataset,
            output_dir=output_dir,
        )

    def export_excel(
        self,
        *,
        dataset: str = "lexemes",
        output_dir: str = "outputs",
    ) -> str | None:
        return self._export_dataset_file(
            dataset=dataset,
            output_dir=output_dir,
            file_format="xlsx",
        )

    def export_spreadsheet(
        self,
        *,
        dataset: str = "lexemes",
        output_dir: str = "outputs",
    ) -> str | None:
        return self._export_dataset_file(
            dataset=dataset,
            output_dir=output_dir,
            file_format="ods",
        )

    def _export_dataset_file(
        self,
        *,
        dataset: str = "lexemes",
        output_dir: str = "outputs",
        file_format: str = "ods",
    ) -> str | None:
        if not self.enabled or not self.db_path or not os.path.isfile(self.db_path):
            return None
        normalized_format = str(file_format or "ods").strip().lower().lstrip(".")
        if normalized_format not in {"ods", "csv", "txt", "xlsx"}:
            raise ValueError(f"Unsupported morphology export format: {file_format}")

        export_dir = self._resolve_vocabulary_export_dir(output_dir)
        table_name, sheet_name, headers, rows = self._build_export_dataset(dataset)
        if not rows:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            export_dir,
            f"{table_name}_{timestamp}.{normalized_format}",
        )

        if normalized_format == "csv":
            self._write_csv(headers, rows, output_path)
            return output_path
        if normalized_format == "txt":
            self._write_txt(headers, rows, output_path)
            return output_path
        if normalized_format == "xlsx":
            self._write_xlsx(headers, rows, output_path, sheet_name=sheet_name)
            return output_path
        self._write_ods(headers, rows, output_path, sheet_name=sheet_name)
        return output_path

    def _build_export_dataset(
        self,
        dataset: str,
    ) -> tuple[str, str, list[str], list[list[str]]]:
        normalized_dataset = (dataset or "lexemes").strip().lower()
        if normalized_dataset in ("pos_table", "upos_table", "parts_of_speech"):
            headers, rows = self._build_pos_table_dataset()
            return f"{self.table_prefix}general_table", "General Table", headers, rows

        table_name = self._table_name_for_dataset(normalized_dataset)
        headers, rows = self._query_table_rows(table_name)
        return table_name, table_name, headers, rows

    def _write_ods(
        self,
        headers: list[str],
        rows: list[list[str]],
        output_path: str,
        *,
        sheet_name: str,
    ) -> None:
        try:
            from odf.opendocument import OpenDocumentSpreadsheet
            from odf.table import Table, TableCell, TableColumn, TableRow
            from odf.text import P
        except Exception as exc:
            raise RuntimeError("ODF export requires odfpy. Install dependency and retry.") from exc

        document = OpenDocumentSpreadsheet()
        table = Table(name=sheet_name[:31] or "Sheet1")
        for _ in headers:
            table.addElement(TableColumn())

        def add_row(values: list[str]) -> None:
            row = TableRow()
            for value in values:
                cell = TableCell(valuetype="string")
                cell.addElement(P(text=str(value)))
                row.addElement(cell)
            table.addElement(row)

        add_row(headers)
        for data_row in rows:
            add_row(data_row)

        document.spreadsheet.addElement(table)
        document.save(output_path, addsuffix=False)

    def _write_csv(
        self,
        headers: list[str],
        rows: list[list[str]],
        output_path: str,
    ) -> None:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)

    def _write_txt(
        self,
        headers: list[str],
        rows: list[list[str]],
        output_path: str,
    ) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(headers)
            writer.writerows(rows)

    def _write_xlsx(
        self,
        headers: list[str],
        rows: list[list[str]],
        output_path: str,
        *,
        sheet_name: str,
    ) -> None:
        try:
            from openpyxl import Workbook
        except Exception as exc:
            raise RuntimeError(
                "Excel export requires openpyxl. Install dependency and retry."
            ) from exc

        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = sheet_name[:31] or "Sheet1"
        worksheet.append([str(header) for header in headers])
        for row_values in rows:
            worksheet.append([str(value) for value in row_values[: len(headers)]])
        workbook.save(output_path)

    def _segment_hash(self, segment_text: str) -> str:
        return hashlib.sha1(
            segment_text.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()

    def _serialize_feats_json(self, raw_feats: object) -> str:
        if not isinstance(raw_feats, dict) or not raw_feats:
            return "{}"
        return json.dumps(raw_feats, ensure_ascii=False, sort_keys=True)

    def _collect_segment_tokens(
        self,
        *,
        source: str,
        part_index: int,
        segment_index: int,
        voice: str,
        segment_text: str,
        segment_hash: str,
        analysis: object | None = None,
    ) -> list[MorphRow]:
        token_rows: list[MorphRow] = []
        token_templates = self._collect_segment_token_templates(
            segment_text=segment_text,
            analysis=analysis,
        )
        self._append_token_rows(
            token_rows,
            source=source,
            part_index=part_index,
            segment_index=segment_index,
            voice=voice,
            segment_hash=segment_hash,
            token_templates=token_templates,
        )
        return token_rows

    def _collect_segment_expressions(
        self,
        *,
        source: str,
        part_index: int,
        segment_index: int,
        voice: str,
        segment_text: str,
        segment_hash: str,
        expressions: object | None = None,
    ) -> list[ExpressionRow]:
        expression_rows: list[ExpressionRow] = []
        expression_templates = self._collect_segment_expression_templates(
            segment_text=segment_text,
            expressions=expressions,
        )
        self._append_expression_rows(
            expression_rows,
            source=source,
            part_index=part_index,
            segment_index=segment_index,
            voice=voice,
            segment_hash=segment_hash,
            expression_templates=expression_templates,
        )
        return expression_rows

    def _collect_ingest_rows(
        self,
        parts: Sequence[Sequence[tuple[str, str]]],
        source: str,
    ) -> tuple[list[MorphRow], list[ExpressionRow]]:
        source = (source or "unknown").strip() or "unknown"
        token_rows: list[MorphRow] = []
        expression_rows: list[ExpressionRow] = []
        for part_index, segments in enumerate(parts):
            for segment_index, (voice, text) in enumerate(segments):
                segment_text = (text or "").strip()
                if not segment_text:
                    continue
                voice_text = str(voice or "")
                segment_hash, token_templates, expression_templates = self._segment_templates(
                    segment_text
                )
                if token_templates:
                    self._append_token_rows(
                        token_rows,
                        source=source,
                        part_index=part_index,
                        segment_index=segment_index,
                        voice=voice_text,
                        segment_hash=segment_hash,
                        token_templates=token_templates,
                    )
                if expression_templates:
                    self._append_expression_rows(
                        expression_rows,
                        source=source,
                        part_index=part_index,
                        segment_index=segment_index,
                        voice=voice_text,
                        segment_hash=segment_hash,
                        expression_templates=expression_templates,
                    )
        return token_rows, expression_rows

    def _collect_segment_token_templates(
        self,
        *,
        segment_text: str,
        analysis: object | None = None,
    ) -> tuple[_TokenTemplate, ...]:
        if analysis is None:
            analysis = self.analyzer(segment_text)
        items = analysis.get("items", []) if isinstance(analysis, dict) else []
        if not isinstance(items, list):
            return ()
        templates: list[_TokenTemplate] = []
        append_template = templates.append
        serialize_feats = self._serialize_feats_json
        for token_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            token = str(item.get("token", "")).strip()
            if not token:
                continue
            lemma = str(item.get("lemma", token)).strip() or token
            upos = str(item.get("upos", "X")).strip().upper() or "X"
            start = _to_int(item.get("start"), 0)
            end = _to_int(item.get("end"), start)
            dedup_key = str(item.get("key", "")).strip() or _build_key(lemma, upos)
            append_template(
                _TokenTemplate(
                    token_index=token_index,
                    token=token,
                    lemma=lemma,
                    upos=upos,
                    feats_json=serialize_feats(item.get("feats")),
                    start=start,
                    end=end,
                    key=dedup_key,
                )
            )
        return tuple(templates)

    def _collect_segment_expression_templates(
        self,
        *,
        segment_text: str,
        expressions: object | None = None,
    ) -> tuple[_ExpressionTemplate, ...]:
        if expressions is None:
            expressions = self.expression_extractor(segment_text)
        if not expressions:
            return ()
        templates: list[_ExpressionTemplate] = []
        append_template = templates.append
        for expression_index, item in enumerate(expressions):
            if not isinstance(item, dict):
                continue
            expression_text = str(item.get("text", "")).strip()
            if not expression_text:
                continue
            expression_lemma = str(item.get("lemma", expression_text)).strip() or expression_text
            expression_type = str(item.get("kind", "expression")).strip() or "expression"
            start = _to_int(item.get("start"), 0)
            end = _to_int(item.get("end"), start)
            expression_key = str(item.get("key", "")).strip() or (
                f"{expression_lemma.lower()}|{expression_type.lower()}"
            )
            match_source = str(item.get("source", "unknown")).strip() or "unknown"
            append_template(
                _ExpressionTemplate(
                    expression_index=expression_index,
                    expression_text=expression_text,
                    expression_lemma=expression_lemma,
                    expression_type=expression_type,
                    start=start,
                    end=end,
                    expression_key=expression_key,
                    match_source=match_source,
                    wordnet=1 if bool(item.get("wordnet")) else 0,
                )
            )
        return tuple(templates)

    def _segment_templates(
        self,
        segment_text: str,
    ) -> tuple[str, tuple[_TokenTemplate, ...], tuple[_ExpressionTemplate, ...]]:
        cached = self._cache_lookup(self._segment_templates_cache, segment_text)
        if cached is not None:
            return cached
        analysis = self.analyzer(segment_text)
        expressions = self.expression_extractor(segment_text)
        payload = (
            self._segment_hash(segment_text),
            self._collect_segment_token_templates(
                segment_text=segment_text,
                analysis=analysis,
            ),
            self._collect_segment_expression_templates(
                segment_text=segment_text,
                expressions=expressions,
            ),
        )
        self._cache_store(self._segment_templates_cache, segment_text, payload)
        return payload

    def _append_token_rows(
        self,
        target: list[MorphRow],
        *,
        source: str,
        part_index: int,
        segment_index: int,
        voice: str,
        segment_hash: str,
        token_templates: Sequence[_TokenTemplate],
    ) -> None:
        append_row = target.append
        for template in token_templates:
            append_row(
                MorphRow(
                    source=source,
                    part_index=part_index,
                    segment_index=segment_index,
                    token_index=template.token_index,
                    voice=voice,
                    token=template.token,
                    lemma=template.lemma,
                    upos=template.upos,
                    feats_json=template.feats_json,
                    start=template.start,
                    end=template.end,
                    key=template.key,
                    text_sha1=segment_hash,
                )
            )

    def _append_expression_rows(
        self,
        target: list[ExpressionRow],
        *,
        source: str,
        part_index: int,
        segment_index: int,
        voice: str,
        segment_hash: str,
        expression_templates: Sequence[_ExpressionTemplate],
    ) -> None:
        append_row = target.append
        for template in expression_templates:
            append_row(
                ExpressionRow(
                    source=source,
                    part_index=part_index,
                    segment_index=segment_index,
                    expression_index=template.expression_index,
                    voice=voice,
                    expression_text=template.expression_text,
                    expression_lemma=template.expression_lemma,
                    expression_type=template.expression_type,
                    start=template.start,
                    end=template.end,
                    expression_key=template.expression_key,
                    match_source=template.match_source,
                    wordnet=template.wordnet,
                    text_sha1=segment_hash,
                )
            )

    def _cache_lookup(self, cache: OrderedDict[str, object], key: str) -> object | None:
        if self._segment_cache_size <= 0:
            return None
        with self._cache_lock:
            value = cache.get(key)
            if value is None:
                return None
            cache.move_to_end(key)
            return value

    def _cache_store(self, cache: OrderedDict[str, object], key: str, value: object) -> None:
        if self._segment_cache_size <= 0:
            return
        with self._cache_lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > self._segment_cache_size:
                cache.popitem(last=False)

    def _resolve_crud_table(self, dataset: str) -> tuple[str, str]:
        normalized = (dataset or "").strip().lower()
        if normalized in ("lexemes", "lexeme"):
            return "lexemes", self.lexemes_table
        if normalized in ("occurrences", "token_occurrences", "occurrence"):
            return "occurrences", self.occurrences_table
        if normalized in ("expressions", "expression", "mwe", "mwes", "idioms"):
            return "expressions", self.expressions_table
        raise ValueError("Unsupported dataset. Use lexemes, occurrences, or expressions.")

    def _validate_table_name(self, table_name: str) -> str:
        if table_name not in self._allowed_tables:
            raise ValueError("Unsupported table requested.")
        return table_name

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout=30000")
        if self.db_path and self.db_path != ":memory:":
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError:
                # Keep runtime resilient on restricted filesystems.
                pass
        return connection

    def _table_headers(self, connection: sqlite3.Connection, table_name: str) -> list[str]:
        table_name = self._validate_table_name(table_name)
        cursor = connection.execute(f'PRAGMA table_info("{table_name}")')
        return [str(row[1]) for row in cursor.fetchall() if len(row) > 1]

    def _ensure_parent_dir(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _ensure_dir(self, path: str) -> None:
        target = os.path.abspath(path)
        os.makedirs(target, exist_ok=True)

    def _resolve_vocabulary_export_dir(self, output_dir: str) -> str:
        date_dir = os.path.join(output_dir, datetime.now().strftime("%Y-%m-%d"))
        records_dir = os.path.join(date_dir, "records")
        vocabulary_dir = os.path.join(date_dir, "vocabulary")
        self._ensure_dir(records_dir)
        self._ensure_dir(vocabulary_dir)
        return vocabulary_dir

    def _table_name_for_dataset(self, dataset: str) -> str:
        normalized = (dataset or "lexemes").strip().lower()
        if normalized in ("occurrences", "token_occurrences"):
            return self.occurrences_table
        if normalized in ("expressions", "mwe", "mwes", "phrasal_verbs", "idioms"):
            return self.expressions_table
        return self.lexemes_table

    def _query_table_rows(self, table_name: str) -> tuple[list[str], list[list[str]]]:
        table_name = self._validate_table_name(table_name)
        with self._db_lock:
            connection = None
            try:
                connection = self._open_connection()
                cursor = connection.execute(f'SELECT * FROM "{table_name}" ORDER BY ROWID')  # nosec
                headers = [description[0] for description in cursor.description or []]
                rows = [[_stringify_cell(item) for item in row] for row in cursor.fetchall()]
                return headers, rows
            finally:
                if connection is not None:
                    connection.close()

    def _build_pos_table_dataset(self) -> tuple[list[str], list[list[str]]]:
        with self._db_lock:
            connection = None
            try:
                connection = self._open_connection()
                cursor = connection.execute(
                    f'SELECT "upos", "lemma" FROM "{self.lexemes_table}" ORDER BY "upos", "lemma" COLLATE NOCASE'  # nosec
                )
                entries = cursor.fetchall()
                if not entries:
                    return [], []
                return _build_pos_table_from_entries(entries)
            finally:
                if connection is not None:
                    connection.close()

    def _sql_create_lexemes_table(self) -> str:
        return f"""
CREATE TABLE IF NOT EXISTS "{self.lexemes_table}" (
  "dedup_key" TEXT NOT NULL PRIMARY KEY,
  "lemma" TEXT NOT NULL,
  "upos" TEXT NOT NULL,
  "feats_json" TEXT NOT NULL,
  "created_at" TEXT NOT NULL DEFAULT (datetime('now'))
)
""".strip()

    def _sql_create_occurrences_table(self) -> str:
        return f"""
CREATE TABLE IF NOT EXISTS "{self.occurrences_table}" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "source" TEXT NOT NULL,
  "part_index" INTEGER NOT NULL,
  "segment_index" INTEGER NOT NULL,
  "token_index" INTEGER NOT NULL,
  "voice" TEXT NOT NULL,
  "token_text" TEXT NOT NULL,
  "lemma" TEXT NOT NULL,
  "upos" TEXT NOT NULL,
  "feats_json" TEXT NOT NULL,
  "start_offset" INTEGER NOT NULL,
  "end_offset" INTEGER NOT NULL,
  "dedup_key" TEXT NOT NULL,
  "text_sha1" TEXT NOT NULL,
  "created_at" TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE ("source", "text_sha1", "part_index", "segment_index", "token_index", "dedup_key", "start_offset", "end_offset")
)
""".strip()

    def _sql_create_expressions_table(self) -> str:
        return f"""
CREATE TABLE IF NOT EXISTS "{self.expressions_table}" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "source" TEXT NOT NULL,
  "part_index" INTEGER NOT NULL,
  "segment_index" INTEGER NOT NULL,
  "expression_index" INTEGER NOT NULL,
  "voice" TEXT NOT NULL,
  "expression_text" TEXT NOT NULL,
  "expression_lemma" TEXT NOT NULL,
  "expression_type" TEXT NOT NULL,
  "start_offset" INTEGER NOT NULL,
  "end_offset" INTEGER NOT NULL,
  "expression_key" TEXT NOT NULL,
  "match_source" TEXT NOT NULL,
  "wordnet_hit" INTEGER NOT NULL DEFAULT 0,
  "text_sha1" TEXT NOT NULL,
  "created_at" TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE ("source", "text_sha1", "part_index", "segment_index", "expression_index", "expression_key", "start_offset", "end_offset")
)
""".strip()

    def _sql_create_occurrence_indexes(self) -> list[str]:
        return [
            (
                f'CREATE INDEX IF NOT EXISTS "{self.occurrences_table}_dedup_key_idx" '
                f'ON "{self.occurrences_table}" ("dedup_key")'
            ),
            (
                f'CREATE INDEX IF NOT EXISTS "{self.expressions_table}_key_idx" '
                f'ON "{self.expressions_table}" ("expression_key")'
            ),
            (
                f'CREATE INDEX IF NOT EXISTS "{self.expressions_table}_type_idx" '
                f'ON "{self.expressions_table}" ("expression_type")'
            ),
        ]

    def _sql_insert_lexeme(self) -> str:
        return (
            f'INSERT OR IGNORE INTO "{self.lexemes_table}" '
            '("dedup_key", "lemma", "upos", "feats_json") VALUES (?, ?, ?, ?)'
        )

    def _sql_insert_occurrence(self) -> str:
        return (
            f'INSERT OR IGNORE INTO "{self.occurrences_table}" '
            '("source", "part_index", "segment_index", "token_index", "voice", '
            '"token_text", "lemma", "upos", "feats_json", "start_offset", '
            '"end_offset", "dedup_key", "text_sha1") VALUES '
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

    def _sql_insert_expression(self) -> str:
        return (
            f'INSERT OR IGNORE INTO "{self.expressions_table}" '
            '("source", "part_index", "segment_index", "expression_index", "voice", '
            '"expression_text", "expression_lemma", "expression_type", "start_offset", '
            '"end_offset", "expression_key", "match_source", "wordnet_hit", "text_sha1") VALUES '
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )


def _normalize_prefix(prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", prefix or "morph_")
    if not cleaned:
        return "morph_"
    return cleaned


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_key(lemma: str, upos: str) -> str:
    return f"{lemma.strip().lower()}|{upos.strip().lower()}"


def _sha1_text(value: str) -> str:
    text = str(value or "")
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _normalize_json_text(value: object) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return "{}"
    raw = str(value).strip()
    if not raw:
        return "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in feats_json.") from exc
    if not isinstance(parsed, (dict, list)):
        raise ValueError("feats_json must be a JSON object or array.")
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def _stringify_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _build_pos_table_from_entries(
    entries: Sequence[tuple[object, object]],
) -> tuple[list[str], list[list[str]]]:
    by_upos: dict[str, list[str]] = {}
    seen_by_upos: dict[str, set[str]] = {}
    for upos, lemma in entries:
        upos_text = str(upos or "").strip().upper()
        lemma_text = str(lemma or "").strip()
        if not upos_text or not lemma_text:
            continue
        seen = seen_by_upos.setdefault(upos_text, set())
        if lemma_text in seen:
            continue
        seen.add(lemma_text)
        by_upos.setdefault(upos_text, []).append(lemma_text)

    if not by_upos:
        return [], []
    column_pairs = _resolve_pos_column_pairs(by_upos)
    headers = [label for label, _ in column_pairs]
    max_rows = max((len(by_upos.get(upos, [])) for _, upos in column_pairs), default=0)
    if max_rows == 0:
        return [], []

    rows: list[list[str]] = []
    for row_index in range(max_rows):
        row: list[str] = []
        for _, upos in column_pairs:
            words = by_upos.get(upos, [])
            row.append(words[row_index] if row_index < len(words) else "")
        rows.append(row)
    return headers, rows


def _resolve_pos_column_pairs(by_upos: dict[str, list[str]]) -> list[tuple[str, str]]:
    column_pairs = list(POS_TABLE_COLUMNS)
    known_upos = {upos for _, upos in column_pairs}
    extra_upos = sorted([upos for upos in by_upos if upos not in known_upos])
    for upos in extra_upos:
        column_pairs.append((upos, upos))
    return column_pairs


def _occurrence_rows(rows: Iterable[MorphRow]) -> list[tuple[object, ...]]:
    return [
        (
            row.source,
            row.part_index,
            row.segment_index,
            row.token_index,
            row.voice,
            row.token,
            row.lemma,
            row.upos,
            row.feats_json,
            row.start,
            row.end,
            row.key,
            row.text_sha1,
        )
        for row in rows
    ]


def _expression_rows(rows: Iterable[ExpressionRow]) -> list[tuple[object, ...]]:
    return [
        (
            row.source,
            row.part_index,
            row.segment_index,
            row.expression_index,
            row.voice,
            row.expression_text,
            row.expression_lemma,
            row.expression_type,
            row.start,
            row.end,
            row.expression_key,
            row.match_source,
            row.wordnet,
            row.text_sha1,
        )
        for row in rows
    ]


def _unique_lexeme_rows(rows: Iterable[MorphRow]) -> list[tuple[str, str, str, str]]:
    unique: dict[tuple[str, str, str, str], None] = {}
    for row in rows:
        unique[(row.key, row.lemma, row.upos, row.feats_json)] = None
    return list(unique.keys())


def _unique_occurrence_rows(rows: Iterable[MorphRow]) -> list[tuple[object, ...]]:
    unique: dict[tuple[object, ...], None] = {}
    for row in rows:
        unique[
            (
                row.source,
                row.part_index,
                row.segment_index,
                row.token_index,
                row.voice,
                row.token,
                row.lemma,
                row.upos,
                row.feats_json,
                row.start,
                row.end,
                row.key,
                row.text_sha1,
            )
        ] = None
    return list(unique.keys())


def _unique_expression_rows(rows: Iterable[ExpressionRow]) -> list[tuple[object, ...]]:
    unique: dict[tuple[object, ...], None] = {}
    for row in rows:
        unique[
            (
                row.source,
                row.part_index,
                row.segment_index,
                row.expression_index,
                row.voice,
                row.expression_text,
                row.expression_lemma,
                row.expression_type,
                row.start,
                row.end,
                row.expression_key,
                row.match_source,
                row.wordnet,
                row.text_sha1,
            )
        ] = None
    return list(unique.keys())
