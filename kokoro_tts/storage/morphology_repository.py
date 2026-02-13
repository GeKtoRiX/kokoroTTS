"""SQLite persistence for token morphology analysis."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..domain.expressions import extract_english_expressions
from ..domain.morphology import analyze_english_text

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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    ) -> None:
        self.logger = logger_instance or logger
        self.enabled = bool(enabled)
        self.db_path = db_path
        self.table_prefix = _normalize_prefix(table_prefix)
        self.lexemes_table = f"{self.table_prefix}lexemes"
        self.occurrences_table = f"{self.table_prefix}token_occurrences"
        self.expressions_table = f"{self.table_prefix}expressions"
        self.analyzer = analyzer or analyze_english_text
        self.expression_extractor = expression_extractor or extract_english_expressions
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
        connection = sqlite3.connect(self.db_path)
        try:
            with connection:
                connection.execute(self._sql_create_lexemes_table())
                connection.execute(self._sql_create_occurrences_table())
                connection.execute(self._sql_create_expressions_table())
                for statement in self._sql_create_occurrence_indexes():
                    connection.execute(statement)
                self._schema_ready = True
        finally:
            connection.close()

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
        _ = table_key
        self.ensure_schema()
        safe_limit = max(1, min(int(limit), 1000))
        safe_offset = max(0, int(offset))

        connection = sqlite3.connect(self.db_path)
        try:
            headers = self._table_headers(connection, table_name)
            if not headers:
                return [], []
            order_clause = '"id" DESC' if "id" in headers else "ROWID DESC"
            query = (
                f'SELECT * FROM "{table_name}" '
                f"ORDER BY {order_clause} LIMIT ? OFFSET ?"
            )
            cursor = connection.execute(query, (safe_limit, safe_offset))
            rows = [[_stringify_cell(item) for item in row] for row in cursor.fetchall()]
            return headers, rows
        finally:
            connection.close()

    def insert_row(
        self,
        *,
        dataset: str,
        payload: dict[str, Any],
    ) -> str:
        if not self.enabled:
            raise RuntimeError("Morphology DB is disabled.")
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object.")

        table_key, table_name = self._resolve_crud_table(dataset)
        row = self._normalize_insert_payload(table_key, payload)
        columns = list(row.keys())
        values = [row[column] for column in columns]
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})'

        self.ensure_schema()
        connection = sqlite3.connect(self.db_path)
        try:
            with connection:
                cursor = connection.execute(sql, values)
                if table_key == "lexemes":
                    return str(row["dedup_key"])
                return str(cursor.lastrowid)
        finally:
            connection.close()

    def update_row(
        self,
        *,
        dataset: str,
        row_id: str,
        payload: dict[str, Any],
    ) -> int:
        if not self.enabled:
            raise RuntimeError("Morphology DB is disabled.")
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object.")

        table_key, table_name = self._resolve_crud_table(dataset)
        primary_column = "dedup_key" if table_key == "lexemes" else "id"
        update_payload = self._normalize_update_payload(table_key, payload)
        if not update_payload:
            raise ValueError("No editable fields in payload.")
        row_id_value = self._normalize_row_identifier(table_key, row_id)

        assignments = ", ".join(f'"{column}" = ?' for column in update_payload.keys())
        values = list(update_payload.values()) + [row_id_value]
        sql = f'UPDATE "{table_name}" SET {assignments} WHERE "{primary_column}" = ?'

        self.ensure_schema()
        connection = sqlite3.connect(self.db_path)
        try:
            with connection:
                cursor = connection.execute(sql, values)
                return int(cursor.rowcount or 0)
        finally:
            connection.close()

    def delete_row(
        self,
        *,
        dataset: str,
        row_id: str,
    ) -> int:
        if not self.enabled:
            raise RuntimeError("Morphology DB is disabled.")
        table_key, table_name = self._resolve_crud_table(dataset)
        primary_column = "dedup_key" if table_key == "lexemes" else "id"
        row_id_value = self._normalize_row_identifier(table_key, row_id)
        sql = f'DELETE FROM "{table_name}" WHERE "{primary_column}" = ?'

        self.ensure_schema()
        connection = sqlite3.connect(self.db_path)
        try:
            with connection:
                cursor = connection.execute(sql, (row_id_value,))
                return int(cursor.rowcount or 0)
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
        lexeme_rows = _unique_lexeme_rows(rows)
        occurrence_rows = _occurrence_rows(rows)
        unique_expression_rows = _expression_rows(expression_rows)
        if not lexeme_rows and not occurrence_rows and not unique_expression_rows:
            return
        connection = None
        try:
            self._ensure_parent_dir()
            connection = sqlite3.connect(self.db_path)
            with connection:
                if not self._schema_ready:
                    connection.execute(self._sql_create_lexemes_table())
                    connection.execute(self._sql_create_occurrences_table())
                    connection.execute(self._sql_create_expressions_table())
                    for statement in self._sql_create_occurrence_indexes():
                        connection.execute(statement)
                    self._schema_ready = True
                if lexeme_rows:
                    connection.executemany(self._sql_insert_lexeme(), lexeme_rows)
                if occurrence_rows:
                    connection.executemany(self._sql_insert_occurrence(), occurrence_rows)
                if unique_expression_rows:
                    connection.executemany(
                        self._sql_insert_expression(),
                        unique_expression_rows,
                    )
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
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    self.logger.exception("Morphology DB close failed")

    def export_csv(
        self,
        *,
        dataset: str = "lexemes",
        output_dir: str = "outputs",
    ) -> str | None:
        if not self.enabled or not self.db_path or not os.path.isfile(self.db_path):
            return None
        export_dir = self._resolve_vocabulary_export_dir(output_dir)
        normalized_dataset = (dataset or "lexemes").strip().lower()
        if normalized_dataset in ("pos_table", "upos_table", "parts_of_speech"):
            return self._export_pos_table_csv(export_dir)
        table_name = self._table_name_for_dataset(normalized_dataset)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{table_name}_{timestamp}.csv"
        csv_path = os.path.join(export_dir, filename)

        connection = None
        try:
            connection = sqlite3.connect(self.db_path)
            cursor = connection.execute(f'SELECT * FROM "{table_name}" ORDER BY ROWID')
            headers = [description[0] for description in cursor.description or []]
            rows = cursor.fetchall()
            if not rows:
                return None
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                writer.writerows(rows)
            return csv_path
        finally:
            if connection is not None:
                connection.close()

    def export_spreadsheet(
        self,
        *,
        dataset: str = "lexemes",
        output_dir: str = "outputs",
    ) -> str | None:
        if not self.enabled or not self.db_path or not os.path.isfile(self.db_path):
            return None
        export_dir = self._resolve_vocabulary_export_dir(output_dir)
        normalized_dataset = (dataset or "lexemes").strip().lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if normalized_dataset in ("pos_table", "upos_table", "parts_of_speech"):
            headers, rows = self._build_pos_table_dataset()
            if not rows:
                return None
            filename = f"{self.table_prefix}pos_table_{timestamp}.ods"
            output_path = os.path.join(export_dir, filename)
            self._write_ods(headers, rows, output_path, sheet_name="POS Table")
            return output_path

        table_name = self._table_name_for_dataset(normalized_dataset)
        headers, rows = self._query_table_rows(table_name)
        if not rows:
            return None
        filename = f"{table_name}_{timestamp}.ods"
        output_path = os.path.join(export_dir, filename)
        self._write_ods(headers, rows, output_path, sheet_name=table_name)
        return output_path

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
            raise RuntimeError(
                "ODF export requires odfpy. Install dependency and retry."
            ) from exc

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

    def _export_pos_table_csv(self, output_dir: str) -> str | None:
        self._ensure_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.table_prefix}pos_table_{timestamp}.csv"
        csv_path = os.path.join(output_dir, filename)

        connection = None
        try:
            connection = sqlite3.connect(self.db_path)
            cursor = connection.execute(
                f'SELECT "upos", "lemma" FROM "{self.lexemes_table}" ORDER BY "upos", "lemma" COLLATE NOCASE'
            )
            entries = cursor.fetchall()
            if not entries:
                return None
            headers, rows = _build_pos_table_from_entries(entries)
            if not rows:
                return None

            with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                writer.writerows(rows)
            return csv_path
        finally:
            if connection is not None:
                connection.close()

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
                segment_hash = hashlib.sha1(
                    segment_text.encode("utf-8"),
                    usedforsecurity=False,
                ).hexdigest()
                analysis = self.analyzer(segment_text)
                items = analysis.get("items", []) if isinstance(analysis, dict) else []
                if isinstance(items, list):
                    for token_index, item in enumerate(items):
                        if not isinstance(item, dict):
                            continue
                        token = str(item.get("token", "")).strip()
                        if not token:
                            continue
                        lemma = str(item.get("lemma", token)).strip() or token
                        upos = str(item.get("upos", "X")).strip().upper() or "X"
                        raw_feats = item.get("feats")
                        feats = raw_feats if isinstance(raw_feats, dict) else {}
                        start = _to_int(item.get("start"), 0)
                        end = _to_int(item.get("end"), start)
                        dedup_key = str(item.get("key", "")).strip() or _build_key(lemma, upos)
                        token_rows.append(
                            MorphRow(
                                source=source,
                                part_index=part_index,
                                segment_index=segment_index,
                                token_index=token_index,
                                voice=str(voice or ""),
                                token=token,
                                lemma=lemma,
                                upos=upos,
                                feats_json=json.dumps(feats, ensure_ascii=False, sort_keys=True),
                                start=start,
                                end=end,
                                key=dedup_key,
                                text_sha1=segment_hash,
                            )
                        )

                expressions = self.expression_extractor(segment_text)
                if expressions:
                    for expression_index, item in enumerate(expressions):
                        if not isinstance(item, dict):
                            continue
                        expression_text = str(item.get("text", "")).strip()
                        if not expression_text:
                            continue
                        expression_lemma = (
                            str(item.get("lemma", expression_text)).strip() or expression_text
                        )
                        expression_type = str(item.get("kind", "expression")).strip() or "expression"
                        start = _to_int(item.get("start"), 0)
                        end = _to_int(item.get("end"), start)
                        expression_key = str(item.get("key", "")).strip() or (
                            f"{expression_lemma.lower()}|{expression_type.lower()}"
                        )
                        match_source = str(item.get("source", "unknown")).strip() or "unknown"
                        wordnet_hit = 1 if bool(item.get("wordnet")) else 0
                        expression_rows.append(
                            ExpressionRow(
                                source=source,
                                part_index=part_index,
                                segment_index=segment_index,
                                expression_index=expression_index,
                                voice=str(voice or ""),
                                expression_text=expression_text,
                                expression_lemma=expression_lemma,
                                expression_type=expression_type,
                                start=start,
                                end=end,
                                expression_key=expression_key,
                                match_source=match_source,
                                wordnet=wordnet_hit,
                                text_sha1=segment_hash,
                            )
                        )
        return token_rows, expression_rows

    def _resolve_crud_table(self, dataset: str) -> tuple[str, str]:
        normalized = (dataset or "").strip().lower()
        if normalized in ("lexemes", "lexeme"):
            return "lexemes", self.lexemes_table
        if normalized in ("occurrences", "token_occurrences", "occurrence"):
            return "occurrences", self.occurrences_table
        if normalized in ("expressions", "expression", "mwe", "mwes", "idioms"):
            return "expressions", self.expressions_table
        raise ValueError("Unsupported dataset. Use lexemes, occurrences, or expressions.")

    def _table_headers(self, connection: sqlite3.Connection, table_name: str) -> list[str]:
        cursor = connection.execute(f'PRAGMA table_info("{table_name}")')
        return [str(row[1]) for row in cursor.fetchall() if len(row) > 1]

    def _normalize_row_identifier(self, table_key: str, row_id: str) -> Any:
        raw = str(row_id or "").strip()
        if table_key == "lexemes":
            if not raw:
                raise ValueError("row_id is required (dedup_key for lexemes).")
            return raw
        if not raw:
            raise ValueError("row_id is required (numeric id).")
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError("row_id must be an integer for this dataset.") from exc

    def _normalize_insert_payload(self, table_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if table_key == "lexemes":
            lemma = str(payload.get("lemma", "")).strip()
            upos = str(payload.get("upos", "X")).strip().upper() or "X"
            dedup_key = str(payload.get("dedup_key", "")).strip() or _build_key(lemma, upos)
            if not dedup_key:
                raise ValueError("dedup_key or (lemma + upos) is required.")
            return {
                "dedup_key": dedup_key,
                "lemma": lemma,
                "upos": upos,
                "feats_json": _normalize_json_text(payload.get("feats_json", "{}")),
            }

        if table_key == "occurrences":
            token_text = str(payload.get("token_text", payload.get("token", ""))).strip()
            lemma = str(payload.get("lemma", token_text)).strip() or token_text
            upos = str(payload.get("upos", "X")).strip().upper() or "X"
            dedup_key = str(payload.get("dedup_key", "")).strip() or _build_key(lemma, upos)
            text_sha1 = str(payload.get("text_sha1", "")).strip() or _sha1_text(token_text)
            return {
                "source": str(payload.get("source", "manual")).strip() or "manual",
                "part_index": _to_int(payload.get("part_index"), 0),
                "segment_index": _to_int(payload.get("segment_index"), 0),
                "token_index": _to_int(payload.get("token_index"), 0),
                "voice": str(payload.get("voice", "")).strip(),
                "token_text": token_text,
                "lemma": lemma,
                "upos": upos,
                "feats_json": _normalize_json_text(payload.get("feats_json", "{}")),
                "start_offset": _to_int(payload.get("start_offset"), 0),
                "end_offset": _to_int(payload.get("end_offset"), 0),
                "dedup_key": dedup_key,
                "text_sha1": text_sha1,
            }

        expression_text = str(payload.get("expression_text", "")).strip()
        expression_lemma = (
            str(payload.get("expression_lemma", expression_text)).strip() or expression_text
        )
        expression_type = str(payload.get("expression_type", "expression")).strip() or "expression"
        expression_key = str(payload.get("expression_key", "")).strip() or (
            f"{expression_lemma.lower()}|{expression_type.lower()}"
        )
        text_sha1 = str(payload.get("text_sha1", "")).strip() or _sha1_text(expression_text)
        return {
            "source": str(payload.get("source", "manual")).strip() or "manual",
            "part_index": _to_int(payload.get("part_index"), 0),
            "segment_index": _to_int(payload.get("segment_index"), 0),
            "expression_index": _to_int(payload.get("expression_index"), 0),
            "voice": str(payload.get("voice", "")).strip(),
            "expression_text": expression_text,
            "expression_lemma": expression_lemma,
            "expression_type": expression_type,
            "start_offset": _to_int(payload.get("start_offset"), 0),
            "end_offset": _to_int(payload.get("end_offset"), 0),
            "expression_key": expression_key,
            "match_source": str(payload.get("match_source", "manual")).strip() or "manual",
            "wordnet_hit": 1 if bool(payload.get("wordnet_hit")) else 0,
            "text_sha1": text_sha1,
        }

    def _normalize_update_payload(self, table_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if table_key == "lexemes":
            result: dict[str, Any] = {}
            if "lemma" in payload:
                result["lemma"] = str(payload.get("lemma", "")).strip()
            if "upos" in payload:
                result["upos"] = str(payload.get("upos", "X")).strip().upper() or "X"
            if "feats_json" in payload:
                result["feats_json"] = _normalize_json_text(payload.get("feats_json", "{}"))
            return result

        if table_key == "occurrences":
            result = {}
            if "source" in payload:
                result["source"] = str(payload.get("source", "manual")).strip() or "manual"
            if "part_index" in payload:
                result["part_index"] = _to_int(payload.get("part_index"), 0)
            if "segment_index" in payload:
                result["segment_index"] = _to_int(payload.get("segment_index"), 0)
            if "token_index" in payload:
                result["token_index"] = _to_int(payload.get("token_index"), 0)
            if "voice" in payload:
                result["voice"] = str(payload.get("voice", "")).strip()
            if "token_text" in payload or "token" in payload:
                token_text = str(
                    payload.get("token_text", payload.get("token", ""))
                ).strip()
                result["token_text"] = token_text
            if "lemma" in payload:
                result["lemma"] = str(payload.get("lemma", "")).strip()
            if "upos" in payload:
                result["upos"] = str(payload.get("upos", "X")).strip().upper() or "X"
            if "feats_json" in payload:
                result["feats_json"] = _normalize_json_text(payload.get("feats_json", "{}"))
            if "start_offset" in payload:
                result["start_offset"] = _to_int(payload.get("start_offset"), 0)
            if "end_offset" in payload:
                result["end_offset"] = _to_int(payload.get("end_offset"), 0)
            if "dedup_key" in payload:
                result["dedup_key"] = str(payload.get("dedup_key", "")).strip()
            if "text_sha1" in payload:
                result["text_sha1"] = str(payload.get("text_sha1", "")).strip()
            return result

        result = {}
        if "source" in payload:
            result["source"] = str(payload.get("source", "manual")).strip() or "manual"
        if "part_index" in payload:
            result["part_index"] = _to_int(payload.get("part_index"), 0)
        if "segment_index" in payload:
            result["segment_index"] = _to_int(payload.get("segment_index"), 0)
        if "expression_index" in payload:
            result["expression_index"] = _to_int(payload.get("expression_index"), 0)
        if "voice" in payload:
            result["voice"] = str(payload.get("voice", "")).strip()
        if "expression_text" in payload:
            result["expression_text"] = str(payload.get("expression_text", "")).strip()
        if "expression_lemma" in payload:
            result["expression_lemma"] = str(payload.get("expression_lemma", "")).strip()
        if "expression_type" in payload:
            result["expression_type"] = (
                str(payload.get("expression_type", "expression")).strip() or "expression"
            )
        if "start_offset" in payload:
            result["start_offset"] = _to_int(payload.get("start_offset"), 0)
        if "end_offset" in payload:
            result["end_offset"] = _to_int(payload.get("end_offset"), 0)
        if "expression_key" in payload:
            result["expression_key"] = str(payload.get("expression_key", "")).strip()
        if "match_source" in payload:
            result["match_source"] = str(payload.get("match_source", "manual")).strip() or "manual"
        if "wordnet_hit" in payload:
            result["wordnet_hit"] = 1 if bool(payload.get("wordnet_hit")) else 0
        if "text_sha1" in payload:
            result["text_sha1"] = str(payload.get("text_sha1", "")).strip()
        return result

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
        connection = None
        try:
            connection = sqlite3.connect(self.db_path)
            cursor = connection.execute(f'SELECT * FROM "{table_name}" ORDER BY ROWID')
            headers = [description[0] for description in cursor.description or []]
            rows = [[_stringify_cell(item) for item in row] for row in cursor.fetchall()]
            return headers, rows
        finally:
            if connection is not None:
                connection.close()

    def _build_pos_table_dataset(self) -> tuple[list[str], list[list[str]]]:
        connection = None
        try:
            connection = sqlite3.connect(self.db_path)
            cursor = connection.execute(
                f'SELECT "upos", "lemma" FROM "{self.lexemes_table}" ORDER BY "upos", "lemma" COLLATE NOCASE'
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
