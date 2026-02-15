"""SQLite persistence for token morphology analysis."""
from __future__ import annotations

import concurrent.futures
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

from ..domain.expressions import extract_english_expressions
from ..domain.morphology import analyze_english_text
from ..domain.splitting import split_sentences

logger = logging.getLogger(__name__)

Analyzer = Callable[[str], dict[str, object]]
ExpressionExtractor = Callable[[str], list[dict[str, object]]]
LmVerifier = Callable[[dict[str, object]], dict[str, object]]

_LATIN_TEXT_RE = re.compile(r"[A-Za-z]")
_VERIFY_EXPRESSION_KINDS = {"phrasal_verb", "idiom"}

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


@dataclass(frozen=True)
class ReviewSegment:
    source: str
    part_index: int
    segment_index: int
    voice: str
    segment_text: str
    text_sha1: str
    tokens: list[MorphRow]
    locked_expressions: list[ExpressionRow]
    offset_base: int = 0


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
        lm_verifier: LmVerifier | None = None,
        lm_verify_enabled: bool = False,
        lm_verify_model: str = "",
        lm_verify_retries: int = 2,
        lm_verify_workers: int = 1,
    ) -> None:
        self.logger = logger_instance or logger
        self.enabled = bool(enabled)
        self.db_path = db_path
        self.table_prefix = _normalize_prefix(table_prefix)
        self.lexemes_table = f"{self.table_prefix}lexemes"
        self.occurrences_table = f"{self.table_prefix}token_occurrences"
        self.expressions_table = f"{self.table_prefix}expressions"
        self.reviews_table = f"{self.table_prefix}reviews"
        self.analyzer = analyzer or analyze_english_text
        self.expression_extractor = expression_extractor or extract_english_expressions
        self.lm_verifier = lm_verifier
        self.lm_verify_enabled = bool(lm_verify_enabled and callable(lm_verifier))
        self.lm_verify_model = str(lm_verify_model or "").strip()
        self.lm_verify_retries = max(0, _to_int(lm_verify_retries, 2))
        self.lm_verify_workers = max(1, _to_int(lm_verify_workers, 1))
        self._db_lock = threading.Lock()
        self._review_futures: list[concurrent.futures.Future[None]] = []
        self._review_futures_lock = threading.Lock()
        self._review_executor: concurrent.futures.ThreadPoolExecutor | None = None
        if self.lm_verify_enabled:
            self._review_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.lm_verify_workers,
                thread_name_prefix="morph-review",
            )
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
            connection = sqlite3.connect(self.db_path)
            try:
                with connection:
                    self._ensure_schema_with_connection(connection)
            finally:
                connection.close()

    def wait_for_pending_reviews(self, timeout: float | None = None) -> None:
        if self._review_executor is None:
            return
        with self._review_futures_lock:
            pending = [future for future in self._review_futures if not future.done()]
            self._review_futures = pending
        if not pending:
            return
        done, not_done = concurrent.futures.wait(pending, timeout=timeout)
        _ = done
        with self._review_futures_lock:
            self._review_futures = [future for future in not_done if not future.done()]

    def _ensure_schema_with_connection(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        connection.execute(self._sql_create_lexemes_table())
        connection.execute(self._sql_create_occurrences_table())
        connection.execute(self._sql_create_expressions_table())
        connection.execute(self._sql_create_reviews_table())
        for statement in self._sql_create_occurrence_indexes():
            connection.execute(statement)
        self._schema_ready = True

    def _track_review_future(self, future: concurrent.futures.Future[None]) -> None:
        with self._review_futures_lock:
            self._review_futures.append(future)

        def _on_done(done_future: concurrent.futures.Future[None]) -> None:
            try:
                done_future.result()
            except Exception:
                self.logger.exception("Morphology review task failed unexpectedly")
            finally:
                with self._review_futures_lock:
                    self._review_futures = [
                        item for item in self._review_futures if item is not done_future
                    ]

        future.add_done_callback(_on_done)

    def _run_review_segment(self, segment: ReviewSegment) -> None:
        if not callable(self.lm_verifier):
            return
        ordered_tokens = sorted(segment.tokens, key=lambda item: item.token_index)
        token_index_map = {local_index: token.token_index for local_index, token in enumerate(ordered_tokens)}
        verify_payload = {
            "segment_text": segment.segment_text,
            "tokens": [
                {
                    "token_index": local_index,
                    "token": token.token,
                    "lemma": token.lemma,
                    "upos": token.upos,
                    "start": token.start - segment.offset_base,
                    "end": token.end - segment.offset_base,
                    "key": token.key,
                    "feats": _json_to_dict(token.feats_json),
                }
                for local_index, token in enumerate(ordered_tokens)
            ],
            "locked_expressions": [
                {
                    "text": item.expression_text,
                    "lemma": item.expression_lemma,
                    "kind": item.expression_type,
                    "start": item.start - segment.offset_base,
                    "end": item.end - segment.offset_base,
                    "key": item.expression_key,
                    "source": item.match_source,
                }
                for item in segment.locked_expressions
            ],
        }
        max_attempts = max(1, self.lm_verify_retries + 1)
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.lm_verifier(verify_payload)
                if not isinstance(response, dict):
                    raise ValueError("LM verifier must return a JSON object.")
                normalized_response = dict(response)
                raw_checks = normalized_response.get("token_checks", [])
                if isinstance(raw_checks, list):
                    mapped_checks: list[dict[str, object]] = []
                    for item in raw_checks:
                        if not isinstance(item, dict):
                            continue
                        local_index = _to_int(item.get("token_index"), -1)
                        if local_index < 0 or local_index >= len(ordered_tokens):
                            continue
                        mapped_item = dict(item)
                        mapped_item["token_index"] = token_index_map[local_index]
                        mapped_checks.append(mapped_item)
                    normalized_response["token_checks"] = mapped_checks
                self._persist_review_success(segment, normalized_response, attempt)
                return
            except Exception as exc:
                last_error = str(exc).strip() or exc.__class__.__name__
                if attempt < max_attempts:
                    self.logger.warning(
                        "LM verify retry: source=%s part=%s segment=%s attempt=%s/%s error=%s",
                        segment.source,
                        segment.part_index,
                        segment.segment_index,
                        attempt,
                        max_attempts,
                        last_error,
                    )
                    continue
                self.logger.warning(
                    "LM verify failed: source=%s part=%s segment=%s attempts=%s error=%s",
                    segment.source,
                    segment.part_index,
                    segment.segment_index,
                    max_attempts,
                    last_error,
                )
                self._persist_review_failure(segment, attempt, last_error)

    def _persist_review_success(
        self,
        segment: ReviewSegment,
        response: dict[str, object],
        attempt: int,
    ) -> None:
        raw_checks = response.get("token_checks", [])
        if not isinstance(raw_checks, list):
            raise ValueError("LM verifier response must include token_checks list.")
        checks_by_index: dict[int, dict[str, object]] = {}
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            token_index = _to_int(item.get("token_index"), -1)
            if token_index < 0:
                continue
            checks_by_index[token_index] = item

        review_rows: list[tuple[object, ...]] = []
        for token in segment.tokens:
            item = checks_by_index.get(token.token_index, {})
            lm_lemma = str(item.get("lemma", token.lemma)).strip() or token.lemma
            lm_upos = str(item.get("upos", token.upos)).strip().upper() or token.upos
            lm_feats = _normalize_json_text(item.get("feats", {}))
            mismatch_fields: list[str] = []
            if lm_lemma.strip().lower() != token.lemma.strip().lower():
                mismatch_fields.append("lemma")
            if lm_upos != token.upos:
                mismatch_fields.append("upos")
            is_match = 0 if mismatch_fields else 1
            review_rows.append(
                (
                    token.source,
                    token.part_index,
                    token.segment_index,
                    token.token_index,
                    token.voice,
                    token.token,
                    token.lemma,
                    token.upos,
                    token.feats_json,
                    lm_lemma,
                    lm_upos,
                    lm_feats,
                    token.start,
                    token.end,
                    token.key,
                    is_match,
                    ",".join(mismatch_fields),
                    "success",
                    "",
                    self.lm_verify_model,
                    attempt,
                    token.text_sha1,
                )
            )

        raw_new_expressions = response.get("new_expressions", [])
        if isinstance(raw_new_expressions, str):
            raw_new_expressions = [raw_new_expressions]
        elif not isinstance(raw_new_expressions, list):
            raw_new_expressions = []
        auto_expression_rows = self._build_auto_expression_rows(
            segment,
            raw_new_expressions,
        )

        with self._db_lock:
            connection = sqlite3.connect(self.db_path)
            try:
                with connection:
                    self._ensure_schema_with_connection(connection)
                    if review_rows:
                        connection.executemany(self._sql_upsert_review(), review_rows)
                    if auto_expression_rows:
                        connection.executemany(
                            self._sql_insert_expression(),
                            _expression_rows(auto_expression_rows),
                        )
            finally:
                connection.close()

    def _persist_review_failure(
        self,
        segment: ReviewSegment,
        attempt: int,
        error_text: str,
    ) -> None:
        snippet = str(error_text or "").strip()[:500]
        review_rows = [
            (
                token.source,
                token.part_index,
                token.segment_index,
                token.token_index,
                token.voice,
                token.token,
                token.lemma,
                token.upos,
                token.feats_json,
                "",
                "",
                "{}",
                token.start,
                token.end,
                token.key,
                0,
                "",
                "failed",
                snippet,
                self.lm_verify_model,
                attempt,
                token.text_sha1,
            )
            for token in segment.tokens
        ]
        with self._db_lock:
            connection = sqlite3.connect(self.db_path)
            try:
                with connection:
                    self._ensure_schema_with_connection(connection)
                    if review_rows:
                        connection.executemany(self._sql_upsert_review(), review_rows)
            finally:
                connection.close()

    def _build_auto_expression_rows(
        self,
        segment: ReviewSegment,
        raw_new_expressions: list[object],
    ) -> list[ExpressionRow]:
        out: list[ExpressionRow] = []
        seen: set[tuple[object, ...]] = set()
        locked_spans = [(item.start, item.end) for item in segment.locked_expressions]
        offset_base = max(0, int(segment.offset_base))

        def _append_candidate(
            *,
            start: int,
            end: int,
            text_hint: str,
            lemma_hint: str = "",
            kind_hint: str = "",
        ) -> None:
            if start < 0 or end <= start or end > len(segment.segment_text):
                return
            span_text = segment.segment_text[start:end]
            if text_hint and _normalize_compare_text(span_text) != _normalize_compare_text(text_hint):
                return
            absolute_start = start + offset_base
            absolute_end = end + offset_base
            if any(
                _spans_overlap(absolute_start, absolute_end, lock_start, lock_end)
                for lock_start, lock_end in locked_spans
            ):
                return

            covered_tokens = [
                token
                for token in segment.tokens
                if _spans_overlap(absolute_start, absolute_end, token.start, token.end)
            ]
            if len(covered_tokens) < 2:
                return

            covered_upos = {token.upos for token in covered_tokens}
            kind = str(kind_hint or "").strip().lower()
            if kind not in _VERIFY_EXPRESSION_KINDS:
                if "VERB" in covered_upos and covered_upos.intersection({"ADP", "PART", "ADV"}):
                    kind = "phrasal_verb"
                elif len(covered_tokens) >= 3:
                    kind = "idiom"
                else:
                    return
            if kind == "phrasal_verb":
                if "VERB" not in covered_upos:
                    return
                if not covered_upos.intersection({"ADP", "PART", "ADV"}):
                    return
            elif kind == "idiom":
                if len(covered_tokens) < 3:
                    return

            lemma = str(lemma_hint or "").strip() or span_text.strip()
            if not lemma:
                return
            expression_key = f"{lemma.lower()}|{kind}"
            dedupe_key = (
                segment.source,
                segment.text_sha1,
                segment.part_index,
                segment.segment_index,
                expression_key,
                absolute_start,
                absolute_end,
            )
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
            out.append(
                ExpressionRow(
                    source=segment.source,
                    part_index=segment.part_index,
                    segment_index=segment.segment_index,
                    expression_index=absolute_start,
                    voice=segment.voice,
                    expression_text=span_text.strip(),
                    expression_lemma=lemma,
                    expression_type=kind,
                    start=absolute_start,
                    end=absolute_end,
                    expression_key=expression_key,
                    match_source="lm_verify_auto",
                    wordnet=0,
                    text_sha1=segment.text_sha1,
                )
            )

        for item in raw_new_expressions:
            if isinstance(item, dict):
                text_value = str(item.get("text", "")).strip()
                if not text_value:
                    continue
                lemma_hint = str(item.get("lemma", "")).strip()
                kind_hint = str(item.get("kind", "")).strip().lower()
                start = _to_int(item.get("start"), -1)
                end = _to_int(item.get("end"), -1)
                if start >= 0 and end > start:
                    _append_candidate(
                        start=start,
                        end=end,
                        text_hint=text_value,
                        lemma_hint=lemma_hint,
                        kind_hint=kind_hint,
                    )
                    continue
                for phrase in _split_expression_phrases(text_value):
                    for match_start, match_end in _find_expression_spans(segment.segment_text, phrase):
                        _append_candidate(
                            start=match_start,
                            end=match_end,
                            text_hint=phrase,
                            lemma_hint=lemma_hint,
                            kind_hint=kind_hint,
                        )
                continue
            if not isinstance(item, str):
                continue
            for phrase in _split_expression_phrases(item):
                for match_start, match_end in _find_expression_spans(segment.segment_text, phrase):
                    _append_candidate(
                        start=match_start,
                        end=match_end,
                        text_hint=phrase,
                    )
        return out

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

        with self._db_lock:
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
        if table_key == "reviews":
            raise ValueError("Dataset reviews is read-only.")
        row = self._normalize_insert_payload(table_key, payload)
        columns = list(row.keys())
        values = [row[column] for column in columns]
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})'

        self.ensure_schema()
        with self._db_lock:
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
        if table_key == "reviews":
            raise ValueError("Dataset reviews is read-only.")
        primary_column = "dedup_key" if table_key == "lexemes" else "id"
        update_payload = self._normalize_update_payload(table_key, payload)
        if not update_payload:
            raise ValueError("No editable fields in payload.")
        row_id_value = self._normalize_row_identifier(table_key, row_id)

        assignments = ", ".join(f'"{column}" = ?' for column in update_payload.keys())
        values = list(update_payload.values()) + [row_id_value]
        sql = f'UPDATE "{table_name}" SET {assignments} WHERE "{primary_column}" = ?'

        self.ensure_schema()
        with self._db_lock:
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
        if table_key == "reviews":
            raise ValueError("Dataset reviews is read-only.")
        primary_column = "dedup_key" if table_key == "lexemes" else "id"
        row_id_value = self._normalize_row_identifier(table_key, row_id)
        sql = f'DELETE FROM "{table_name}" WHERE "{primary_column}" = ?'

        self.ensure_schema()
        with self._db_lock:
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
        rows, expression_rows, review_segments = self._collect_ingest_rows(parts, source)
        if not rows and not expression_rows and not review_segments:
            return
        lexeme_rows = _unique_lexeme_rows(rows)
        occurrence_rows = _occurrence_rows(rows)
        unique_expression_rows = _expression_rows(expression_rows)
        if not lexeme_rows and not occurrence_rows and not unique_expression_rows and not review_segments:
            return
        try:
            self._ensure_parent_dir()
            with self._db_lock:
                connection = sqlite3.connect(self.db_path)
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

        if not self.lm_verify_enabled or self._review_executor is None:
            return
        for segment in review_segments:
            future = self._review_executor.submit(self._run_review_segment, segment)
            self._track_review_future(future)

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
        worksheet.title = (sheet_name[:31] or "Sheet1")
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
    ) -> list[MorphRow]:
        token_rows: list[MorphRow] = []
        analysis = self.analyzer(segment_text)
        items = analysis.get("items", []) if isinstance(analysis, dict) else []
        if not isinstance(items, list):
            return token_rows

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
            token_rows.append(
                MorphRow(
                    source=source,
                    part_index=part_index,
                    segment_index=segment_index,
                    token_index=token_index,
                    voice=voice,
                    token=token,
                    lemma=lemma,
                    upos=upos,
                    feats_json=self._serialize_feats_json(item.get("feats")),
                    start=start,
                    end=end,
                    key=dedup_key,
                    text_sha1=segment_hash,
                )
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
    ) -> list[ExpressionRow]:
        expression_rows: list[ExpressionRow] = []
        expressions = self.expression_extractor(segment_text)
        if not expressions:
            return expression_rows
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
                    voice=voice,
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
        return expression_rows

    def _collect_review_segments(
        self,
        *,
        segment_meta: Sequence[tuple[str, int, int, str, str, str]],
        token_rows: Sequence[MorphRow],
        expression_rows: Sequence[ExpressionRow],
    ) -> list[ReviewSegment]:
        review_segments: list[ReviewSegment] = []
        tokens_by_segment: dict[tuple[str, int, int], list[MorphRow]] = {}
        expressions_by_segment: dict[tuple[str, int, int], list[ExpressionRow]] = {}
        for row in token_rows:
            key = (row.source, row.part_index, row.segment_index)
            tokens_by_segment.setdefault(key, []).append(row)
        for row in expression_rows:
            key = (row.source, row.part_index, row.segment_index)
            expressions_by_segment.setdefault(key, []).append(row)
        for (
            meta_source,
            part_index,
            segment_index,
            voice,
            segment_text,
            segment_hash,
        ) in segment_meta:
            if not _should_verify_segment_text(segment_text):
                continue
            segment_key = (meta_source, part_index, segment_index)
            segment_tokens_all = sorted(
                tokens_by_segment.get(segment_key, []),
                key=lambda row: row.token_index,
            )
            if not segment_tokens_all:
                continue
            segment_expressions_all = sorted(
                expressions_by_segment.get(segment_key, []),
                key=lambda row: (row.start, row.end, row.expression_index),
            )
            sentence_spans = _split_sentence_spans(segment_text)
            for sentence_start, sentence_end, sentence_text in sentence_spans:
                sentence_tokens = [
                    token
                    for token in segment_tokens_all
                    if token.start >= sentence_start and token.end <= sentence_end
                ]
                if not sentence_tokens:
                    continue
                sentence_expressions = [
                    expression
                    for expression in segment_expressions_all
                    if expression.start >= sentence_start and expression.end <= sentence_end
                ]
                review_segments.append(
                    ReviewSegment(
                        source=meta_source,
                        part_index=part_index,
                        segment_index=segment_index,
                        voice=voice,
                        segment_text=sentence_text,
                        text_sha1=segment_hash,
                        tokens=sentence_tokens,
                        locked_expressions=sentence_expressions,
                        offset_base=sentence_start,
                    )
                )
        return review_segments

    def _collect_ingest_rows(
        self,
        parts: Sequence[Sequence[tuple[str, str]]],
        source: str,
    ) -> tuple[list[MorphRow], list[ExpressionRow], list[ReviewSegment]]:
        source = (source or "unknown").strip() or "unknown"
        token_rows: list[MorphRow] = []
        expression_rows: list[ExpressionRow] = []
        collect_review_meta = self.lm_verify_enabled and callable(self.lm_verifier)
        segment_meta: list[tuple[str, int, int, str, str, str]] = []
        for part_index, segments in enumerate(parts):
            for segment_index, (voice, text) in enumerate(segments):
                segment_text = (text or "").strip()
                if not segment_text:
                    continue
                segment_hash = self._segment_hash(segment_text)
                voice_text = str(voice or "")
                if collect_review_meta:
                    segment_meta.append(
                        (
                            source,
                            part_index,
                            segment_index,
                            voice_text,
                            segment_text,
                            segment_hash,
                        )
                    )
                token_rows.extend(
                    self._collect_segment_tokens(
                        source=source,
                        part_index=part_index,
                        segment_index=segment_index,
                        voice=voice_text,
                        segment_text=segment_text,
                        segment_hash=segment_hash,
                    )
                )
                expression_rows.extend(
                    self._collect_segment_expressions(
                        source=source,
                        part_index=part_index,
                        segment_index=segment_index,
                        voice=voice_text,
                        segment_text=segment_text,
                        segment_hash=segment_hash,
                    )
                )
        review_segments: list[ReviewSegment] = []
        if collect_review_meta:
            review_segments = self._collect_review_segments(
                segment_meta=segment_meta,
                token_rows=token_rows,
                expression_rows=expression_rows,
            )
        return token_rows, expression_rows, review_segments

    def _resolve_crud_table(self, dataset: str) -> tuple[str, str]:
        normalized = (dataset or "").strip().lower()
        if normalized in ("lexemes", "lexeme"):
            return "lexemes", self.lexemes_table
        if normalized in ("occurrences", "token_occurrences", "occurrence"):
            return "occurrences", self.occurrences_table
        if normalized in ("expressions", "expression", "mwe", "mwes", "idioms"):
            return "expressions", self.expressions_table
        if normalized in ("reviews", "review", "lm_reviews"):
            return "reviews", self.reviews_table
        raise ValueError("Unsupported dataset. Use lexemes, occurrences, expressions, or reviews.")

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
        if table_key == "reviews":
            raise ValueError("Dataset reviews is read-only.")
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
        if table_key == "reviews":
            raise ValueError("Dataset reviews is read-only.")
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
        if normalized in ("reviews", "review", "lm_reviews"):
            return self.reviews_table
        return self.lexemes_table

    def _query_table_rows(self, table_name: str) -> tuple[list[str], list[list[str]]]:
        with self._db_lock:
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
        with self._db_lock:
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

    def _sql_create_reviews_table(self) -> str:
        return f"""
CREATE TABLE IF NOT EXISTS "{self.reviews_table}" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "source" TEXT NOT NULL,
  "part_index" INTEGER NOT NULL,
  "segment_index" INTEGER NOT NULL,
  "token_index" INTEGER NOT NULL,
  "voice" TEXT NOT NULL,
  "token_text" TEXT NOT NULL,
  "local_lemma" TEXT NOT NULL,
  "local_upos" TEXT NOT NULL,
  "local_feats_json" TEXT NOT NULL,
  "lm_lemma" TEXT NOT NULL,
  "lm_upos" TEXT NOT NULL,
  "lm_feats_json" TEXT NOT NULL,
  "start_offset" INTEGER NOT NULL,
  "end_offset" INTEGER NOT NULL,
  "dedup_key" TEXT NOT NULL,
  "is_match" INTEGER NOT NULL DEFAULT 0,
  "mismatch_fields" TEXT NOT NULL DEFAULT "",
  "status" TEXT NOT NULL,
  "error_text" TEXT NOT NULL DEFAULT "",
  "model" TEXT NOT NULL DEFAULT "",
  "attempt_count" INTEGER NOT NULL DEFAULT 0,
  "text_sha1" TEXT NOT NULL,
  "created_at" TEXT NOT NULL DEFAULT (datetime('now')),
  "updated_at" TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE ("source", "text_sha1", "part_index", "segment_index", "token_index", "dedup_key", "start_offset", "end_offset")
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
            (
                f'CREATE INDEX IF NOT EXISTS "{self.reviews_table}_status_idx" '
                f'ON "{self.reviews_table}" ("status")'
            ),
            (
                f'CREATE INDEX IF NOT EXISTS "{self.reviews_table}_is_match_idx" '
                f'ON "{self.reviews_table}" ("is_match")'
            ),
            (
                f'CREATE INDEX IF NOT EXISTS "{self.reviews_table}_dedup_key_idx" '
                f'ON "{self.reviews_table}" ("dedup_key")'
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

    def _sql_upsert_review(self) -> str:
        return (
            f'INSERT INTO "{self.reviews_table}" '
            '("source", "part_index", "segment_index", "token_index", "voice", '
            '"token_text", "local_lemma", "local_upos", "local_feats_json", "lm_lemma", '
            '"lm_upos", "lm_feats_json", "start_offset", "end_offset", "dedup_key", "is_match", '
            '"mismatch_fields", "status", "error_text", "model", "attempt_count", "text_sha1", "updated_at") VALUES '
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(source, text_sha1, part_index, segment_index, token_index, dedup_key, start_offset, end_offset) DO UPDATE SET "
            'voice=excluded.voice, token_text=excluded.token_text, local_lemma=excluded.local_lemma, '
            'local_upos=excluded.local_upos, local_feats_json=excluded.local_feats_json, '
            'lm_lemma=excluded.lm_lemma, lm_upos=excluded.lm_upos, lm_feats_json=excluded.lm_feats_json, '
            'is_match=excluded.is_match, mismatch_fields=excluded.mismatch_fields, status=excluded.status, '
            'error_text=excluded.error_text, model=excluded.model, attempt_count=excluded.attempt_count, '
            'updated_at=datetime(\'now\')'
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


def _json_to_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    raw = str(value or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _normalize_compare_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _split_sentence_spans(segment_text: str) -> list[tuple[int, int, str]]:
    text = str(segment_text or "")
    stripped = text.strip()
    if not stripped:
        return []
    sentences = split_sentences(text)
    if not sentences:
        return []

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in sentences:
        sentence_text = str(sentence or "").strip()
        if not sentence_text:
            continue
        start = text.find(sentence_text, cursor)
        if start < 0:
            start = text.find(sentence_text)
        if start < 0:
            continue
        end = start + len(sentence_text)
        spans.append((start, end, sentence_text))
        cursor = end
    if spans:
        return spans
    return [(0, len(stripped), stripped)]


def _split_expression_phrases(value: object) -> list[str]:
    raw = str(value or "").replace("\r", "\n").strip()
    if not raw:
        return []
    parts = re.split(r"[\n,;]+", raw)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        candidate = re.sub(r"^\s*(?:[-*]+|\d+[.)-])\s*", "", str(part or ""))
        candidate = candidate.strip().strip("'\"`")
        candidate = " ".join(candidate.split())
        if not candidate or len(candidate.split()) < 2:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _find_expression_spans(segment_text: str, phrase: str) -> list[tuple[int, int]]:
    normalized_phrase = " ".join(str(phrase or "").split())
    if not normalized_phrase:
        return []
    escaped = re.escape(normalized_phrase).replace(r"\ ", r"\s+")
    pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
    text = str(segment_text or "")
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _should_verify_segment_text(value: str) -> bool:
    return bool(_LATIN_TEXT_RE.search(str(value or "")))


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
