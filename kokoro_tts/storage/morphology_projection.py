"""Shared morphology table projection and formatting helpers."""

from __future__ import annotations

from typing import Any, Sequence

from .morphology_repository import POS_TABLE_COLUMNS


def project_morphology_preview_rows(
    dataset: str,
    table_update: dict[str, Any],
    *,
    max_rows: int = 50,
) -> tuple[list[str], list[list[str]]]:
    headers = [str(item) for item in _extract_headers(table_update)]
    rows = _normalize_table_rows(table_update.get("value", []))

    column_specs: dict[str, list[tuple[str, tuple[str, ...]]]] = {
        "lexemes": [
            ("token_text", ("token_text", "dedup_key", "lemma")),
            ("lemma", ("lemma",)),
            ("upos", ("upos",)),
        ],
        "occurrences": [
            ("token_text", ("token_text", "token")),
            ("lemma", ("lemma",)),
        ],
        "expressions": [
            ("expression_text", ("expression_text",)),
            ("type", ("expression_type", "type")),
        ],
    }

    normalized_dataset = str(dataset or "").strip().lower()
    specs = column_specs.get(normalized_dataset)
    if specs is None:
        preview_headers = headers[:]
        preview_rows = [row[: len(preview_headers)] for row in rows[:max_rows]]
        return preview_headers, preview_rows

    header_index = {name.strip().lower(): idx for idx, name in enumerate(headers)}
    preview_headers = [display for display, _aliases in specs]
    preview_rows: list[list[str]] = []
    for row in rows[:max_rows]:
        projected_row: list[str] = []
        for _display, aliases in specs:
            selected_value = ""
            for alias in aliases:
                idx = header_index.get(alias.lower())
                if idx is None or idx >= len(row):
                    continue
                selected_value = str(row[idx] or "")
                break
            projected_row.append(selected_value)
        preview_rows.append(projected_row)
    return preview_headers, preview_rows


def build_pos_table_preview_from_lexemes(
    table_update: dict[str, Any],
    *,
    max_rows: int = 50,
) -> dict[str, Any]:
    headers = [str(item) for item in _extract_headers(table_update)]
    raw_rows = table_update.get("value", [])
    if not isinstance(raw_rows, list):
        raw_rows = []
    if not headers or not raw_rows:
        return {"headers": ["No data"], "value": []}

    header_index = {name.strip().lower(): idx for idx, name in enumerate(headers)}
    upos_idx = header_index.get("upos")
    lemma_idx = header_index.get("lemma")
    if upos_idx is None or lemma_idx is None:
        return {"headers": ["No data"], "value": []}

    upos_buckets: dict[str, list[str]] = {}
    upos_seen: dict[str, set[str]] = {}
    for row in raw_rows:
        if not isinstance(row, (list, tuple)):
            continue
        upos_text = str(row[upos_idx] if upos_idx < len(row) else "").strip().upper()
        lemma_text = str(row[lemma_idx] if lemma_idx < len(row) else "").strip()
        if not upos_text or not lemma_text:
            continue
        seen_values = upos_seen.setdefault(upos_text, set())
        if lemma_text in seen_values:
            continue
        seen_values.add(lemma_text)
        upos_buckets.setdefault(upos_text, []).append(lemma_text)

    if not upos_buckets:
        return {"headers": ["No data"], "value": []}

    column_pairs = _resolve_pos_column_pairs(upos_buckets)
    preview_headers = [label for label, _upos in column_pairs]
    max_row_count = max((len(upos_buckets.get(upos, [])) for _label, upos in column_pairs), default=0)
    preview_rows: list[list[str]] = []
    for row_idx in range(min(max_row_count, max_rows)):
        values: list[str] = []
        for _label, upos in column_pairs:
            items = upos_buckets.get(upos, [])
            values.append(items[row_idx] if row_idx < len(items) else "")
        preview_rows.append(values)
    return {"headers": preview_headers, "value": preview_rows}


def format_morphology_preview_table(
    headers: Sequence[Any],
    rows: Sequence[Sequence[Any]],
) -> tuple[list[str], list[list[str]]]:
    safe_headers = [str(item) for item in (headers or ["No data"])]
    safe_rows: list[list[str]] = []
    for row in rows:
        row_values = list(row) if isinstance(row, (list, tuple)) else [str(row)]
        if len(row_values) < len(safe_headers):
            row_values.extend([""] * (len(safe_headers) - len(row_values)))
        safe_rows.append([str(item or "") for item in row_values[: len(safe_headers)]])
    return safe_headers, safe_rows


def count_unique_non_empty_cells(rows: Sequence[Sequence[Any]]) -> int:
    unique_values: set[str] = set()
    for row in rows:
        for cell in row:
            text = str(cell or "").strip()
            if text:
                unique_values.add(text)
    return len(unique_values)


def _extract_headers(table_update: Any) -> list[str]:
    if not isinstance(table_update, dict):
        return []
    headers = table_update.get("headers")
    if not isinstance(headers, list):
        return []
    return [str(header) for header in headers]


def _normalize_table_rows(raw_rows: Any) -> list[list[str]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[list[str]] = []
    for row in raw_rows:
        if isinstance(row, (list, tuple)):
            rows.append([str(item or "") for item in row])
        else:
            rows.append([str(row or "")])
    return rows


def _resolve_pos_column_pairs(by_upos: dict[str, list[str]]) -> list[tuple[str, str]]:
    column_pairs = list(POS_TABLE_COLUMNS)
    known_upos = {upos for _, upos in column_pairs}
    extra_upos = sorted([upos for upos in by_upos if upos not in known_upos])
    for upos in extra_upos:
        column_pairs.append((upos, upos))
    return column_pairs
