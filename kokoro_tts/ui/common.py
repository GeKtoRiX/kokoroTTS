"""UI-neutral helpers shared by desktop UI implementations."""
from __future__ import annotations

import inspect
import json
from typing import Any, Sequence

from ..domain.morphology_datasets import (
    morphology_primary_key,
    normalize_morphology_dataset,
)

APP_TITLE = "KokoroTTS"

TOKEN_NOTE = (
    "\nCustomize pronunciation with Markdown link syntax and /slashes/ like "
    "`[Kokoro](/k o k o r o/)`\n\n"
    "To adjust intonation, try punctuation `;:,.!?\\\"()` and stress markers.\n\n"
    "Lower stress: `[1 level](-1)` or `[2 levels](-2)`\n\n"
    "Raise stress: `[or](+2)` (or +1 where supported)\n"
)

DIALOGUE_NOTE = (
    "\nUse [voice=af_heart] to switch speakers inside the text.\n"
    "Use [style=neutral|narrator|energetic] to switch style per segment.\n"
    "Use [pause=0.35] (or [pause=350ms], [pause=default]) to control pauses per segment.\n"
    "Mix voices with commas: [voice=af_heart,am_michael].\n"
)

RUNTIME_MODE_DEFAULT = "default"
RUNTIME_MODE_TTS_MORPH = "tts_morph"
RUNTIME_MODE_FULL = "full"
RUNTIME_MODE_CHOICES: list[tuple[str, str]] = [
    ("Default", RUNTIME_MODE_DEFAULT),
    ("TTS + Morphology", RUNTIME_MODE_TTS_MORPH),
    ("Full", RUNTIME_MODE_FULL),
]


def tts_only_mode_status_text(enabled: bool) -> str:
    if enabled:
        return "TTS-only mode is ON: Morphology DB and LLM requests are disabled."
    return "TTS-only mode is OFF: Morphology DB and LLM requests are enabled."


def llm_only_mode_status_text(enabled: bool, *, tts_only_enabled: bool) -> str:
    if tts_only_enabled:
        return "TTS + Morphology mode is overridden by TTS-only mode."
    if enabled:
        return "TTS + Morphology mode is ON: LLM requests are disabled, Morphology DB stays enabled."
    return "TTS + Morphology mode is OFF: LLM requests are enabled."


def runtime_mode_from_flags(*, tts_only_enabled: bool, llm_only_enabled: bool) -> str:
    if tts_only_enabled:
        return RUNTIME_MODE_DEFAULT
    if llm_only_enabled:
        return RUNTIME_MODE_TTS_MORPH
    return RUNTIME_MODE_FULL


def normalize_runtime_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == RUNTIME_MODE_TTS_MORPH:
        return RUNTIME_MODE_TTS_MORPH
    if normalized == RUNTIME_MODE_DEFAULT:
        return RUNTIME_MODE_DEFAULT
    return RUNTIME_MODE_FULL


def runtime_mode_status_text(mode_value: Any) -> str:
    mode = normalize_runtime_mode(mode_value)
    if mode == RUNTIME_MODE_DEFAULT:
        return "Default mode is active: TTS only."
    if mode == RUNTIME_MODE_TTS_MORPH:
        return "TTS + Morphology mode is active."
    return "Full mode is active."


def runtime_mode_tab_visibility(mode_value: Any) -> tuple[bool, bool]:
    mode = normalize_runtime_mode(mode_value)
    if mode == RUNTIME_MODE_DEFAULT:
        return False, False
    if mode == RUNTIME_MODE_TTS_MORPH:
        return False, True
    return True, True


def normalize_morph_dataset(dataset: Any) -> str:
    return normalize_morphology_dataset(dataset)


def extract_morph_headers(table_update: Any) -> list[str]:
    if not isinstance(table_update, dict):
        return []
    headers = table_update.get("headers")
    if not isinstance(headers, list):
        return []
    return [str(header) for header in headers]


def supports_export_format_arg(callback: Any) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return True

    parameters = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return True
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return True
    positional = [
        param
        for param in parameters
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return True
    return any(param.name == "file_format" for param in parameters)


def coerce_int_value(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    raw = str(value or "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def coerce_wordnet_hit(value: Any) -> int:
    raw = str(value or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return 0
    if raw in ("1", "true", "yes", "on"):
        return 1
    return 1 if coerce_int_value(value, 0) != 0 else 0


def coerce_morph_cell_value(dataset: str, column: str, value: Any) -> Any:
    numeric_columns = {
        "occurrences": {"part_index", "segment_index", "token_index", "start_offset", "end_offset"},
        "expressions": {"part_index", "segment_index", "expression_index", "start_offset", "end_offset"},
        "reviews": {
            "part_index",
            "segment_index",
            "token_index",
            "start_offset",
            "end_offset",
            "is_match",
            "attempt_count",
        },
    }
    if dataset == "expressions" and column == "wordnet_hit":
        return coerce_wordnet_hit(value)
    if column in numeric_columns.get(dataset, set()):
        return coerce_int_value(value, 0)
    if value is None:
        return ""
    return str(value)


def build_morph_update_payload(
    dataset: Any,
    headers: Sequence[Any],
    row_value: Sequence[Any] | None,
) -> tuple[str, dict[str, Any]]:
    if not row_value:
        raise ValueError("No row selected.")

    normalized_dataset = normalize_morph_dataset(dataset)
    normalized_headers = [str(header) for header in (headers or [])]
    if not normalized_headers:
        raise ValueError("No table headers available.")

    primary_key = morphology_primary_key(normalized_dataset)
    if primary_key not in normalized_headers:
        raise ValueError(f"Primary key column '{primary_key}' is missing.")

    row_items = list(row_value)
    key_index = normalized_headers.index(primary_key)
    if key_index >= len(row_items):
        raise ValueError("Selected row does not include a primary key value.")

    selected_row_id = str(row_items[key_index] or "").strip()
    if not selected_row_id:
        raise ValueError("Selected row id/key is empty.")

    payload: dict[str, Any] = {}
    for index, column in enumerate(normalized_headers):
        if index >= len(row_items):
            continue
        if column in ("id", "created_at"):
            continue
        if normalized_dataset == "lexemes" and column == "dedup_key":
            continue
        payload[column] = coerce_morph_cell_value(normalized_dataset, column, row_items[index])

    return selected_row_id, payload


def resolve_morph_delete_confirmation(
    selected_row_id: Any,
    armed_row_id: Any,
) -> tuple[bool, str, str]:
    selected = str(selected_row_id or "").strip()
    armed = str(armed_row_id or "").strip()
    if not selected:
        return False, "", "Select a row before deleting."
    if armed != selected:
        return (
            False,
            selected,
            f"Press Delete row again to confirm deleting id/key={selected}.",
        )
    return True, "", ""


def to_table_update(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
    return {
        "headers": [str(header) for header in headers],
        "value": [[cell for cell in row] for row in rows],
    }


def pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
