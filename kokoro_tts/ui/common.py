"""UI-neutral helpers shared by desktop UI implementations."""

from __future__ import annotations

import inspect
import json
from typing import Any, Sequence

from ..domain.morphology_datasets import (
    normalize_morphology_dataset,
)

APP_TITLE = "KokoroTTS"

TOKEN_NOTE = (
    "\nCustomize pronunciation with Markdown link syntax and /slashes/ like "
    "`[Kokoro](/k o k o r o/)`\n\n"
    'To adjust intonation, try punctuation `;:,.!?\\"()` and stress markers.\n\n'
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
RUNTIME_MODE_CHOICES: list[tuple[str, str]] = [
    ("Default", RUNTIME_MODE_DEFAULT),
    ("TTS + Morphology", RUNTIME_MODE_TTS_MORPH),
]


def tts_only_mode_status_text(enabled: bool) -> str:
    if enabled:
        return "TTS-only mode is ON: Morphology DB writes are disabled."
    return "TTS-only mode is OFF: Morphology DB writes are enabled."


def runtime_mode_from_flags(*, tts_only_enabled: bool) -> str:
    if tts_only_enabled:
        return RUNTIME_MODE_DEFAULT
    return RUNTIME_MODE_TTS_MORPH


def normalize_runtime_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == RUNTIME_MODE_TTS_MORPH:
        return RUNTIME_MODE_TTS_MORPH
    return RUNTIME_MODE_DEFAULT


def runtime_mode_status_text(mode_value: Any) -> str:
    mode = normalize_runtime_mode(mode_value)
    if mode == RUNTIME_MODE_DEFAULT:
        return "Default mode is active: TTS only."
    return "TTS + Morphology mode is active."


def runtime_mode_tab_visibility(mode_value: Any) -> bool:
    mode = normalize_runtime_mode(mode_value)
    return mode == RUNTIME_MODE_TTS_MORPH


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
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return True
    return any(param.name == "file_format" for param in parameters)


def to_table_update(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
    return {
        "headers": [str(header) for header in headers],
        "value": [[cell for cell in row] for row in rows],
    }


def pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
