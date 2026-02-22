"""Persistence helpers for desktop audio player state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_AUDIO_PLAYER_SECTION = "audio_player"
_AUDIO_PLAYER_LEGACY_KEYS = {
    "volume",
    "auto_next",
    "last_path",
    "last_position_seconds",
    "queue_index",
}


def coerce_float(
    value: Any,
    *,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if min_value is not None:
        parsed = max(float(min_value), parsed)
    if max_value is not None:
        parsed = min(float(max_value), parsed)
    return float(parsed)


def coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def load_audio_player_state(path: Path, logger) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read audio player state")
        return {}
    if not isinstance(payload, dict):
        return {}
    section_payload = payload.get(_AUDIO_PLAYER_SECTION)
    if isinstance(section_payload, dict):
        return section_payload
    if _AUDIO_PLAYER_LEGACY_KEYS.intersection(payload.keys()):
        # Backward-compatible read for older dedicated audio-player state files.
        return payload
    return {}


def save_audio_player_state(path: Path, payload: Mapping[str, Any], logger) -> None:
    try:
        app_state_payload: dict[str, Any] = {}
        if path.is_file():
            try:
                existing_payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing_payload, dict):
                    app_state_payload = dict(existing_payload)
            except Exception:
                logger.exception("Failed to read audio player state")
        path.parent.mkdir(parents=True, exist_ok=True)
        app_state_payload[_AUDIO_PLAYER_SECTION] = dict(payload)
        path.write_text(
            json.dumps(app_state_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed to save audio player state")
