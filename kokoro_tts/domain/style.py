"""Style preset helpers for Kokoro runtime tuning."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_STYLE_PRESET = "neutral"
MIN_SPEED = 0.5
MAX_SPEED = 2.0
PIPELINE_STYLE_PARAM_NAMES = ("style_preset", "style", "emotion", "preset")


@dataclass(frozen=True)
class StylePreset:
    key: str
    speed_multiplier: float = 1.0
    pause_multiplier: float = 1.0


STYLE_PRESETS: dict[str, StylePreset] = {
    "neutral": StylePreset("neutral", speed_multiplier=1.0, pause_multiplier=1.0),
    "narrator": StylePreset("narrator", speed_multiplier=0.92, pause_multiplier=1.25),
    "energetic": StylePreset("energetic", speed_multiplier=1.12, pause_multiplier=0.8),
}

STYLE_PRESET_CHOICES: list[tuple[str, str]] = [
    ("Neutral", "neutral"),
    ("Narrator", "narrator"),
    ("Energetic", "energetic"),
]


def normalize_style_preset(value: str | None, default: str = DEFAULT_STYLE_PRESET) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in STYLE_PRESETS:
        return candidate
    if default in STYLE_PRESETS:
        return default
    return DEFAULT_STYLE_PRESET


def resolve_style_runtime(
    style_preset: str | None,
    speed: float,
    pause_seconds: float,
) -> tuple[str, float, float]:
    preset_key = normalize_style_preset(style_preset)
    preset = STYLE_PRESETS[preset_key]
    effective_speed = _clamp_speed(_to_float(speed, 1.0) * preset.speed_multiplier)
    effective_pause = max(0.0, _to_float(pause_seconds, 0.0) * preset.pause_multiplier)
    return preset_key, effective_speed, effective_pause


def _clamp_speed(value: float) -> float:
    if value < MIN_SPEED:
        return MIN_SPEED
    if value > MAX_SPEED:
        return MAX_SPEED
    return value


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
