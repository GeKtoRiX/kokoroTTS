"""Voice catalog and parsing helpers."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("kokoro_app")

VOICE_TAG_RE = re.compile(r"\[(?:voice|speaker|spk|mix|voice_mix)\s*=\s*([^\]]+?)\]", re.IGNORECASE)

CHOICES = {
    "\U0001f1fa\U0001f1f8 \U0001f6ba Heart \u2764\ufe0f": "af_heart",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Bella \U0001f525": "af_bella",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Nicole \U0001f3a7": "af_nicole",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Aoede": "af_aoede",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Kore": "af_kore",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Sarah": "af_sarah",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Nova": "af_nova",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Sky": "af_sky",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Alloy": "af_alloy",
    "\U0001f1fa\U0001f1f8 \U0001f6ba Jessica": "af_jessica",
    "\U0001f1fa\U0001f1f8 \U0001f6ba River": "af_river",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Michael": "am_michael",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Fenrir": "am_fenrir",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Puck": "am_puck",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Echo": "am_echo",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Eric": "am_eric",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Liam": "am_liam",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Onyx": "am_onyx",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Santa": "am_santa",
    "\U0001f1fa\U0001f1f8 \U0001f6b9 Adam": "am_adam",
    "\U0001f1ec\U0001f1e7 \U0001f6ba Emma": "bf_emma",
    "\U0001f1ec\U0001f1e7 \U0001f6ba Isabella": "bf_isabella",
    "\U0001f1ec\U0001f1e7 \U0001f6ba Alice": "bf_alice",
    "\U0001f1ec\U0001f1e7 \U0001f6ba Lily": "bf_lily",
    "\U0001f1ec\U0001f1e7 \U0001f6b9 George": "bm_george",
    "\U0001f1ec\U0001f1e7 \U0001f6b9 Fable": "bm_fable",
    "\U0001f1ec\U0001f1e7 \U0001f6b9 Lewis": "bm_lewis",
    "\U0001f1ec\U0001f1e7 \U0001f6b9 Daniel": "bm_daniel",
}


def normalize_voice_input(voice: str, voice_mix: list[str] | tuple[str, ...] | str | None = None) -> str:
    raw = voice_mix if voice_mix and str(voice_mix).strip() else voice
    if isinstance(raw, (list, tuple)):
        raw = ",".join(str(v) for v in raw)
    if raw is None:
        return "af_heart"
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return "af_heart"
    lang = parts[0][0] if parts[0] else "a"
    mismatched = [p for p in parts[1:] if p and p[0] != lang]
    if mismatched:
        logger.warning(
            "Mixed voices across languages; using pipeline for %s: %s",
            lang,
            parts,
        )
    return ",".join(parts)


def resolve_voice(voice: str, voice_mix: list[str] | tuple[str, ...] | str | None, mix_enabled: bool) -> str:
    if mix_enabled and voice_mix:
        return normalize_voice_input(voice, voice_mix)
    return normalize_voice_input(voice)


def normalize_voice_tag(raw_value: str, default_voice: str) -> str:
    cleaned = str(raw_value or "").strip().strip('"').strip("'")
    if not cleaned:
        return default_voice
    lowered = cleaned.lower()
    if lowered in ("default", "auto"):
        return default_voice
    cleaned = cleaned.replace("+", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        return default_voice
    resolved: list[str] = []
    for part in parts:
        part_id = CHOICES.get(part, part)
        if part_id not in CHOICES.values():
            logger.warning('Unknown voice tag "%s"; using default voice', part)
            continue
        resolved.append(part_id)
    if not resolved:
        return default_voice
    return normalize_voice_input(",".join(resolved))


def parse_voice_segments(text: str, default_voice: str) -> list[tuple[str, str]]:
    current_voice = default_voice
    segments: list[tuple[str, str]] = []
    last = 0
    for match in VOICE_TAG_RE.finditer(text):
        start, end = match.span()
        chunk = text[last:start]
        if chunk.strip():
            segments.append((current_voice, chunk.strip()))
        current_voice = normalize_voice_tag(match.group(1), default_voice)
        last = end
    tail = text[last:]
    if tail.strip():
        segments.append((current_voice, tail.strip()))
    return segments


def limit_dialogue_parts(
    parts: list[list[tuple[str, str]]],
    char_limit: int | None,
) -> tuple[list[list[tuple[str, str]]], bool]:
    if char_limit is None:
        return parts, False
    remaining = char_limit
    limited_parts: list[list[tuple[str, str]]] = []
    truncated = False
    for segments in parts:
        limited_segments: list[tuple[str, str]] = []
        for voice, text in segments:
            text = text.strip()
            if not text:
                continue
            if remaining <= 0:
                truncated = True
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip()
                truncated = True
            if text:
                limited_segments.append((voice, text))
                remaining -= len(text)
        if limited_segments:
            limited_parts.append(limited_segments)
        if remaining <= 0:
            break
    return limited_parts, truncated


def summarize_voice(parts: list[list[tuple[str, str]]], default_voice: str) -> str:
    voices = {voice for segments in parts for voice, _ in segments if voice}
    if not voices:
        return default_voice
    if len(voices) == 1:
        return next(iter(voices))
    return "multi"
