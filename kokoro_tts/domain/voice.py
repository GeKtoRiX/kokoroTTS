"""Voice catalog and parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import re
from typing import Iterable

from .style import DEFAULT_STYLE_PRESET, STYLE_PRESETS, normalize_style_preset

logger = logging.getLogger("kokoro_app")

VOICE_TAG_RE = re.compile(
    r"\[(?:voice|speaker|spk|mix|voice_mix)\s*=\s*([^\]]+?)\]",
    re.IGNORECASE,
)
DIALOGUE_TAG_RE = re.compile(
    r"\[(voice|speaker|spk|mix|voice_mix|style|pause)\s*=\s*([^\]]+?)\]",
    re.IGNORECASE,
)

DEFAULT_VOICE = "af_heart"
BASE_LANGUAGE_ORDER: list[str] = ["a", "b", "e", "f", "h", "i", "j", "p", "z"]


@dataclass(frozen=True)
class DialogueSegment:
    voice: str
    text: str
    style_preset: str = DEFAULT_STYLE_PRESET
    pause_seconds: float | None = None


LANGUAGE_LABELS = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

LANGUAGE_ALIASES = {
    "en-us": "a",
    "en_gb": "b",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "pt-br": "p",
    "zh": "z",
    "zh-cn": "z",
}

LANGUAGE_CHOICES: list[tuple[str, str]] = [
    ("🇺🇸 American English (a)", "a"),
    ("🇬🇧 British English (b)", "b"),
    ("🇪🇸 Spanish (e)", "e"),
    ("🇫🇷 French (f)", "f"),
    ("🇮🇳 Hindi (h)", "h"),
    ("🇮🇹 Italian (i)", "i"),
    ("🇯🇵 Japanese (j)", "j"),
    ("🇧🇷 Brazilian Portuguese (p)", "p"),
    ("🇨🇳 Mandarin Chinese (z)", "z"),
]

VOICE_ITEMS: list[tuple[str, str]] = [
    ("🇺🇸 🚺 Heart ❤️", "af_heart"),
    ("🇺🇸 🚺 Bella 🔥", "af_bella"),
    ("🇺🇸 🚺 Nicole 🎧", "af_nicole"),
    ("🇺🇸 🚺 Aoede", "af_aoede"),
    ("🇺🇸 🚺 Kore", "af_kore"),
    ("🇺🇸 🚺 Sarah", "af_sarah"),
    ("🇺🇸 🚺 Nova", "af_nova"),
    ("🇺🇸 🚺 Sky", "af_sky"),
    ("🇺🇸 🚺 Alloy", "af_alloy"),
    ("🇺🇸 🚺 Jessica", "af_jessica"),
    ("🇺🇸 🚺 River", "af_river"),
    ("🇺🇸 🚹 Michael", "am_michael"),
    ("🇺🇸 🚹 Fenrir", "am_fenrir"),
    ("🇺🇸 🚹 Puck", "am_puck"),
    ("🇺🇸 🚹 Echo", "am_echo"),
    ("🇺🇸 🚹 Eric", "am_eric"),
    ("🇺🇸 🚹 Liam", "am_liam"),
    ("🇺🇸 🚹 Onyx", "am_onyx"),
    ("🇺🇸 🚹 Santa", "am_santa"),
    ("🇺🇸 🚹 Adam", "am_adam"),
    ("🇬🇧 🚺 Emma", "bf_emma"),
    ("🇬🇧 🚺 Isabella", "bf_isabella"),
    ("🇬🇧 🚺 Alice", "bf_alice"),
    ("🇬🇧 🚺 Lily", "bf_lily"),
    ("🇬🇧 🚹 George", "bm_george"),
    ("🇬🇧 🚹 Fable", "bm_fable"),
    ("🇬🇧 🚹 Lewis", "bm_lewis"),
    ("🇬🇧 🚹 Daniel", "bm_daniel"),
    ("🇪🇸 🚺 Dora", "ef_dora"),
    ("🇪🇸 🚹 Alex", "em_alex"),
    ("🇪🇸 🚹 Santa", "em_santa"),
    ("🇫🇷 🚺 Siwis", "ff_siwis"),
    ("🇮🇳 🚺 Alpha", "hf_alpha"),
    ("🇮🇳 🚺 Beta", "hf_beta"),
    ("🇮🇳 🚹 Omega", "hm_omega"),
    ("🇮🇳 🚹 Psi", "hm_psi"),
    ("🇮🇹 🚺 Sara", "if_sara"),
    ("🇮🇹 🚹 Nicola", "im_nicola"),
    ("🇯🇵 🚺 Alpha", "jf_alpha"),
    ("🇯🇵 🚺 Gongitsune", "jf_gongitsune"),
    ("🇯🇵 🚺 Nezumi", "jf_nezumi"),
    ("🇯🇵 🚺 Tebukuro", "jf_tebukuro"),
    ("🇯🇵 🚹 Kumo", "jm_kumo"),
    ("🇧🇷 🚺 Dora", "pf_dora"),
    ("🇧🇷 🚹 Alex", "pm_alex"),
    ("🇧🇷 🚹 Santa", "pm_santa"),
    ("🇨🇳 🚺 Xiaobei", "zf_xiaobei"),
    ("🇨🇳 🚺 Xiaoni", "zf_xiaoni"),
    ("🇨🇳 🚺 Xiaoxiao", "zf_xiaoxiao"),
    ("🇨🇳 🚺 Xiaoyi", "zf_xiaoyi"),
    ("🇨🇳 🚹 Yunjian", "zm_yunjian"),
    ("🇨🇳 🚹 Yunxi", "zm_yunxi"),
    ("🇨🇳 🚹 Yunxia", "zm_yunxia"),
    ("🇨🇳 🚹 Yunyang", "zm_yunyang"),
]

CHOICES = dict(VOICE_ITEMS)
VALID_VOICE_IDS = set(CHOICES.values())
VOICE_OPTIONS_BY_LANG: dict[str, list[tuple[str, str]]] = {code: [] for code in LANGUAGE_LABELS}
for label, voice_id in VOICE_ITEMS:
    VOICE_OPTIONS_BY_LANG.setdefault(voice_id[0], []).append((label, voice_id))


def _normalized_runtime_lang_code(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return raw[:1]


def _language_choice_label(lang_code: str) -> str:
    return f"{LANGUAGE_LABELS.get(lang_code, lang_code)} ({lang_code})"


def _upsert_language_choice(lang_code: str) -> None:
    if not lang_code:
        return
    label = _language_choice_label(lang_code)
    for index, (_old_label, old_code) in enumerate(LANGUAGE_CHOICES):
        if old_code == lang_code:
            LANGUAGE_CHOICES[index] = (label, lang_code)
            return
    LANGUAGE_CHOICES.append((label, lang_code))


def available_language_codes() -> list[str]:
    ordered: list[str] = []
    for code in BASE_LANGUAGE_ORDER:
        if code in LANGUAGE_LABELS and code not in ordered:
            ordered.append(code)
    for code in LANGUAGE_LABELS:
        if code not in ordered:
            ordered.append(code)
    return ordered


def register_runtime_language(
    lang_code: str,
    *,
    label: str,
    aliases: Iterable[str] | None = None,
) -> str:
    normalized = _normalized_runtime_lang_code(lang_code)
    if not normalized:
        raise ValueError("Language code must be a non-empty string.")
    language_label = str(label or "").strip() or normalized
    LANGUAGE_LABELS[normalized] = language_label
    VOICE_OPTIONS_BY_LANG.setdefault(normalized, [])
    _upsert_language_choice(normalized)
    if aliases:
        for alias in aliases:
            key = str(alias or "").strip().lower()
            if key:
                LANGUAGE_ALIASES[key] = normalized
    return normalized


def register_runtime_voices(
    lang_code: str,
    *,
    voices: Iterable[tuple[str, str]],
    language_label: str,
    aliases: Iterable[str] | None = None,
) -> list[str]:
    normalized_lang = register_runtime_language(
        lang_code,
        label=language_label,
        aliases=aliases,
    )
    registered: list[str] = []
    for raw_label, raw_voice_id in voices:
        voice_id = str(raw_voice_id or "").strip()
        if not voice_id:
            continue
        label = str(raw_label or "").strip() or voice_id
        VALID_VOICE_IDS.add(voice_id)
        CHOICES[label] = voice_id
        option = (label, voice_id)

        existing_global = next(
            (
                index
                for index, (_existing_label, existing_id) in enumerate(VOICE_ITEMS)
                if existing_id == voice_id
            ),
            None,
        )
        if existing_global is None:
            VOICE_ITEMS.append(option)
        else:
            VOICE_ITEMS[existing_global] = option

        lang_options = VOICE_OPTIONS_BY_LANG.setdefault(normalized_lang, [])
        existing_local = next(
            (
                index
                for index, (_existing_label, existing_id) in enumerate(lang_options)
                if existing_id == voice_id
            ),
            None,
        )
        if existing_local is None:
            lang_options.append(option)
        else:
            lang_options[existing_local] = option
        registered.append(voice_id)
    return registered


def normalize_lang_code(value: str | None, default: str = "a") -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    raw = LANGUAGE_ALIASES.get(raw, raw)
    if raw in LANGUAGE_LABELS:
        return raw
    return default


def voice_language(voice_id: str, default: str = "a") -> str:
    if not voice_id:
        return default
    return normalize_lang_code(voice_id[0], default=default)


def get_voice_choices(lang_code: str | None = None) -> list[tuple[str, str]]:
    if not lang_code:
        return list(VOICE_ITEMS)
    lang = normalize_lang_code(lang_code, default=voice_language(DEFAULT_VOICE))
    return list(
        VOICE_OPTIONS_BY_LANG.get(lang) or VOICE_OPTIONS_BY_LANG[voice_language(DEFAULT_VOICE)]
    )


def default_voice_for_lang(lang_code: str) -> str:
    voices = get_voice_choices(lang_code)
    if voices:
        return voices[0][1]
    return DEFAULT_VOICE


def _to_voice_id(value: str | None) -> str | None:
    cleaned = str(value or "").strip().strip('"').strip("'")
    if not cleaned:
        return None
    voice_id = CHOICES.get(cleaned, cleaned)
    if voice_id in VALID_VOICE_IDS:
        return voice_id
    return None


def _normalize_voice_parts(parts: list[str], fallback_voice: str, unknown_label: str) -> str:
    resolved: list[str] = []
    for part in parts:
        voice_id = _to_voice_id(part)
        if voice_id is None:
            logger.warning('Unknown %s "%s"; skipping', unknown_label, part)
            continue
        resolved.append(voice_id)
    if not resolved:
        return fallback_voice
    lang = voice_language(resolved[0], default=voice_language(fallback_voice))
    filtered = [voice for voice in resolved if voice_language(voice) == lang]
    if len(filtered) != len(resolved):
        logger.warning(
            "Mixed voices across languages are not allowed; keeping %s voices only",
            LANGUAGE_LABELS.get(lang, lang),
        )
    if not filtered:
        return fallback_voice
    return ",".join(filtered)


def normalize_voice_input(
    voice: str,
    voice_mix: list[str] | tuple[str, ...] | str | None = None,
) -> str:
    fallback_voice = _to_voice_id(voice) or DEFAULT_VOICE
    raw = voice_mix if voice_mix and str(voice_mix).strip() else voice
    if isinstance(raw, (list, tuple)):
        raw = ",".join(str(item) for item in raw)
    parts = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not parts:
        return fallback_voice
    return _normalize_voice_parts(parts, fallback_voice, unknown_label="voice")


def resolve_voice(
    voice: str,
    voice_mix: list[str] | tuple[str, ...] | str | None,
    mix_enabled: bool,
) -> str:
    if mix_enabled and voice_mix:
        return normalize_voice_input(voice, voice_mix)
    return normalize_voice_input(voice)


def normalize_voice_tag(raw_value: str, default_voice: str) -> str:
    cleaned = str(raw_value or "").strip().strip('"').strip("'")
    if not cleaned:
        return _to_voice_id(default_voice) or DEFAULT_VOICE
    lowered = cleaned.lower()
    if lowered in ("default", "auto"):
        return _to_voice_id(default_voice) or DEFAULT_VOICE
    cleaned = cleaned.replace("+", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    fallback_voice = _to_voice_id(default_voice) or DEFAULT_VOICE
    if not parts:
        return fallback_voice
    return _normalize_voice_parts(parts, fallback_voice, unknown_label="voice tag")


def normalize_style_tag(raw_value: str, current_style: str, default_style: str) -> str:
    cleaned = str(raw_value or "").strip().strip('"').strip("'")
    if not cleaned:
        return current_style
    lowered = cleaned.lower()
    if lowered in ("default", "auto"):
        return normalize_style_preset(default_style)
    if lowered not in STYLE_PRESETS:
        logger.warning('Unknown style tag "%s"; keeping "%s"', cleaned, current_style)
        return current_style
    return normalize_style_preset(lowered, default=current_style)


def normalize_pause_tag(raw_value: str, current_pause_seconds: float | None) -> float | None:
    cleaned = str(raw_value or "").strip().strip('"').strip("'")
    if not cleaned:
        return current_pause_seconds
    lowered = cleaned.lower()
    if lowered in ("default", "auto"):
        return None
    if lowered in ("off", "none"):
        return 0.0
    try:
        if lowered.endswith("ms"):
            value = float(lowered[:-2].strip()) / 1000.0
        elif lowered.endswith("s"):
            value = float(lowered[:-1].strip())
        else:
            value = float(lowered)
    except ValueError:
        logger.warning('Invalid pause tag "%s"; keeping current value', cleaned)
        return current_pause_seconds
    return max(0.0, value)


def parse_dialogue_segments(
    text: str,
    default_voice: str,
    default_style_preset: str = DEFAULT_STYLE_PRESET,
) -> list[DialogueSegment]:
    current_voice = _to_voice_id(default_voice) or DEFAULT_VOICE
    base_style = normalize_style_preset(default_style_preset)
    current_style = base_style
    current_pause: float | None = None
    segments: list[DialogueSegment] = []
    last = 0
    for match in DIALOGUE_TAG_RE.finditer(text):
        start, end = match.span()
        chunk = text[last:start]
        if chunk.strip():
            segments.append(
                DialogueSegment(
                    voice=current_voice,
                    text=chunk.strip(),
                    style_preset=current_style,
                    pause_seconds=current_pause,
                )
            )
        tag_name = match.group(1).strip().lower()
        tag_value = match.group(2)
        if tag_name in ("voice", "speaker", "spk", "mix", "voice_mix"):
            current_voice = normalize_voice_tag(tag_value, current_voice)
        elif tag_name == "style":
            current_style = normalize_style_tag(tag_value, current_style, base_style)
        elif tag_name == "pause":
            current_pause = normalize_pause_tag(tag_value, current_pause)
        last = end
    tail = text[last:]
    if tail.strip():
        segments.append(
            DialogueSegment(
                voice=current_voice,
                text=tail.strip(),
                style_preset=current_style,
                pause_seconds=current_pause,
            )
        )
    return segments


def parse_voice_segments(text: str, default_voice: str) -> list[tuple[str, str]]:
    current_voice = _to_voice_id(default_voice) or DEFAULT_VOICE
    segments: list[tuple[str, str]] = []
    last = 0
    for match in VOICE_TAG_RE.finditer(text):
        start, end = match.span()
        chunk = text[last:start]
        if chunk.strip():
            segments.append((current_voice, chunk.strip()))
        current_voice = normalize_voice_tag(match.group(1), current_voice)
        last = end
    tail = text[last:]
    if tail.strip():
        segments.append((current_voice, tail.strip()))
    return segments


def limit_dialogue_segment_parts(
    parts: list[list[DialogueSegment]],
    char_limit: int | None,
) -> tuple[list[list[DialogueSegment]], bool]:
    if char_limit is None:
        return parts, False
    remaining = char_limit
    limited_parts: list[list[DialogueSegment]] = []
    truncated = False
    for segments in parts:
        limited_segments: list[DialogueSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if remaining <= 0:
                truncated = True
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip()
                truncated = True
            if text:
                limited_segments.append(replace(segment, text=text))
                remaining -= len(text)
        if limited_segments:
            limited_parts.append(limited_segments)
        if remaining <= 0:
            break
    return limited_parts, truncated


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


def summarize_dialogue_voice(parts: list[list[DialogueSegment]], default_voice: str) -> str:
    voices = {segment.voice for segments in parts for segment in segments if segment.voice}
    if not voices:
        return _to_voice_id(default_voice) or DEFAULT_VOICE
    if len(voices) == 1:
        return next(iter(voices))
    return "multi"


def summarize_voice(parts: list[list[tuple[str, str]]], default_voice: str) -> str:
    voices = {voice for segments in parts for voice, _ in segments if voice}
    if not voices:
        return _to_voice_id(default_voice) or DEFAULT_VOICE
    if len(voices) == 1:
        return next(iter(voices))
    return "multi"
