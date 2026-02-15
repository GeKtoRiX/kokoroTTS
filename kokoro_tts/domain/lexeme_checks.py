"""Helpers for validating English lexeme splitting and optional LM verification."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

from ..integrations.lm_studio import PosVerifyRequest, verify_pos_with_context
from .morphology import UPOS_VALUES, analyze_english_text

Analyzer = Callable[[str], dict[str, object]]
Verifier = Callable[[PosVerifyRequest], dict[str, object]]


@dataclass(frozen=True)
class LmVerifySettings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    temperature: float
    max_tokens: int


def load_lm_verify_settings_from_env() -> LmVerifySettings:
    return LmVerifySettings(
        base_url=(os.getenv("LM_VERIFY_BASE_URL") or "http://127.0.0.1:1234/v1").strip(),
        api_key=(os.getenv("LM_VERIFY_API_KEY") or "lm-studio").strip(),
        model=(os.getenv("LM_VERIFY_MODEL") or "").strip(),
        timeout_seconds=_to_int(os.getenv("LM_VERIFY_TIMEOUT_SECONDS"), 30),
        temperature=_to_float(os.getenv("LM_VERIFY_TEMPERATURE"), 0.0),
        max_tokens=_to_int(os.getenv("LM_VERIFY_MAX_TOKENS"), 512),
    )


def analyze_and_validate_english_lexemes(
    text: str,
    *,
    analyzer: Analyzer | None = None,
) -> dict[str, object]:
    normalized_text = str(text or "")
    if not normalized_text.strip():
        raise ValueError("Input text is empty.")

    run_analyzer = analyzer or analyze_english_text
    payload = run_analyzer(normalized_text)
    if not isinstance(payload, dict):
        raise ValueError("Analyzer must return a JSON object.")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("Analyzer payload must include items list.")
    if not raw_items:
        raise ValueError("Analyzer produced zero lexemes.")

    _validate_items(normalized_text, raw_items)
    return payload


def verify_english_lexemes_with_lm(
    text: str,
    analysis: dict[str, object],
    *,
    settings: LmVerifySettings,
    verifier: Verifier | None = None,
) -> dict[str, object]:
    model = str(settings.model or "").strip()
    if not model:
        raise ValueError("LM_VERIFY_MODEL is empty.")

    items = analysis.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Analysis must include non-empty items.")
    tokens = _build_verify_tokens(items)
    request = PosVerifyRequest(
        segment_text=str(text or ""),
        tokens=tokens,
        locked_expressions=[],
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=model,
        timeout_seconds=max(0, int(settings.timeout_seconds)),
        temperature=float(settings.temperature),
        max_tokens=max(64, int(settings.max_tokens)),
    )

    run_verifier = verifier or verify_pos_with_context
    response = run_verifier(request)
    if not isinstance(response, dict):
        raise ValueError("LM verifier must return a JSON object.")
    raw_checks = response.get("token_checks")
    if not isinstance(raw_checks, list):
        raise ValueError("LM verifier response must include token_checks list.")

    checks_by_index: dict[int, dict[str, object]] = {}
    for item in raw_checks:
        if not isinstance(item, dict):
            continue
        token_index = _to_int(item.get("token_index"), -1)
        if token_index < 0 or token_index >= len(tokens):
            continue
        checks_by_index[token_index] = item

    if len(checks_by_index) != len(tokens):
        raise ValueError("LM verifier token checks do not cover all token indices.")

    mismatches: list[dict[str, object]] = []
    for index, local in enumerate(tokens):
        check = checks_by_index[index]
        lm_lemma = str(check.get("lemma", local["lemma"])).strip() or str(local["lemma"])
        lm_upos = str(check.get("upos", local["upos"])).strip().upper() or str(local["upos"])
        local_lemma = str(local["lemma"])
        local_upos = str(local["upos"])
        if lm_lemma.lower() != local_lemma.lower() or lm_upos != local_upos:
            mismatches.append(
                {
                    "token_index": index,
                    "token": local["token"],
                    "local_lemma": local_lemma,
                    "lm_lemma": lm_lemma,
                    "local_upos": local_upos,
                    "lm_upos": lm_upos,
                }
            )

    new_expressions = response.get("new_expressions", [])
    if not isinstance(new_expressions, list):
        new_expressions = []
    return {
        "token_count": len(tokens),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "new_expressions": new_expressions,
    }


def _validate_items(text: str, items: list[object]) -> None:
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{index} must be an object.")

        token = str(item.get("token", ""))
        lemma = str(item.get("lemma", "")).strip()
        upos = str(item.get("upos", "")).strip().upper()
        key = str(item.get("key", "")).strip()
        feats = item.get("feats")

        if not token:
            raise ValueError(f"Item #{index} has empty token.")
        if not lemma:
            raise ValueError(f"Item #{index} has empty lemma.")
        if upos not in UPOS_VALUES:
            raise ValueError(f"Item #{index} has invalid upos: {upos}.")
        if upos == "PUNCT":
            raise ValueError(f"Item #{index} must not be punctuation.")
        if not isinstance(feats, dict):
            raise ValueError(f"Item #{index} feats must be a JSON object.")

        start = _to_int(item.get("start"), -1)
        end = _to_int(item.get("end"), -1)
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"Item #{index} has invalid offsets: start={start} end={end}.")
        if text[start:end] != token:
            raise ValueError(
                f"Item #{index} token/offset mismatch: expected '{text[start:end]}' got '{token}'."
            )

        expected_key = _build_key(lemma, upos)
        if key != expected_key:
            raise ValueError(f"Item #{index} has invalid key '{key}', expected '{expected_key}'.")


def _build_verify_tokens(items: list[object]) -> list[dict[str, object]]:
    tokens: list[dict[str, object]] = []
    for index, item in enumerate(items):
        payload = item if isinstance(item, dict) else {}
        tokens.append(
            {
                "token_index": index,
                "token": str(payload.get("token", "")),
                "lemma": str(payload.get("lemma", "")),
                "upos": str(payload.get("upos", "")).upper(),
                "start": _to_int(payload.get("start"), 0),
                "end": _to_int(payload.get("end"), 0),
                "key": str(payload.get("key", "")),
                "feats": payload.get("feats", {}),
            }
        )
    return tokens


def _build_key(lemma: str, upos: str) -> str:
    return f"{lemma.strip().lower()}|{upos.strip().lower()}"


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
