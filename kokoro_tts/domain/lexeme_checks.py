"""Helpers for validating English lexeme splitting.

Only local lexeme validation is supported.
"""
from __future__ import annotations

from typing import Callable

from .morphology import UPOS_VALUES, analyze_english_text

Analyzer = Callable[[str], dict[str, object]]


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


def _build_key(lemma: str, upos: str) -> str:
    return f"{lemma.strip().lower()}|{upos.strip().lower()}"


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
