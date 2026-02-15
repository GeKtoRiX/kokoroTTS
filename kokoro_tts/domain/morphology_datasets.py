"""Canonical dataset helpers for morphology DB UI and API wrappers."""
from __future__ import annotations

from typing import Any

_DATASET_ALIASES: dict[str, str] = {
    "lexemes": "lexemes",
    "lexeme": "lexemes",
    "occurrences": "occurrences",
    "occurrence": "occurrences",
    "token_occurrences": "occurrences",
    "expressions": "expressions",
    "expression": "expressions",
    "mwe": "expressions",
    "mwes": "expressions",
    "idioms": "expressions",
    "reviews": "reviews",
    "review": "reviews",
    "lm_reviews": "reviews",
}


def normalize_morphology_dataset(dataset: Any, *, default: str = "occurrences") -> str:
    """Map dataset aliases to stable canonical keys."""
    normalized_default = _DATASET_ALIASES.get(str(default or "").strip().lower(), "occurrences")
    raw = str(dataset or "").strip().lower()
    if not raw:
        return normalized_default
    return _DATASET_ALIASES.get(raw, normalized_default)


def morphology_primary_key(dataset: Any) -> str:
    """Return the primary key column used by CRUD wrappers."""
    return "dedup_key" if normalize_morphology_dataset(dataset) == "lexemes" else "id"
