"""Generic helper utilities used across the project."""
from __future__ import annotations

import os
from typing import Optional


def resolve_path(value: str, base_dir: str) -> str:
    """Resolve a path relative to base_dir when value is not absolute."""
    if not os.path.isabs(value):
        return os.path.join(base_dir, value)
    return value


def parse_int_env(
    name: str,
    default: int,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Parse an integer environment variable with optional bounds."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def parse_float_env(
    name: str,
    default: float,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Parse a float environment variable with optional bounds."""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value
