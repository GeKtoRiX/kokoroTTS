"""UI notification hooks for application layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class UiHooks:
    warn: Callable[[str], None]
    info: Callable[[str], None]
    error: Callable[[Exception], Exception]
    error_type: type[Exception]
