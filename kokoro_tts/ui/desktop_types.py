"""Desktop UI interfaces."""

from __future__ import annotations

from typing import Protocol


class DesktopApp(Protocol):
    """Desktop application contract."""

    title: str

    def launch(self) -> None:
        """Start the UI main loop."""
