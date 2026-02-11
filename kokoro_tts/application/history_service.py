"""History management for generated outputs."""
from __future__ import annotations

import os

from ..storage.history_repository import HistoryRepository
from .state import KokoroState


class HistoryService:
    def __init__(
        self,
        history_limit: int,
        repository: HistoryRepository,
        state: KokoroState,
        logger,
    ) -> None:
        self.history_limit = history_limit
        self.repository = repository
        self.state = state
        self.logger = logger

    def update_history(self, history: list[str]) -> list[str]:
        history = list(history or [])
        saved_paths = getattr(self.state, "last_saved_paths", []) or []
        if saved_paths:
            for path in reversed(saved_paths):
                if path and os.path.isfile(path):
                    history.insert(0, path)
        history = history[: self.history_limit]
        return history

    def clear_history(self, history: list[str]) -> list[str]:
        deleted = self.repository.delete_paths(list(history or []))
        self.state.last_saved_paths = []
        self.logger.info("Cleared history: deleted=%s", deleted)
        return []
