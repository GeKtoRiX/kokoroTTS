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
        deleted_from_history = self.repository.delete_paths(list(history or []))
        deleted_from_today = self.repository.delete_current_date_files()
        self.state.last_saved_paths = []
        self.logger.info(
            "Cleared history: deleted_history=%s deleted_today=%s",
            deleted_from_history,
            deleted_from_today,
        )
        return []

    def remove_selected_history(self, history: list[str], selected_indices: list[int]) -> list[str]:
        history_list = list(history or [])
        normalized_set: set[int] = set()
        for raw_index in selected_indices or []:
            try:
                index = int(raw_index)
            except Exception:
                continue
            if 0 <= index < len(history_list):
                normalized_set.add(index)
        normalized = sorted(normalized_set)
        if not normalized:
            return history_list
        selected_paths = [history_list[index] for index in normalized]
        selected_set = set(selected_paths)
        deleted_count = self.repository.delete_paths(selected_paths)
        updated = [
            value for index, value in enumerate(history_list) if index not in set(normalized)
        ]
        last_saved = list(getattr(self.state, "last_saved_paths", []) or [])
        self.state.last_saved_paths = [path for path in last_saved if path not in selected_set]
        self.logger.info(
            "Deleted selected history items: selected=%s deleted=%s",
            len(normalized),
            deleted_count,
        )
        return updated[: self.history_limit]
