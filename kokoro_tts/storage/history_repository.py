"""Storage for managing saved audio history."""
from __future__ import annotations

import os


class HistoryRepository:
    def __init__(self, output_dir_abs: str, logger) -> None:
        self.output_dir_abs = output_dir_abs
        self.logger = logger

    def delete_paths(self, paths: list[str]) -> int:
        deleted = 0
        for path in paths:
            if not path:
                continue
            try:
                abs_path = os.path.abspath(path)
                if os.path.commonpath([abs_path, self.output_dir_abs]) != self.output_dir_abs:
                    self.logger.warning("Skip delete outside output dir: %s", path)
                    continue
                if os.path.isfile(abs_path):
                    os.remove(abs_path)
                    deleted += 1
            except Exception:
                self.logger.exception("Failed to delete history file: %s", path)
        return deleted
