"""Storage for managing saved audio history."""
from __future__ import annotations

from datetime import datetime
import os


class HistoryRepository:
    def __init__(self, output_dir_abs: str, logger) -> None:
        self.output_dir_abs = output_dir_abs
        self.logger = logger

    def _is_within_output_dir(self, path: str) -> bool:
        try:
            return os.path.commonpath([path, self.output_dir_abs]) == self.output_dir_abs
        except ValueError:
            return False

    def delete_paths(self, paths: list[str]) -> int:
        deleted = 0
        for path in paths:
            if not path:
                continue
            try:
                abs_path = os.path.abspath(path)
                if not self._is_within_output_dir(abs_path):
                    self.logger.warning("Skip delete outside output dir: %s", path)
                    continue
                if os.path.isfile(abs_path):
                    os.remove(abs_path)
                    deleted += 1
            except Exception:
                self.logger.exception("Failed to delete history file: %s", path)
        return deleted

    def delete_current_date_files(self) -> int:
        records_dir = os.path.abspath(
            os.path.join(
                self.output_dir_abs,
                datetime.now().strftime("%Y-%m-%d"),
                "records",
            )
        )
        if not self._is_within_output_dir(records_dir):
            self.logger.warning("Skip records cleanup outside output dir: %s", records_dir)
            return 0
        if not os.path.isdir(records_dir):
            return 0

        deleted = 0
        for root, _dirs, files in os.walk(records_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    deleted += 1
                except Exception:
                    self.logger.exception(
                        "Failed to delete generated file during date cleanup: %s",
                        file_path,
                    )

        for root, dirs, files in os.walk(records_dir, topdown=False):
            if dirs or files:
                continue
            try:
                os.rmdir(root)
            except OSError:
                continue
        return deleted
