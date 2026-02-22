"""Persistent pronunciation dictionary rules."""

from __future__ import annotations

import copy
import json
import logging
import os
import threading

from ..domain.voice import LANGUAGE_LABELS, normalize_lang_code

logger = logging.getLogger(__name__)


class PronunciationRepository:
    def __init__(self, path: str, logger_instance=None) -> None:
        self.path = str(path or "").strip()
        self.logger = logger_instance or logger
        self._lock = threading.Lock()
        self._cached_rules: dict[str, dict[str, str]] = {}
        self._cached_mtime: float | None = None
        self._loaded = False

    def load_rules(self) -> dict[str, dict[str, str]]:
        if not self.path:
            return {}
        with self._lock:
            self._refresh_cache_locked()
            return copy.deepcopy(self._cached_rules)

    def load_rules_shared(self) -> tuple[dict[str, dict[str, str]], bool]:
        """
        Return an internal cached rules mapping and changed flag.

        The returned mapping is shared and must be treated as read-only.
        """
        if not self.path:
            return {}, False
        with self._lock:
            changed = self._refresh_cache_locked()
            return self._cached_rules, changed

    def parse_rules_json(self, raw_json: str) -> dict[str, dict[str, str]]:
        raw_text = str(raw_json or "").strip()
        if not raw_text:
            return {}
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc.msg}") from exc
        return self._normalize_rules(payload)

    def load_rules_from_file(self, file_path: str) -> dict[str, dict[str, str]]:
        target = str(file_path or "").strip()
        if not target:
            raise ValueError("No JSON file selected.")
        try:
            with open(target, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc.msg}") from exc
        except FileNotFoundError as exc:
            raise ValueError("JSON file not found.") from exc
        except OSError as exc:
            raise ValueError(f"Failed to read JSON file: {exc}") from exc
        return self._normalize_rules(payload)

    def save_rules(self, rules: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        if not self.path:
            raise ValueError("Pronunciation rules path is not configured.")
        normalized = self._normalize_rules(rules)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(normalized, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            try:
                self._cached_mtime = os.path.getmtime(self.path)
            except OSError:
                self._cached_mtime = None
            self._cached_rules = normalized
            self._loaded = True
        return copy.deepcopy(normalized)

    def to_pretty_json(self, rules: dict[str, dict[str, str]] | None = None) -> str:
        payload = self._normalize_rules(rules if rules is not None else self.load_rules())
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    def _normalize_rules(self, raw_rules: object) -> dict[str, dict[str, str]]:
        if raw_rules in (None, ""):
            return {}
        if not isinstance(raw_rules, dict):
            raise ValueError("Rules JSON must be an object mapping language codes to rule maps.")
        normalized: dict[str, dict[str, str]] = {}
        for raw_lang, raw_entries in raw_rules.items():
            lang = self._normalize_language_key(raw_lang)
            if not lang:
                self.logger.warning(
                    "Skipping pronunciation rules for unknown language: %s", raw_lang
                )
                continue
            if not isinstance(raw_entries, dict):
                self.logger.warning(
                    "Skipping pronunciation rules for language %s: expected object map",
                    lang,
                )
                continue
            entries: dict[str, str] = {}
            for raw_word, raw_phoneme in raw_entries.items():
                word = str(raw_word or "").strip()
                phoneme = str(raw_phoneme or "").strip()
                if not word or not phoneme:
                    continue
                entries[word] = phoneme
            if entries:
                normalized.setdefault(lang, {}).update(entries)
        return normalized

    def _normalize_language_key(self, raw_lang: object) -> str:
        value = str(raw_lang or "").strip().lower()
        if not value:
            return ""
        normalized = normalize_lang_code(value, default=value)
        if normalized in LANGUAGE_LABELS:
            return normalized
        return ""

    def _refresh_cache_locked(self) -> bool:
        try:
            mtime = os.path.getmtime(self.path)
        except FileNotFoundError:
            changed = self._loaded and (self._cached_mtime is not None or bool(self._cached_rules))
            self._cached_rules = {}
            self._cached_mtime = None
            self._loaded = True
            return changed
        except Exception:
            self.logger.exception(
                "Failed to inspect pronunciation rules file: %s",
                self.path,
            )
            return False

        if self._loaded and self._cached_mtime == mtime:
            return False

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            normalized = self._normalize_rules(payload)
        except Exception:
            self.logger.exception("Failed to load pronunciation rules: %s", self.path)
            return False

        changed = (not self._loaded) or normalized != self._cached_rules
        self._cached_rules = normalized
        self._cached_mtime = mtime
        self._loaded = True
        return changed
