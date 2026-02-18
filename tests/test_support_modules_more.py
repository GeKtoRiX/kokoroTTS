import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kokoro_tts.storage.history_repository import HistoryRepository
from kokoro_tts.storage.pronunciation_repository import PronunciationRepository
from kokoro_tts.ui import common as ui_common
from kokoro_tts.utils import parse_float_env


class _Logger:
    def __init__(self):
        self.warnings = []
        self.exceptions = []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)


def test_ui_common_helpers_cover_export_signature_and_modes():
    assert "ON" in ui_common.tts_only_mode_status_text(True)
    assert "OFF" in ui_common.tts_only_mode_status_text(False)
    assert ui_common.runtime_mode_from_flags(tts_only_enabled=True) == ui_common.RUNTIME_MODE_DEFAULT
    assert ui_common.runtime_mode_from_flags(tts_only_enabled=False) == ui_common.RUNTIME_MODE_TTS_MORPH
    assert ui_common.normalize_runtime_mode("TTS_MORPH") == ui_common.RUNTIME_MODE_TTS_MORPH
    assert ui_common.normalize_runtime_mode("  ") == ui_common.RUNTIME_MODE_DEFAULT
    assert "Default mode" in ui_common.runtime_mode_status_text("default")
    assert "Morphology mode" in ui_common.runtime_mode_status_text("tts_morph")
    assert ui_common.runtime_mode_tab_visibility("tts_morph") is True
    assert ui_common.runtime_mode_tab_visibility("anything") is False
    assert ui_common.extract_morph_headers("bad") == []
    assert ui_common.extract_morph_headers({"headers": "bad"}) == []
    assert ui_common.extract_morph_headers({"headers": ["id", 2]}) == ["id", "2"]

    def _cb_plain(dataset):
        return dataset

    def _cb_two_args(dataset, file_format):
        return dataset, file_format

    def _cb_varargs(*args):
        return args

    def _cb_kwargs(**kwargs):
        return kwargs

    assert ui_common.supports_export_format_arg(_cb_varargs) is True
    assert ui_common.supports_export_format_arg(_cb_kwargs) is True
    assert ui_common.supports_export_format_arg(_cb_two_args) is True
    assert ui_common.supports_export_format_arg(_cb_plain) is False
    assert ui_common.supports_export_format_arg(123) is True

    table = ui_common.to_table_update(["id"], [[1], [2]])
    assert table == {"headers": ["id"], "value": [[1], [2]]}
    payload = ui_common.pretty_json({"b": 2, "a": 1})
    assert json.loads(payload) == {"a": 1, "b": 2}


def test_parse_float_env_applies_defaults_and_bounds(monkeypatch):
    monkeypatch.setenv("FLOAT_ENV_TEST", "bad")
    assert parse_float_env("FLOAT_ENV_TEST", 0.7, min_value=0.1, max_value=1.0) == 0.7

    monkeypatch.setenv("FLOAT_ENV_TEST", "2.5")
    assert parse_float_env("FLOAT_ENV_TEST", 0.7, min_value=0.1, max_value=1.0) == 1.0

    monkeypatch.setenv("FLOAT_ENV_TEST", "-2.5")
    assert parse_float_env("FLOAT_ENV_TEST", 0.7, min_value=0.1, max_value=1.0) == 0.1


def test_history_repository_error_paths(monkeypatch, tmp_path: Path):
    logger = _Logger()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    repository = HistoryRepository(str(output_dir), logger)

    monkeypatch.setattr(
        "kokoro_tts.storage.history_repository.os.path.commonpath",
        lambda _values: (_ for _ in ()).throw(ValueError("bad-path")),
    )
    assert repository._is_within_output_dir(str(output_dir / "a.wav")) is False

    file_path = output_dir / "a.wav"
    file_path.write_bytes(b"x")
    monkeypatch.setattr(repository, "_is_within_output_dir", lambda _path: True)
    monkeypatch.setattr(
        "kokoro_tts.storage.history_repository.os.remove",
        lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
    )
    deleted = repository.delete_paths([str(file_path)])
    assert deleted == 0
    assert logger.exceptions


def test_history_repository_date_cleanup_guards(monkeypatch, tmp_path: Path):
    logger = _Logger()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    repository = HistoryRepository(str(output_dir), logger)

    monkeypatch.setattr(repository, "_is_within_output_dir", lambda _path: False)
    assert repository.delete_current_date_files() == 0
    assert logger.warnings

    monkeypatch.setattr(repository, "_is_within_output_dir", lambda _path: True)
    assert repository.delete_current_date_files() == 0

    date_dir = output_dir / "2000-01-01" / "records"
    date_dir.mkdir(parents=True)
    nested = date_dir / "nested"
    nested.mkdir()
    (date_dir / "sample.wav").write_bytes(b"x")

    monkeypatch.setattr(
        "kokoro_tts.storage.history_repository.datetime",
        SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _fmt: "2000-01-01")),
    )
    monkeypatch.setattr(
        "kokoro_tts.storage.history_repository.os.remove",
        lambda _path: (_ for _ in ()).throw(OSError("busy")),
    )
    monkeypatch.setattr(
        "kokoro_tts.storage.history_repository.os.rmdir",
        lambda _path: (_ for _ in ()).throw(OSError("locked")),
    )
    assert repository.delete_current_date_files() == 0
    assert logger.exceptions


def test_pronunciation_repository_extended_error_and_cache_paths(monkeypatch, tmp_path: Path):
    logger = _Logger()
    repo_no_path = PronunciationRepository("", logger_instance=logger)
    assert repo_no_path.load_rules() == {}
    with pytest.raises(ValueError, match="not configured"):
        repo_no_path.save_rules({"a": {"x": "y"}})

    path = tmp_path / "rules.json"
    repo = PronunciationRepository(str(path), logger_instance=logger)
    assert repo.load_rules() == {}

    path.write_text('{"a":{"OpenAI":"o"}}', encoding="utf-8")
    loaded_first = repo.load_rules()
    assert loaded_first["a"]["OpenAI"] == "o"
    loaded_second = repo.load_rules()
    assert loaded_second == loaded_first

    monkeypatch.setattr(
        "kokoro_tts.storage.pronunciation_repository.os.path.getmtime",
        lambda _path: (_ for _ in ()).throw(OSError("stat-failed")),
    )
    assert repo.load_rules() == loaded_first
    assert logger.exceptions

    path.write_text("{bad", encoding="utf-8")
    repo._cached_rules = {"a": {"cached": "yes"}}
    repo._loaded = True
    monkeypatch.setattr(
        "kokoro_tts.storage.pronunciation_repository.os.path.getmtime",
        lambda _path: 1.0,
    )
    assert repo.load_rules() == {"a": {"cached": "yes"}}

    assert repo.parse_rules_json("") == {}
    with pytest.raises(ValueError, match="Rules JSON must be an object"):
        repo._normalize_rules(["bad"])
    assert repo._normalize_language_key("unknown") == ""


def test_pronunciation_repository_file_and_write_errors(monkeypatch, tmp_path: Path):
    logger = _Logger()
    repo = PronunciationRepository(str(tmp_path / "rules.json"), logger_instance=logger)

    with pytest.raises(ValueError, match="No JSON file selected"):
        repo.load_rules_from_file("")
    with pytest.raises(ValueError, match="JSON file not found"):
        repo.load_rules_from_file(str(tmp_path / "missing.json"))

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        repo.load_rules_from_file(str(invalid))

    folder = tmp_path / "folder"
    folder.mkdir()
    with pytest.raises(ValueError, match="Failed to read JSON file"):
        repo.load_rules_from_file(str(folder))

    monkeypatch.setattr(
        "kokoro_tts.storage.pronunciation_repository.os.path.getmtime",
        lambda _path: (_ for _ in ()).throw(OSError("no-mtime")),
    )
    saved = repo.save_rules({"a": {"OpenAI": "o"}})
    assert saved["a"]["OpenAI"] == "o"
    assert "OpenAI" in repo.to_pretty_json(saved)


def test_main_module_guard_executes_launch(monkeypatch):
    called = {"launch": False}
    monkeypatch.setitem(sys.modules, "app", SimpleNamespace(launch=lambda: called.__setitem__("launch", True)))
    sys.modules.pop("kokoro_tts.main", None)
    runpy.run_module("kokoro_tts.main", run_name="__main__")
    assert called["launch"] is True
