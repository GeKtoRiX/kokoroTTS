import importlib
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import app
import kokoro_tts.application.bootstrap as bootstrap_mod
import kokoro_tts.config as config_mod
import kokoro_tts.integrations.ffmpeg as ffmpeg_mod
import kokoro_tts.logging_config as logging_mod
import pytest
from kokoro_tts.config import AppConfig


class _Logger:
    def __init__(self):
        self.debugs = []
        self.infos = []
        self.exceptions = []

    def debug(self, message, *args):
        self.debugs.append(message % args if args else message)

    def info(self, message, *args):
        self.infos.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)


def _build_config(tmp_path: Path, *, is_duplicate: bool) -> AppConfig:
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        log_level="INFO",
        file_log_level="DEBUG",
        log_dir=str(logs),
        log_file=str(logs / "app.log"),
        repo_id="repo/x",
        output_dir=str(outputs),
        output_dir_abs=str(outputs.resolve()),
        max_chunk_chars=250,
        history_limit=5,
        normalize_times=True,
        normalize_numbers=True,
        default_output_format="wav",
        default_concurrency_limit=None,
        log_segment_every=5,
        morph_db_enabled=True,
        morph_db_path=str(tmp_path / "data" / "morph.sqlite3"),
        morph_db_table_prefix="morph_",
        space_id="" if is_duplicate else "hexgrad/Kokoro-82M",
        is_duplicate=is_duplicate,
        char_limit=None if is_duplicate else 5000,
        ru_tts_enabled=False,
        ru_tts_model_id="v5_cis_base",
        ru_tts_cache_dir=str(tmp_path / "cache" / "torch"),
        ru_tts_cpu_only=True,
    )


def test_model_manager_runtime_mode_and_status_branches(monkeypatch):
    class _StateWithError:
        def set_aux_features_enabled(self, _enabled):
            raise RuntimeError("state-failed")

    class _RepoWithSetterError:
        @property
        def expression_extractor(self):
            return None

        @expression_extractor.setter
        def expression_extractor(self, _value):
            raise RuntimeError("repo-failed")

    logger = _Logger()
    monkeypatch.setattr(app, "logger", logger)
    monkeypatch.setattr(app, "APP_STATE", _StateWithError())
    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", _RepoWithSetterError())
    monkeypatch.setattr(app, "MORPH_DEFAULT_EXPRESSION_EXTRACTOR", None)
    monkeypatch.setattr(app, "TTS_ONLY_MODE", False)

    app._apply_runtime_modes()
    assert any("app state" in message for message in logger.exceptions)
    assert any("morphology repository" in message for message in logger.exceptions)
    assert "disabled" in app._tts_only_status_message()

    monkeypatch.setattr(app, "TTS_ONLY_MODE", True)
    assert "enabled" in app._tts_only_status_message()

    repo_ok = SimpleNamespace(expression_extractor=lambda _text: ["x"])
    default_extractor = lambda _text: []
    monkeypatch.setattr(app, "APP_STATE", None)
    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", repo_ok)
    monkeypatch.setattr(app, "MORPH_DEFAULT_EXPRESSION_EXTRACTOR", default_extractor)
    app._apply_runtime_modes()
    assert repo_ok.expression_extractor is default_extractor


def test_model_manager_and_pronunciation_helpers(monkeypatch):
    model_manager = SimpleNamespace(set_pronunciation_rules=lambda rules: len(rules))
    state = SimpleNamespace(model_manager=model_manager)
    repo = SimpleNamespace(load_rules=lambda: {"a": {"x": "y"}}, save_rules=lambda rules: dict(rules))

    monkeypatch.setattr(app, "MODEL_MANAGER", model_manager)
    monkeypatch.setattr(app, "APP_STATE", None)
    assert app._model_manager_instance() is model_manager

    monkeypatch.setattr(app, "MODEL_MANAGER", None)
    monkeypatch.setattr(app, "APP_STATE", state)
    assert app._model_manager_instance() is model_manager

    monkeypatch.setattr(app, "APP_STATE", None)
    assert app._model_manager_instance() is None

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", repo)
    monkeypatch.setattr(app, "MODEL_MANAGER", model_manager)
    app._refresh_pronunciation_rules_from_store()

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", None)
    with pytest.raises(RuntimeError, match="not configured"):
        app._save_and_apply_pronunciation_rules({"a": {"x": "y"}})

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", repo)
    monkeypatch.setattr(app, "MODEL_MANAGER", None)
    monkeypatch.setattr(app, "APP_STATE", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        app._save_and_apply_pronunciation_rules({"a": {"x": "y"}})

    monkeypatch.setattr(app, "MODEL_MANAGER", model_manager)
    saved, count = app._save_and_apply_pronunciation_rules({"a": {"x": "y"}})
    assert saved["a"]["x"] == "y"
    assert count == 1


def test_pronunciation_json_branches(monkeypatch, tmp_path):
    logger = _Logger()
    monkeypatch.setattr(app, "logger", logger)
    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", None)
    text, status = app.load_pronunciation_rules_json()
    assert text == "{}"
    assert "not configured" in status

    dummy_repo = SimpleNamespace(load_rules=lambda: {"a": {"x": "y"}})
    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", dummy_repo)
    monkeypatch.setattr(app, "MODEL_MANAGER", None)
    monkeypatch.setattr(app, "APP_STATE", None)
    _, status = app.load_pronunciation_rules_json()
    assert "not initialized" in status

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", None)
    _, status = app.apply_pronunciation_rules_json("{}")
    assert "not configured" in status

    class _BadParseRepo:
        def parse_rules_json(self, _raw):
            raise ValueError("bad-json")

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", _BadParseRepo())
    text, status = app.apply_pronunciation_rules_json("bad")
    assert text == "bad"
    assert "Invalid dictionary JSON" in status

    class _ExplosiveParseRepo:
        def parse_rules_json(self, _raw):
            return {"a": {"x": "y"}}

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", _ExplosiveParseRepo())
    monkeypatch.setattr(
        app,
        "_save_and_apply_pronunciation_rules",
        lambda _rules: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    text, status = app.apply_pronunciation_rules_json("{}")
    assert text == "{}"
    assert "Failed to apply dictionary" in status

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", None)
    _, status = app.import_pronunciation_rules_json("x.json")
    assert "not configured" in status

    class _ImportRepo:
        def load_rules_from_file(self, _path):
            raise ValueError("bad-file")

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", _ImportRepo())
    _, status = app.import_pronunciation_rules_json("")
    assert "No JSON file selected" in status
    _, status = app.import_pronunciation_rules_json("rules.json")
    assert status == "bad-file"

    class _ImportCrashRepo:
        def load_rules_from_file(self, _path):
            raise RuntimeError("io-failed")

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", _ImportCrashRepo())
    _, status = app.import_pronunciation_rules_json({"path": "x.json"})
    assert "Failed to import dictionary" in status

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", None)
    export_path, status = app.export_pronunciation_rules_json()
    assert export_path is None
    assert "not configured" in status

    class _ExportCrashRepo:
        def load_rules(self):
            raise RuntimeError("load-failed")

    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", _ExportCrashRepo())
    monkeypatch.setattr(app, "CONFIG", SimpleNamespace(output_dir_abs=str(tmp_path)))
    export_path, status = app.export_pronunciation_rules_json()
    assert export_path is None
    assert "Export failed" in status


def test_resolve_uploaded_and_morphology_branches(monkeypatch, tmp_path):
    assert app._resolve_uploaded_file_path(None) == ""
    assert app._resolve_uploaded_file_path("x.json") == "x.json"
    assert app._resolve_uploaded_file_path([]) == ""
    assert app._resolve_uploaded_file_path(["a.json"]) == "a.json"
    assert app._resolve_uploaded_file_path(({"path": "b.json"},)) == "b.json"
    assert app._resolve_uploaded_file_path({"name": "c.json"}) == "c.json"
    assert app._resolve_uploaded_file_path(Path("d.json")) == "d.json"

    monkeypatch.setattr(app, "TTS_ONLY_MODE", True)
    path, status = app.export_morphology_sheet("lexemes", "csv")
    assert path is None
    assert "disabled" in status.lower()

    class _Repo:
        def export_csv(self, **_kwargs):
            return ""

    monkeypatch.setattr(app, "TTS_ONLY_MODE", False)
    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", _Repo())
    monkeypatch.setattr(app, "CONFIG", SimpleNamespace(output_dir_abs=str(tmp_path)))
    path, status = app.export_morphology_sheet("lexemes", "csv")
    assert path is None
    assert "No rows available" in status

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", None)
    table_update, status = app.morphology_db_view()
    assert table_update["headers"] == ["No data"]
    assert "not configured" in status

    class _ViewRepoError:
        def list_rows(self, **_kwargs):
            raise RuntimeError("view-failed")

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", _ViewRepoError())
    table_update, status = app.morphology_db_view(limit="bad", offset=None)
    assert table_update["headers"] == ["No data"]
    assert "View failed" in status

    class _ViewRepoNoHeaders:
        def list_rows(self, **_kwargs):
            return [], [["x"]]

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", _ViewRepoNoHeaders())
    table_update, status = app.morphology_db_view()
    assert table_update["headers"] == ["No data"]
    assert "No table metadata" in status

    class _ViewRepoNoRows:
        def list_rows(self, **_kwargs):
            return ["col"], []

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", _ViewRepoNoRows())
    table_update, status = app.morphology_db_view()
    assert table_update["headers"] == ["col"]
    assert table_update["value"] == []
    assert "No rows found" in status

    assert app._coerce_int("x", 7, 1, 10) == 7


def test_app_module_reload_init_path_and_main_guard(monkeypatch, tmp_path):
    logger = _Logger()
    loaded_config = _build_config(tmp_path, is_duplicate=False)
    called = {"ffmpeg": False, "launch": False}

    services = SimpleNamespace(
        model_manager=object(),
        silero_manager=None,
        text_normalizer=object(),
        audio_writer=object(),
        morphology_repository=SimpleNamespace(expression_extractor="default"),
        pronunciation_repository=object(),
        app_state=SimpleNamespace(set_aux_features_enabled=lambda _enabled: None),
        history_repository=object(),
        history_service=object(),
        app=SimpleNamespace(launch=lambda: called.__setitem__("launch", True)),
    )

    monkeypatch.setenv("KOKORO_SKIP_APP_INIT", "0")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.delenv("TTS_ONLY_MODE", raising=False)
    monkeypatch.setattr(config_mod, "load_config", lambda: loaded_config)
    monkeypatch.setattr(logging_mod, "setup_logging", lambda _cfg: logger)
    monkeypatch.setattr(bootstrap_mod, "initialize_app_services", lambda **_kwargs: services)
    monkeypatch.setattr(ffmpeg_mod, "configure_ffmpeg", lambda *_args, **_kwargs: called.__setitem__("ffmpeg", True))

    reloaded = importlib.reload(app)
    assert reloaded.SKIP_APP_INIT is False
    assert reloaded.MODEL_MANAGER is services.model_manager
    assert called["ffmpeg"] is True
    assert any("HF_TOKEN is set" in line for line in logger.debugs)
    assert any("Kokoro version" in line for line in logger.debugs)
    assert any("Misaki version" in line for line in logger.debugs)

    reloaded.launch()
    assert called["launch"] is True

    monkeypatch.setenv("KOKORO_SKIP_APP_INIT", "1")
    runpy.run_module("app", run_name="__main__")
    importlib.reload(app)
