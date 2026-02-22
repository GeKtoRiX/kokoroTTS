import sys
from pathlib import Path
from types import SimpleNamespace

if "torch" not in sys.modules:
    sys.modules["torch"] = SimpleNamespace(
        __version__="0.0-test",
        cuda=SimpleNamespace(is_available=lambda: False),
    )

from kokoro_tts.config import AppConfig
from kokoro_tts.logging_config import setup_logging
from kokoro_tts.runtime import CUDA_AVAILABLE
from kokoro_tts.ui.common import APP_TITLE
from kokoro_tts.ui.tkinter_app import create_tkinter_app


class _Logger:
    def __init__(self):
        self.debugs = []

    def debug(self, message, *args):
        self.debugs.append(message % args if args else message)


class _HistoryService:
    def update_history(self, history):
        return list(history or [])

    def clear_history(self, history):
        _ = history
        return []


def _build_config(
    tmp_path: Path, *, history_limit: int, morph_enabled: bool, space_id: str
) -> AppConfig:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        log_level="INFO",
        file_log_level="DEBUG",
        log_dir=str(log_dir),
        log_file=str(log_dir / "app.log"),
        repo_id="repo/x",
        output_dir=str(outputs),
        output_dir_abs=str(outputs.resolve()),
        app_state_path=str(tmp_path / "data" / "app_state.json"),
        max_chunk_chars=250,
        history_limit=history_limit,
        normalize_times=True,
        normalize_numbers=True,
        default_output_format="wav",
        default_concurrency_limit=None,
        log_segment_every=5,
        morph_db_enabled=morph_enabled,
        morph_db_path=str(tmp_path / "data" / "morph.sqlite3"),
        morph_db_table_prefix="morph_",
        space_id=space_id,
        is_duplicate=not space_id.startswith("hexgrad/"),
        char_limit=None if not space_id.startswith("hexgrad/") else 5000,
    )


def test_setup_logging_replaces_handlers(tmp_path):
    config = _build_config(tmp_path, history_limit=1, morph_enabled=False, space_id="")
    logger = setup_logging(config)
    logger_again = setup_logging(config)

    assert logger is logger_again
    assert len(logger.handlers) == 2
    assert Path(config.log_file).exists()


def test_runtime_flag_is_boolean():
    assert isinstance(CUDA_AVAILABLE, bool)


def test_create_tkinter_app_builds_root(tmp_path):
    logger = _Logger()
    service = _HistoryService()

    def _generate_first(*_args, **_kwargs):
        return None, ""

    def _tokenize_first(*_args, **_kwargs):
        return ""

    def _generate_all(*_args, **_kwargs):
        yield 24000, []

    def _predict(*_args, **_kwargs):
        return 24000, []

    config_value = _build_config(tmp_path, history_limit=2, morph_enabled=True, space_id="")
    app_instance = create_tkinter_app(
        config=config_value,
        cuda_available=False,
        logger=logger,
        generate_first=_generate_first,
        tokenize_first=_tokenize_first,
        generate_all=_generate_all,
        predict=_predict,
        export_morphology_sheet=lambda dataset: (None, f"dataset={dataset}"),
        history_service=service,
        choices={},
    )
    assert app_instance is not None
    assert app_instance.title == APP_TITLE
    root = app_instance.build_for_test()
    tab_texts = [
        app_instance.notebook.tab(tab_id, "text") for tab_id in app_instance.notebook.tabs()
    ]
    assert "Generate" in tab_texts
    assert "Stream" in tab_texts
    assert "Morphology DB" not in tab_texts

    app_instance._set_runtime_mode("tts_morph", apply_backend=False)
    tab_texts = [
        app_instance.notebook.tab(tab_id, "text") for tab_id in app_instance.notebook.tabs()
    ]
    assert "Morphology DB" in tab_texts

    root.withdraw()
    root.update_idletasks()
    root.destroy()
