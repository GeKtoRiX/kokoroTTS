from pathlib import Path

from kokoro_tts.config import AppConfig
from kokoro_tts.logging_config import setup_logging
from kokoro_tts.runtime import CUDA_AVAILABLE
from kokoro_tts.ui.gradio_app import APP_TITLE, create_gradio_app


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


def _build_config(tmp_path: Path, *, history_limit: int, morph_enabled: bool, space_id: str) -> AppConfig:
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


def test_create_gradio_app_in_both_api_modes(tmp_path):
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

    config_open = _build_config(tmp_path, history_limit=2, morph_enabled=True, space_id="")
    app_open, api_open = create_gradio_app(
        config=config_open,
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
    assert app_open is not None
    assert app_open.title == APP_TITLE
    assert api_open is True
    app_open_components = app_open.get_config_file().get("components", [])
    assert any(
        component.get("type") == "markdown"
        and component.get("props", {}).get("value") == f"# {APP_TITLE}"
        for component in app_open_components
    )

    config_closed = _build_config(
        tmp_path,
        history_limit=0,
        morph_enabled=False,
        space_id="hexgrad/Kokoro-TTS",
    )
    app_closed, api_open_closed = create_gradio_app(
        config=config_closed,
        cuda_available=True,
        logger=logger,
        generate_first=_generate_first,
        tokenize_first=_tokenize_first,
        generate_all=_generate_all,
        predict=_predict,
        export_morphology_sheet=None,
        history_service=service,
        choices={},
    )
    assert app_closed is not None
    assert app_closed.title == APP_TITLE
    assert api_open_closed is False
