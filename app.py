"""Desktop entrypoint and compatibility facade for Kokoro TTS."""

from __future__ import annotations

import atexit
import os
import platform
import sys
import threading
from datetime import datetime

import torch

from kokoro_tts.config import AppConfig, load_config
from kokoro_tts.constants import OUTPUT_FORMATS, SAMPLE_RATE
from kokoro_tts.domain.normalization import (
    DECIMAL_RE,
    DIGIT_RE,
    INT_RE,
    MULTI_DOT_NUMBER_RE,
    ORDINAL_RE,
    ORDINAL_WORDS,
    PERCENT_RE,
    TIME_RE,
    ONES,
    TEENS,
    TENS,
    TextNormalizer,
    digits_to_words,
    normalize_numbers,
    normalize_times,
    number_to_words_0_59,
    number_to_words_0_99,
    number_to_words_0_999,
    number_to_words_0_9999,
    ordinal_to_words,
    ordinalize_words,
    time_to_words,
)
from kokoro_tts.domain.splitting import (
    ABBREV_DOTTED,
    ABBREV_TITLES,
    SENTENCE_BREAK_RE,
    SOFT_BREAK_RE,
    smart_split,
    split_parts,
    split_sentences,
)
from kokoro_tts.domain.expressions import extract_english_expressions
from kokoro_tts.domain.text_utils import (
    MD_LINK_RE,
    SLASHED_RE,
    _apply_outside_spans,
    _find_skip_spans,
    _is_within_spans,
    _merge_spans,
)
from kokoro_tts.domain.voice import (
    CHOICES,
    VOICE_TAG_RE,
    limit_dialogue_parts,
    normalize_voice_input,
    normalize_voice_tag,
    parse_voice_segments,
    resolve_voice,
    summarize_voice,
)
from kokoro_tts.integrations.ffmpeg import configure_ffmpeg
from kokoro_tts.integrations.model_manager import ModelManager
from kokoro_tts.integrations.spaces_gpu import build_forward_gpu
from kokoro_tts.logging_config import setup_logging
from kokoro_tts.storage.audio_writer import AudioWriter
from kokoro_tts.storage.history_repository import HistoryRepository
from kokoro_tts.storage.morphology_repository import MorphologyRepository
from kokoro_tts.storage.pronunciation_repository import PronunciationRepository
from kokoro_tts.application.bootstrap import initialize_app_services
from kokoro_tts.application.context import AppContext
from kokoro_tts.application.history_service import HistoryService
from kokoro_tts.application.local_api import LocalKokoroApi
from kokoro_tts.application.state import KokoroState
from kokoro_tts.application.ui_hooks import UiHooks

CONFIG = load_config()
logger = setup_logging(CONFIG)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


SKIP_APP_INIT = _env_flag("KOKORO_SKIP_APP_INIT")

BASE_DIR = os.path.dirname(__file__)
configure_ffmpeg(logger, BASE_DIR)
PRONUNCIATION_RULES_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        os.getenv("PRONUNCIATION_RULES_PATH", "data/pronunciation_rules.json").strip(),
    )
)

logger.info("Starting app")
logger.info("Log file: %s", CONFIG.log_file)
logger.debug(
    "Log config: LOG_LEVEL=%s FILE_LOG_LEVEL=%s LOG_DIR=%s OUTPUT_DIR=%s REPO_ID=%s "
    "MAX_CHUNK_CHARS=%s HISTORY_LIMIT=%s NORMALIZE_TIMES=%s NORMALIZE_NUMBERS=%s "
    "DEFAULT_OUTPUT_FORMAT=%s DEFAULT_CONCURRENCY_LIMIT=%s LOG_EVERY_N_SEGMENTS=%s "
    "MORPH_DB_ENABLED=%s MORPH_DB_PATH=%s MORPH_DB_TABLE_PREFIX=%s "
    "MORPH_LOCAL_EXPRESSIONS_ENABLED=%s TORCH_NUM_THREADS=%s "
    "TORCH_NUM_INTEROP_THREADS=%s TTS_PREWARM_ENABLED=%s TTS_PREWARM_ASYNC=%s "
    "TTS_PREWARM_VOICE=%s TTS_PREWARM_STYLE=%s MORPH_ASYNC_INGEST=%s "
    "MORPH_ASYNC_MAX_PENDING=%s POSTFX_ENABLED=%s POSTFX_TRIM_ENABLED=%s "
    "POSTFX_TRIM_THRESHOLD_DB=%s POSTFX_TRIM_KEEP_MS=%s POSTFX_FADE_IN_MS=%s "
    "POSTFX_FADE_OUT_MS=%s POSTFX_CROSSFADE_MS=%s POSTFX_LOUDNESS_ENABLED=%s "
    "POSTFX_LOUDNESS_TARGET_LUFS=%s POSTFX_LOUDNESS_TRUE_PEAK_DB=%s",
    CONFIG.log_level,
    CONFIG.file_log_level,
    CONFIG.log_dir,
    CONFIG.output_dir,
    CONFIG.repo_id,
    CONFIG.max_chunk_chars,
    CONFIG.history_limit,
    CONFIG.normalize_times,
    CONFIG.normalize_numbers,
    CONFIG.default_output_format,
    CONFIG.default_concurrency_limit,
    CONFIG.log_segment_every,
    CONFIG.morph_db_enabled,
    CONFIG.morph_db_path,
    CONFIG.morph_db_table_prefix,
    CONFIG.morph_local_expressions_enabled,
    CONFIG.torch_num_threads,
    CONFIG.torch_num_interop_threads,
    CONFIG.tts_prewarm_enabled,
    CONFIG.tts_prewarm_async,
    CONFIG.tts_prewarm_voice,
    CONFIG.tts_prewarm_style,
    CONFIG.morph_async_ingest,
    CONFIG.morph_async_max_pending,
    CONFIG.postfx_enabled,
    CONFIG.postfx_trim_enabled,
    CONFIG.postfx_trim_threshold_db,
    CONFIG.postfx_trim_keep_ms,
    CONFIG.postfx_fade_in_ms,
    CONFIG.postfx_fade_out_ms,
    CONFIG.postfx_crossfade_ms,
    CONFIG.postfx_loudness_enabled,
    CONFIG.postfx_loudness_target_lufs,
    CONFIG.postfx_loudness_true_peak_db,
)
logger.info(
    "Audio post-processing is %s",
    "enabled" if CONFIG.postfx_enabled else "disabled",
)


def _configure_torch_runtime() -> None:
    if CONFIG.torch_num_threads:
        try:
            torch.set_num_threads(CONFIG.torch_num_threads)
            logger.info("Torch CPU threads set to %s", CONFIG.torch_num_threads)
        except Exception:
            logger.exception("Failed to set TORCH_NUM_THREADS=%s", CONFIG.torch_num_threads)
    if CONFIG.torch_num_interop_threads:
        try:
            torch.set_num_interop_threads(CONFIG.torch_num_interop_threads)
            logger.info("Torch interop threads set to %s", CONFIG.torch_num_interop_threads)
        except Exception:
            logger.exception(
                "Failed to set TORCH_NUM_INTEROP_THREADS=%s",
                CONFIG.torch_num_interop_threads,
            )


_configure_torch_runtime()

CUDA_AVAILABLE = torch.cuda.is_available()
logger.info(
    "Config: SPACE_ID=%s IS_DUPLICATE=%s CHAR_LIMIT=%s CUDA_AVAILABLE=%s",
    CONFIG.space_id,
    CONFIG.is_duplicate,
    CONFIG.char_limit,
    CUDA_AVAILABLE,
)
if os.getenv("HF_TOKEN"):
    logger.debug("HF_TOKEN is set")
else:
    logger.debug("HF_TOKEN is not set; hub requests may be rate-limited")
logger.debug("Python version: %s", sys.version.replace("\n", " "))
logger.debug("Platform: %s", platform.platform())
logger.debug("Torch version: %s", torch.__version__)
if not CONFIG.is_duplicate:
    import kokoro
    import misaki

    logger.debug("Kokoro version: %s", kokoro.__version__)
    logger.debug("Misaki version: %s", misaki.__version__)

DEFAULT_OUTPUT_FORMAT = (
    CONFIG.default_output_format if CONFIG.default_output_format in OUTPUT_FORMATS else "wav"
)

logger.info("Voices available: %s (lazy loading)", len(CHOICES))

APP_CONTEXT = AppContext(
    config=CONFIG,
    logger=logger,
    cuda_available=CUDA_AVAILABLE,
    skip_app_init=SKIP_APP_INIT,
    tts_only_mode=_env_flag("TTS_ONLY_MODE"),
    pronunciation_rules_path=PRONUNCIATION_RULES_PATH,
    default_output_format=DEFAULT_OUTPUT_FORMAT,
)

# Legacy compatibility aliases; runtime logic is context-backed.
MODEL_MANAGER = APP_CONTEXT.model_manager
TEXT_NORMALIZER = APP_CONTEXT.text_normalizer
AUDIO_WRITER = APP_CONTEXT.audio_writer
HISTORY_REPOSITORY = APP_CONTEXT.history_repository
HISTORY_SERVICE = APP_CONTEXT.history_service
MORPHOLOGY_REPOSITORY = APP_CONTEXT.morphology_repository
PRONUNCIATION_REPOSITORY = APP_CONTEXT.pronunciation_repository
APP_STATE = APP_CONTEXT.app_state
app = APP_CONTEXT.app
TTS_ONLY_MODE = APP_CONTEXT.tts_only_mode
MORPH_DEFAULT_EXPRESSION_EXTRACTOR = APP_CONTEXT.morph_default_expression_extractor
_PREWARM_LOCK = threading.Lock()
_PREWARM_STARTED = False


def _legacy_or_context(legacy_value, context_value):
    if legacy_value is context_value:
        return context_value
    return legacy_value


def _current_model_manager():
    return _legacy_or_context(MODEL_MANAGER, APP_CONTEXT.model_manager)


def _current_text_normalizer():
    return _legacy_or_context(TEXT_NORMALIZER, APP_CONTEXT.text_normalizer)


def _current_audio_writer():
    return _legacy_or_context(AUDIO_WRITER, APP_CONTEXT.audio_writer)


def _current_history_repository():
    return _legacy_or_context(HISTORY_REPOSITORY, APP_CONTEXT.history_repository)


def _current_history_service():
    return _legacy_or_context(HISTORY_SERVICE, APP_CONTEXT.history_service)


def _current_morphology_repository():
    return _legacy_or_context(MORPHOLOGY_REPOSITORY, APP_CONTEXT.morphology_repository)


def _current_pronunciation_repository():
    return _legacy_or_context(PRONUNCIATION_REPOSITORY, APP_CONTEXT.pronunciation_repository)


def _current_app_state():
    return _legacy_or_context(APP_STATE, APP_CONTEXT.app_state)


def _current_desktop_app():
    return _legacy_or_context(app, APP_CONTEXT.app)


def _current_morph_default_extractor():
    return _legacy_or_context(
        MORPH_DEFAULT_EXPRESSION_EXTRACTOR,
        APP_CONTEXT.morph_default_expression_extractor,
    )


def _sync_legacy_aliases_from_context() -> None:
    global MODEL_MANAGER
    global TEXT_NORMALIZER
    global AUDIO_WRITER
    global HISTORY_REPOSITORY
    global HISTORY_SERVICE
    global MORPHOLOGY_REPOSITORY
    global PRONUNCIATION_REPOSITORY
    global APP_STATE
    global app
    global MORPH_DEFAULT_EXPRESSION_EXTRACTOR

    MODEL_MANAGER = APP_CONTEXT.model_manager
    TEXT_NORMALIZER = APP_CONTEXT.text_normalizer
    AUDIO_WRITER = APP_CONTEXT.audio_writer
    HISTORY_REPOSITORY = APP_CONTEXT.history_repository
    HISTORY_SERVICE = APP_CONTEXT.history_service
    MORPHOLOGY_REPOSITORY = APP_CONTEXT.morphology_repository
    PRONUNCIATION_REPOSITORY = APP_CONTEXT.pronunciation_repository
    APP_STATE = APP_CONTEXT.app_state
    app = APP_CONTEXT.app
    MORPH_DEFAULT_EXPRESSION_EXTRACTOR = APP_CONTEXT.morph_default_expression_extractor


def _current_tts_port() -> LocalKokoroApi:
    if APP_CONTEXT.tts_port is None:
        APP_CONTEXT.tts_port = LocalKokoroApi(
            state_provider=_current_app_state,
            refresh_pronunciation_rules=_refresh_pronunciation_rules_from_store,
            default_use_gpu=CUDA_AVAILABLE,
        )
    return APP_CONTEXT.tts_port


def _tts_only_status_message() -> str:
    if TTS_ONLY_MODE:
        return "TTS-only mode is enabled: Morphology DB writes are disabled."
    return "TTS-only mode is disabled: Morphology DB writes are enabled."


def _apply_runtime_modes() -> None:
    state = _current_app_state()
    if state is not None:
        try:
            state.set_aux_features_enabled(not TTS_ONLY_MODE)
        except Exception:
            logger.exception("Failed to apply runtime mode to app state")

    morphology_repository = _current_morphology_repository()
    if morphology_repository is None:
        return

    try:
        default_extractor = _current_morph_default_extractor()
        if default_extractor is None:
            default_extractor = getattr(morphology_repository, "expression_extractor", None)
        if not TTS_ONLY_MODE:
            morphology_repository.expression_extractor = extract_english_expressions
        elif default_extractor is not None:
            morphology_repository.expression_extractor = default_extractor
    except Exception:
        logger.exception("Failed to apply runtime mode to morphology repository")


def _prewarm_runtime() -> None:
    global _PREWARM_STARTED
    if SKIP_APP_INIT:
        return
    if not CONFIG.tts_prewarm_enabled:
        return
    with _PREWARM_LOCK:
        if _PREWARM_STARTED:
            return
        _PREWARM_STARTED = True

    def run() -> None:
        app_state = _current_app_state()
        if app_state is None:
            return
        prewarm = getattr(app_state, "prewarm_inference", None)
        if not callable(prewarm):
            return
        warm_voice = normalize_voice_input(CONFIG.tts_prewarm_voice or "af_heart")
        use_gpu = CUDA_AVAILABLE
        try:
            prewarm(
                voice=warm_voice,
                use_gpu=use_gpu,
                style_preset=CONFIG.tts_prewarm_style,
            )
            logger.info(
                "TTS prewarm complete: voice=%s use_gpu=%s",
                warm_voice,
                use_gpu,
            )
        except Exception:
            logger.exception(
                "TTS prewarm failed: voice=%s use_gpu=%s",
                warm_voice,
                use_gpu,
            )

    if CONFIG.tts_prewarm_async:
        worker = threading.Thread(
            target=run,
            name="tts-prewarm",
            daemon=True,
        )
        worker.start()
        logger.info("Started asynchronous TTS prewarm")
    else:
        logger.info("Running synchronous TTS prewarm")
        run()


def set_tts_only_mode(enabled: bool) -> str:
    global TTS_ONLY_MODE
    TTS_ONLY_MODE = bool(enabled)
    APP_CONTEXT.tts_only_mode = TTS_ONLY_MODE
    _apply_runtime_modes()
    logger.info("TTS-only mode set to %s", TTS_ONLY_MODE)
    return _tts_only_status_message()


def _model_manager_instance():
    model_manager = _current_model_manager()
    if model_manager is not None:
        return model_manager
    state = _current_app_state()
    if state is not None:
        return getattr(state, "model_manager", None)
    return None


def _refresh_pronunciation_rules_from_store() -> None:
    model_manager = _model_manager_instance()
    pronunciation_repository = _current_pronunciation_repository()
    if model_manager is None or pronunciation_repository is None:
        return
    load_rules_shared = getattr(pronunciation_repository, "load_rules_shared", None)
    if callable(load_rules_shared):
        rules, changed = load_rules_shared()
        if not changed:
            return
    else:
        rules = pronunciation_repository.load_rules()
    model_manager.set_pronunciation_rules(rules)


def _save_and_apply_pronunciation_rules(
    rules: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], int]:
    pronunciation_repository = _current_pronunciation_repository()
    if pronunciation_repository is None:
        raise RuntimeError("Pronunciation dictionary is not configured.")
    model_manager = _model_manager_instance()
    if model_manager is None:
        raise RuntimeError("Model manager is not initialized.")
    saved_rules = pronunciation_repository.save_rules(rules)
    entry_count = model_manager.set_pronunciation_rules(saved_rules)
    return saved_rules, entry_count


def _resolve_uploaded_file_path(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    if isinstance(uploaded_file, str):
        return uploaded_file
    if isinstance(uploaded_file, (list, tuple)):
        return _resolve_uploaded_file_path(uploaded_file[0]) if uploaded_file else ""
    if isinstance(uploaded_file, dict):
        return str(uploaded_file.get("path") or uploaded_file.get("name") or "").strip()
    return str(uploaded_file).strip()


def load_pronunciation_rules_json():
    pronunciation_repository = _current_pronunciation_repository()
    if pronunciation_repository is None:
        return "{}", "Pronunciation dictionary is not configured."
    model_manager = _model_manager_instance()
    if model_manager is None:
        return "{}", "Model manager is not initialized."
    rules = pronunciation_repository.load_rules()
    entry_count = model_manager.set_pronunciation_rules(rules)
    return (
        pronunciation_repository.to_pretty_json(rules),
        f"Loaded {entry_count} rule(s) across {len(rules)} language(s).",
    )


def apply_pronunciation_rules_json(raw_json):
    pronunciation_repository = _current_pronunciation_repository()
    if pronunciation_repository is None:
        return "{}", "Pronunciation dictionary is not configured."
    try:
        parsed_rules = pronunciation_repository.parse_rules_json(raw_json)
        saved_rules, entry_count = _save_and_apply_pronunciation_rules(parsed_rules)
    except ValueError as exc:
        return str(raw_json or "{}"), f"Invalid dictionary JSON: {exc}"
    except Exception:
        logger.exception("Failed to apply pronunciation dictionary")
        return str(raw_json or "{}"), "Failed to apply dictionary. Check logs for details."
    return (
        pronunciation_repository.to_pretty_json(saved_rules),
        f"Applied {entry_count} rule(s) across {len(saved_rules)} language(s).",
    )


def import_pronunciation_rules_json(uploaded_file):
    pronunciation_repository = _current_pronunciation_repository()
    if pronunciation_repository is None:
        return "{}", "Pronunciation dictionary is not configured."
    file_path = _resolve_uploaded_file_path(uploaded_file)
    if not file_path:
        return "{}", "No JSON file selected."
    try:
        parsed_rules = pronunciation_repository.load_rules_from_file(file_path)
        saved_rules, entry_count = _save_and_apply_pronunciation_rules(parsed_rules)
    except ValueError as exc:
        return "{}", str(exc)
    except Exception:
        logger.exception("Failed to import pronunciation dictionary: %s", file_path)
        return "{}", "Failed to import dictionary. Check logs for details."
    return (
        pronunciation_repository.to_pretty_json(saved_rules),
        f"Imported {entry_count} rule(s) across {len(saved_rules)} language(s).",
    )


def export_pronunciation_rules_json():
    pronunciation_repository = _current_pronunciation_repository()
    if pronunciation_repository is None:
        return None, "Pronunciation dictionary is not configured."
    config = CONFIG
    try:
        rules = pronunciation_repository.load_rules()
        json_text = pronunciation_repository.to_pretty_json(rules)
        date_dir = os.path.join(
            config.output_dir_abs,
            datetime.now().strftime("%Y-%m-%d"),
            "vocabulary",
        )
        os.makedirs(date_dir, exist_ok=True)
        filename = f"pronunciation_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(date_dir, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(json_text)
    except Exception:
        logger.exception("Pronunciation dictionary export failed")
        return None, "Export failed. Check logs for details."
    return output_path, f"Export ready: {filename}"


def forward_gpu(ps, ref_s, speed):
    state = _current_app_state()
    if state is None:
        raise RuntimeError("App state is not initialized.")
    return state.model_manager.get_model(True)(ps, ref_s, speed)


def generate_first(
    text,
    voice="af_heart",
    mix_enabled=False,
    voice_mix=None,
    speed=1,
    use_gpu=CUDA_AVAILABLE,
    pause_seconds=0.0,
    output_format="wav",
    normalize_times_enabled=None,
    normalize_numbers_enabled=None,
    style_preset="neutral",
):
    return _current_tts_port().generate_first(
        text=text,
        voice=voice,
        mix_enabled=mix_enabled,
        voice_mix=voice_mix,
        speed=speed,
        use_gpu=use_gpu,
        pause_seconds=pause_seconds,
        output_format=output_format,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
        style_preset=style_preset,
    )


# Arena API
def predict(
    text,
    voice="af_heart",
    mix_enabled=False,
    voice_mix=None,
    speed=1,
    normalize_times_enabled=None,
    normalize_numbers_enabled=None,
    style_preset="neutral",
):
    return _current_tts_port().predict(
        text=text,
        voice=voice,
        mix_enabled=mix_enabled,
        voice_mix=voice_mix,
        speed=speed,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
        style_preset=style_preset,
    )


def tokenize_first(
    text,
    voice="af_heart",
    mix_enabled=False,
    voice_mix=None,
    speed=1,
    normalize_times_enabled=None,
    normalize_numbers_enabled=None,
    style_preset="neutral",
):
    return _current_tts_port().tokenize_first(
        text=text,
        voice=voice,
        mix_enabled=mix_enabled,
        voice_mix=voice_mix,
        speed=speed,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
        style_preset=style_preset,
    )


def generate_all(
    text,
    voice="af_heart",
    mix_enabled=False,
    voice_mix=None,
    speed=1,
    use_gpu=CUDA_AVAILABLE,
    pause_seconds=0.0,
    normalize_times_enabled=None,
    normalize_numbers_enabled=None,
    style_preset="neutral",
):
    yield from _current_tts_port().generate_all(
        text=text,
        voice=voice,
        mix_enabled=mix_enabled,
        voice_mix=voice_mix,
        speed=speed,
        use_gpu=use_gpu,
        pause_seconds=pause_seconds,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
        style_preset=style_preset,
    )


def export_morphology_sheet(dataset: str = "lexemes", file_format: str = "ods"):
    if TTS_ONLY_MODE:
        return None, "TTS-only mode is enabled. Morphology DB export is disabled."
    morphology_repository = _current_morphology_repository()
    if morphology_repository is None:
        return None, "Morphology DB is not configured."
    config = CONFIG
    normalized_format = str(file_format or "ods").strip().lower().lstrip(".")
    try:
        if normalized_format == "csv":
            sheet_path = morphology_repository.export_csv(
                dataset=dataset,
                output_dir=config.output_dir_abs,
            )
        elif normalized_format == "txt":
            sheet_path = morphology_repository.export_txt(
                dataset=dataset,
                output_dir=config.output_dir_abs,
            )
        elif normalized_format in ("xlsx", "excel"):
            sheet_path = morphology_repository.export_excel(
                dataset=dataset,
                output_dir=config.output_dir_abs,
            )
        elif normalized_format in ("docx", "word"):
            # Backward compatibility: old UI alias now points to Excel export.
            sheet_path = morphology_repository.export_excel(
                dataset=dataset,
                output_dir=config.output_dir_abs,
            )
        elif normalized_format in ("ods", "spreadsheet"):
            sheet_path = morphology_repository.export_spreadsheet(
                dataset=dataset,
                output_dir=config.output_dir_abs,
            )
        else:
            return None, "Unsupported export format. Use ods, csv, txt, or xlsx."
    except Exception:
        logger.exception("Morphology export failed: format=%s", normalized_format)
        return None, "Export failed. Check logs for details."
    if not sheet_path:
        return None, "No rows available for export."
    return sheet_path, f"Export ready: {os.path.basename(sheet_path)}"


def _empty_morphology_table_update():
    return {"value": [[]], "headers": ["No data"]}


def morphology_db_view(dataset="occurrences", limit=100, offset=0):
    if TTS_ONLY_MODE:
        return (
            _empty_morphology_table_update(),
            "TTS-only mode is enabled. Morphology DB is disabled.",
        )
    morphology_repository = _current_morphology_repository()
    if morphology_repository is None:
        return _empty_morphology_table_update(), "Morphology DB is not configured."
    try:
        headers, rows = morphology_repository.list_rows(
            dataset=str(dataset or "occurrences"),
            limit=_coerce_int(limit, 100, 1, 1000),
            offset=_coerce_int(offset, 0, 0, 1_000_000),
        )
    except Exception as exc:
        logger.exception("Morphology DB view failed")
        return _empty_morphology_table_update(), f"View failed: {exc}"
    if not headers:
        return _empty_morphology_table_update(), "No table metadata available."
    if not rows:
        return {"value": [], "headers": headers}, "No rows found."
    return {"value": rows, "headers": headers}, f"Loaded {len(rows)} row(s)."


def _coerce_int(value, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


if not SKIP_APP_INIT:
    services = initialize_app_services(
        config=CONFIG,
        cuda_available=CUDA_AVAILABLE,
        logger=logger,
        pronunciation_rules_path=PRONUNCIATION_RULES_PATH,
        generate_first=generate_first,
        tokenize_first=tokenize_first,
        generate_all=generate_all,
        predict=predict,
        export_morphology_sheet=export_morphology_sheet,
        morphology_db_view=morphology_db_view,
        load_pronunciation_rules=load_pronunciation_rules_json,
        apply_pronunciation_rules=apply_pronunciation_rules_json,
        import_pronunciation_rules=import_pronunciation_rules_json,
        export_pronunciation_rules=export_pronunciation_rules_json,
        set_tts_only_mode=set_tts_only_mode,
        tts_only_mode_default=TTS_ONLY_MODE,
        choices=CHOICES,
    )
    APP_CONTEXT.bind_services(services)
    APP_CONTEXT.tts_port = LocalKokoroApi(
        state_provider=_current_app_state,
        refresh_pronunciation_rules=_refresh_pronunciation_rules_from_store,
        default_use_gpu=CUDA_AVAILABLE,
    )
    _sync_legacy_aliases_from_context()
    _apply_runtime_modes()
    _prewarm_runtime()
else:
    logger.info("KOKORO_SKIP_APP_INIT enabled; skipping model and UI initialization")


def _shutdown_runtime() -> None:
    state = _current_app_state()
    if state is None:
        return
    shutdown = getattr(state, "shutdown", None)
    if not callable(shutdown):
        return
    try:
        shutdown(wait=True)
    except Exception:
        logger.exception("Runtime shutdown failed")


atexit.register(_shutdown_runtime)


def launch() -> None:
    if SKIP_APP_INIT:
        logger.info("KOKORO_SKIP_APP_INIT enabled; launch skipped")
        return
    desktop_app = _current_desktop_app()
    if desktop_app is None:
        raise RuntimeError("Desktop app is not initialized.")
    logger.info("Launching desktop app")
    desktop_app.launch()


if __name__ == "__main__":
    launch()
