"""Gradio entrypoint and compatibility facade for Kokoro TTS."""
from __future__ import annotations

import inspect
import json
import os
import platform
import sys
from datetime import datetime

import gradio as gr
import spaces
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
from kokoro_tts.domain.morphology_datasets import normalize_morphology_dataset
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
from kokoro_tts.integrations.lm_studio import (
    LmStudioError,
    LessonRequest,
    PosVerifyRequest,
    generate_lesson_text,
    verify_pos_with_context,
)
from kokoro_tts.integrations.model_manager import ModelManager
from kokoro_tts.integrations.spaces_gpu import build_forward_gpu
from kokoro_tts.logging_config import setup_logging
from kokoro_tts.storage.audio_writer import AudioWriter
from kokoro_tts.storage.history_repository import HistoryRepository
from kokoro_tts.storage.morphology_repository import MorphologyRepository
from kokoro_tts.storage.pronunciation_repository import PronunciationRepository
from kokoro_tts.application.bootstrap import initialize_app_services
from kokoro_tts.application.history_service import HistoryService
from kokoro_tts.application.state import KokoroState
from kokoro_tts.application.ui_hooks import UiHooks
from kokoro_tts.ui.gradio_app import (
    APP_CSS,
    APP_THEME,
    DIALOGUE_NOTE,
    TOKEN_NOTE,
    UI_PRIMARY_HUE,
    create_gradio_app,
)

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
    "LM_STUDIO_BASE_URL=%s LM_STUDIO_MODEL=%s LM_STUDIO_TIMEOUT_SECONDS=%s "
    "LM_STUDIO_TEMPERATURE=%s LM_STUDIO_MAX_TOKENS=%s "
    "LM_VERIFY_ENABLED=%s LM_VERIFY_BASE_URL=%s LM_VERIFY_MODEL=%s "
    "LM_VERIFY_TIMEOUT_SECONDS=%s LM_VERIFY_TEMPERATURE=%s LM_VERIFY_MAX_TOKENS=%s "
    "LM_VERIFY_MAX_RETRIES=%s LM_VERIFY_WORKERS=%s "
    "MORPH_LOCAL_EXPRESSIONS_ENABLED=%s",
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
    CONFIG.lm_studio_base_url,
    CONFIG.lm_studio_model,
    CONFIG.lm_studio_timeout_seconds,
    CONFIG.lm_studio_temperature,
    CONFIG.lm_studio_max_tokens,
    CONFIG.lm_verify_enabled,
    CONFIG.lm_verify_base_url,
    CONFIG.lm_verify_model,
    CONFIG.lm_verify_timeout_seconds,
    CONFIG.lm_verify_temperature,
    CONFIG.lm_verify_max_tokens,
    CONFIG.lm_verify_max_retries,
    CONFIG.lm_verify_workers,
    CONFIG.morph_local_expressions_enabled,
)

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
    CONFIG.default_output_format
    if CONFIG.default_output_format in OUTPUT_FORMATS
    else "wav"
)

logger.info("Voices available: %s (lazy loading)", len(CHOICES))

MODEL_MANAGER = None
TEXT_NORMALIZER = None
AUDIO_WRITER = None
HISTORY_REPOSITORY = None
HISTORY_SERVICE = None
MORPHOLOGY_REPOSITORY = None
PRONUNCIATION_REPOSITORY = None
APP_STATE = None
app = None
API_OPEN = None
SSR_MODE = os.getenv("KOKORO_SSR_MODE", "").strip().lower() in ("1", "true", "yes")
TTS_ONLY_MODE = _env_flag("TTS_ONLY_MODE")
LLM_ONLY_MODE = _env_flag("LLM_ONLY_MODE")
MORPH_DEFAULT_EXPRESSION_EXTRACTOR = None
MORPH_DEFAULT_LM_VERIFY_ENABLED = None


def _tts_only_status_message() -> str:
    if TTS_ONLY_MODE:
        return "TTS-only mode is enabled: Morphology DB writes and LLM requests are disabled."
    return "TTS-only mode is disabled: Morphology DB writes and LLM requests are enabled."


def _llm_only_status_message() -> str:
    if TTS_ONLY_MODE:
        return "TTS + Morphology mode is overridden by TTS-only mode."
    if LLM_ONLY_MODE:
        return "TTS + Morphology mode is enabled: LLM requests are disabled while Morphology DB stays enabled."
    return "TTS + Morphology mode is disabled: LLM requests are enabled."


def _llm_requests_disabled() -> bool:
    return bool(TTS_ONLY_MODE or LLM_ONLY_MODE)


def _apply_runtime_modes() -> None:
    if APP_STATE is not None:
        try:
            APP_STATE.set_aux_features_enabled(not TTS_ONLY_MODE)
        except Exception:
            logger.exception("Failed to apply runtime mode to app state")

    if MORPHOLOGY_REPOSITORY is None:
        return

    try:
        if MORPH_DEFAULT_LM_VERIFY_ENABLED is None:
            default_verify_enabled = bool(getattr(MORPHOLOGY_REPOSITORY, "lm_verify_enabled", False))
        else:
            default_verify_enabled = bool(MORPH_DEFAULT_LM_VERIFY_ENABLED)
        llm_enabled = not _llm_requests_disabled()
        MORPHOLOGY_REPOSITORY.lm_verify_enabled = bool(
            default_verify_enabled
            and llm_enabled
            and callable(getattr(MORPHOLOGY_REPOSITORY, "lm_verifier", None))
        )

        default_extractor = MORPH_DEFAULT_EXPRESSION_EXTRACTOR
        if default_extractor is None:
            default_extractor = getattr(MORPHOLOGY_REPOSITORY, "expression_extractor", None)
        if not llm_enabled and not TTS_ONLY_MODE:
            MORPHOLOGY_REPOSITORY.expression_extractor = extract_english_expressions
        elif default_extractor is not None:
            MORPHOLOGY_REPOSITORY.expression_extractor = default_extractor
    except Exception:
        logger.exception("Failed to apply runtime mode to morphology repository")


def set_tts_only_mode(enabled: bool) -> str:
    global TTS_ONLY_MODE
    TTS_ONLY_MODE = bool(enabled)
    _apply_runtime_modes()
    logger.info("TTS-only mode set to %s", TTS_ONLY_MODE)
    return _tts_only_status_message()


def set_llm_only_mode(enabled: bool) -> str:
    global LLM_ONLY_MODE
    LLM_ONLY_MODE = bool(enabled)
    _apply_runtime_modes()
    logger.info("TTS + Morphology mode set to %s", LLM_ONLY_MODE)
    return _llm_only_status_message()


def _model_manager_instance():
    if MODEL_MANAGER is not None:
        return MODEL_MANAGER
    if APP_STATE is not None:
        return getattr(APP_STATE, "model_manager", None)
    return None


def _refresh_pronunciation_rules_from_store() -> None:
    model_manager = _model_manager_instance()
    if model_manager is None or PRONUNCIATION_REPOSITORY is None:
        return
    rules = PRONUNCIATION_REPOSITORY.load_rules()
    model_manager.set_pronunciation_rules(rules)


def _save_and_apply_pronunciation_rules(
    rules: dict[str, dict[str, str]]
) -> tuple[dict[str, dict[str, str]], int]:
    if PRONUNCIATION_REPOSITORY is None:
        raise RuntimeError("Pronunciation dictionary is not configured.")
    model_manager = _model_manager_instance()
    if model_manager is None:
        raise RuntimeError("Model manager is not initialized.")
    saved_rules = PRONUNCIATION_REPOSITORY.save_rules(rules)
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
    if PRONUNCIATION_REPOSITORY is None:
        return "{}", "Pronunciation dictionary is not configured."
    model_manager = _model_manager_instance()
    if model_manager is None:
        return "{}", "Model manager is not initialized."
    rules = PRONUNCIATION_REPOSITORY.load_rules()
    entry_count = model_manager.set_pronunciation_rules(rules)
    return (
        PRONUNCIATION_REPOSITORY.to_pretty_json(rules),
        f"Loaded {entry_count} rule(s) across {len(rules)} language(s).",
    )


def apply_pronunciation_rules_json(raw_json):
    if PRONUNCIATION_REPOSITORY is None:
        return "{}", "Pronunciation dictionary is not configured."
    try:
        parsed_rules = PRONUNCIATION_REPOSITORY.parse_rules_json(raw_json)
        saved_rules, entry_count = _save_and_apply_pronunciation_rules(parsed_rules)
    except ValueError as exc:
        return str(raw_json or "{}"), f"Invalid dictionary JSON: {exc}"
    except Exception:
        logger.exception("Failed to apply pronunciation dictionary")
        return str(raw_json or "{}"), "Failed to apply dictionary. Check logs for details."
    return (
        PRONUNCIATION_REPOSITORY.to_pretty_json(saved_rules),
        f"Applied {entry_count} rule(s) across {len(saved_rules)} language(s).",
    )


def import_pronunciation_rules_json(uploaded_file):
    if PRONUNCIATION_REPOSITORY is None:
        return "{}", "Pronunciation dictionary is not configured."
    file_path = _resolve_uploaded_file_path(uploaded_file)
    if not file_path:
        return "{}", "No JSON file selected."
    try:
        parsed_rules = PRONUNCIATION_REPOSITORY.load_rules_from_file(file_path)
        saved_rules, entry_count = _save_and_apply_pronunciation_rules(parsed_rules)
    except ValueError as exc:
        return "{}", str(exc)
    except Exception:
        logger.exception("Failed to import pronunciation dictionary: %s", file_path)
        return "{}", "Failed to import dictionary. Check logs for details."
    return (
        PRONUNCIATION_REPOSITORY.to_pretty_json(saved_rules),
        f"Imported {entry_count} rule(s) across {len(saved_rules)} language(s).",
    )


def export_pronunciation_rules_json():
    if PRONUNCIATION_REPOSITORY is None:
        return None, "Pronunciation dictionary is not configured."
    try:
        rules = PRONUNCIATION_REPOSITORY.load_rules()
        json_text = PRONUNCIATION_REPOSITORY.to_pretty_json(rules)
        date_dir = os.path.join(
            CONFIG.output_dir_abs,
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


@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed):
    return APP_STATE.model_manager.get_model(True)(ps, ref_s, speed)


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
    _refresh_pronunciation_rules_from_store()
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.generate_first(
        text=text,
        voice=voice,
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
    _refresh_pronunciation_rules_from_store()
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.generate_first(
        text=text,
        voice=voice,
        speed=speed,
        use_gpu=False,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
        save_outputs=False,
        style_preset=style_preset,
    )[0]


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
    _refresh_pronunciation_rules_from_store()
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.tokenize_first(
        text=text,
        voice=voice,
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
    _refresh_pronunciation_rules_from_store()
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    yield from APP_STATE.generate_all(
        text=text,
        voice=voice,
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
    if MORPHOLOGY_REPOSITORY is None:
        return None, "Morphology DB is not configured."
    normalized_format = str(file_format or "ods").strip().lower().lstrip(".")
    try:
        if normalized_format == "csv":
            sheet_path = MORPHOLOGY_REPOSITORY.export_csv(
                dataset=dataset,
                output_dir=CONFIG.output_dir_abs,
            )
        elif normalized_format == "txt":
            sheet_path = MORPHOLOGY_REPOSITORY.export_txt(
                dataset=dataset,
                output_dir=CONFIG.output_dir_abs,
            )
        elif normalized_format in ("xlsx", "excel"):
            sheet_path = MORPHOLOGY_REPOSITORY.export_excel(
                dataset=dataset,
                output_dir=CONFIG.output_dir_abs,
            )
        elif normalized_format in ("docx", "word"):
            # Backward compatibility: old UI alias now points to Excel export.
            sheet_path = MORPHOLOGY_REPOSITORY.export_excel(
                dataset=dataset,
                output_dir=CONFIG.output_dir_abs,
            )
        elif normalized_format in ("ods", "spreadsheet"):
            sheet_path = MORPHOLOGY_REPOSITORY.export_spreadsheet(
                dataset=dataset,
                output_dir=CONFIG.output_dir_abs,
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
    return gr.update(value=[[]], headers=["No data"])


def _parse_row_payload(raw_json):
    raw_text = str(raw_json or "").strip()
    if not raw_text:
        raise ValueError("Row JSON is empty.")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Row JSON must be an object.")
    return payload


def _normalize_morph_dataset(dataset) -> str:
    return normalize_morphology_dataset(dataset)


def morphology_db_view(dataset="occurrences", limit=100, offset=0):
    if TTS_ONLY_MODE:
        return _empty_morphology_table_update(), "TTS-only mode is enabled. Morphology DB is disabled."
    if MORPHOLOGY_REPOSITORY is None:
        return _empty_morphology_table_update(), "Morphology DB is not configured."
    try:
        headers, rows = MORPHOLOGY_REPOSITORY.list_rows(
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
        return gr.update(value=[], headers=headers), "No rows found."
    return gr.update(value=rows, headers=headers), f"Loaded {len(rows)} row(s)."


def morphology_db_add(dataset, row_json, limit=100, offset=0):
    if TTS_ONLY_MODE:
        return _empty_morphology_table_update(), "TTS-only mode is enabled. Morphology DB is disabled."
    if MORPHOLOGY_REPOSITORY is None:
        return _empty_morphology_table_update(), "Morphology DB is not configured."
    if _normalize_morph_dataset(dataset) == "reviews":
        table_update, view_status = morphology_db_view(dataset, limit, offset)
        return table_update, f"Dataset reviews is read-only. {view_status}"
    try:
        payload = _parse_row_payload(row_json)
        inserted_id = MORPHOLOGY_REPOSITORY.insert_row(
            dataset=str(dataset or "occurrences"),
            payload=payload,
        )
        table_update, view_status = morphology_db_view(dataset, limit, offset)
        return table_update, f"Added row with id/key={inserted_id}. {view_status}"
    except Exception as exc:
        logger.exception("Morphology DB add failed")
        return _empty_morphology_table_update(), f"Add failed: {exc}"


def morphology_db_update(dataset, row_id, row_json, limit=100, offset=0):
    if TTS_ONLY_MODE:
        return _empty_morphology_table_update(), "TTS-only mode is enabled. Morphology DB is disabled."
    if MORPHOLOGY_REPOSITORY is None:
        return _empty_morphology_table_update(), "Morphology DB is not configured."
    if _normalize_morph_dataset(dataset) == "reviews":
        table_update, view_status = morphology_db_view(dataset, limit, offset)
        return table_update, f"Dataset reviews is read-only. {view_status}"
    try:
        payload = _parse_row_payload(row_json)
        updated = MORPHOLOGY_REPOSITORY.update_row(
            dataset=str(dataset or "occurrences"),
            row_id=str(row_id or ""),
            payload=payload,
        )
        table_update, view_status = morphology_db_view(dataset, limit, offset)
        return table_update, f"Updated {updated} row(s). {view_status}"
    except Exception as exc:
        logger.exception("Morphology DB update failed")
        return _empty_morphology_table_update(), f"Update failed: {exc}"


def morphology_db_delete(dataset, row_id, limit=100, offset=0):
    if TTS_ONLY_MODE:
        return _empty_morphology_table_update(), "TTS-only mode is enabled. Morphology DB is disabled."
    if MORPHOLOGY_REPOSITORY is None:
        return _empty_morphology_table_update(), "Morphology DB is not configured."
    if _normalize_morph_dataset(dataset) == "reviews":
        table_update, view_status = morphology_db_view(dataset, limit, offset)
        return table_update, f"Dataset reviews is read-only. {view_status}"
    try:
        deleted = MORPHOLOGY_REPOSITORY.delete_row(
            dataset=str(dataset or "occurrences"),
            row_id=str(row_id or ""),
        )
        table_update, view_status = morphology_db_view(dataset, limit, offset)
        return table_update, f"Deleted {deleted} row(s). {view_status}"
    except Exception as exc:
        logger.exception("Morphology DB delete failed")
        return _empty_morphology_table_update(), f"Delete failed: {exc}"


def _coerce_int(value, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _coerce_float(value, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def build_lesson_for_tts(
    raw_text,
    llm_base_url="",
    llm_api_key="",
    llm_model="",
    llm_temperature=None,
    llm_max_tokens=None,
    llm_timeout_seconds=None,
    llm_extra_instructions="",
):
    if _llm_requests_disabled():
        if TTS_ONLY_MODE:
            return "", "TTS-only mode is enabled. LLM requests are disabled."
        return "", "TTS + Morphology mode is enabled. LLM requests are disabled."
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return "", "Please provide raw text."

    base_url = (llm_base_url or CONFIG.lm_studio_base_url).strip()
    api_key = (llm_api_key or CONFIG.lm_studio_api_key).strip() or "lm-studio"
    model = (llm_model or CONFIG.lm_studio_model).strip()
    temperature = _coerce_float(
        llm_temperature,
        CONFIG.lm_studio_temperature,
        0.0,
        2.0,
    )
    max_tokens = _coerce_int(
        llm_max_tokens,
        CONFIG.lm_studio_max_tokens,
        64,
        32768,
    )
    timeout_seconds = _coerce_int(
        llm_timeout_seconds,
        CONFIG.lm_studio_timeout_seconds,
        0,
        86400,
    )
    extra_instructions = (llm_extra_instructions or "").strip()

    try:
        lesson = generate_lesson_text(
            LessonRequest(
                raw_text=raw_text,
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_instructions=extra_instructions,
            )
        )
    except LmStudioError as exc:
        logger.warning("LM Studio lesson generation failed: %s", exc)
        return "", f"LM Studio error: {exc}"
    except Exception:
        logger.exception("Unexpected LM Studio lesson generation failure")
        return "", "Unexpected LM Studio error. Check logs for details."

    logger.info(
        "LM Studio lesson generated: chars=%s model=%s base_url=%s",
        len(lesson),
        model,
        base_url,
    )
    return lesson, "Lesson generated. Review text and copy to TTS input if needed."


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
        morphology_db_add=morphology_db_add,
        morphology_db_update=morphology_db_update,
        morphology_db_delete=morphology_db_delete,
        load_pronunciation_rules=load_pronunciation_rules_json,
        apply_pronunciation_rules=apply_pronunciation_rules_json,
        import_pronunciation_rules=import_pronunciation_rules_json,
        export_pronunciation_rules=export_pronunciation_rules_json,
        build_lesson_for_tts=build_lesson_for_tts,
        set_tts_only_mode=set_tts_only_mode,
        set_llm_only_mode=set_llm_only_mode,
        tts_only_mode_default=TTS_ONLY_MODE,
        llm_only_mode_default=LLM_ONLY_MODE,
        choices=CHOICES,
    )
    MODEL_MANAGER = services.model_manager
    TEXT_NORMALIZER = services.text_normalizer
    AUDIO_WRITER = services.audio_writer
    MORPHOLOGY_REPOSITORY = services.morphology_repository
    PRONUNCIATION_REPOSITORY = services.pronunciation_repository
    APP_STATE = services.app_state
    HISTORY_REPOSITORY = services.history_repository
    HISTORY_SERVICE = services.history_service
    app = services.app
    API_OPEN = services.api_open
    MORPH_DEFAULT_EXPRESSION_EXTRACTOR = getattr(MORPHOLOGY_REPOSITORY, "expression_extractor", None)
    MORPH_DEFAULT_LM_VERIFY_ENABLED = bool(getattr(MORPHOLOGY_REPOSITORY, "lm_verify_enabled", False))
    _apply_runtime_modes()
else:
    logger.info("KOKORO_SKIP_APP_INIT enabled; skipping model and UI initialization")


def launch() -> None:
    if SKIP_APP_INIT:
        logger.info("KOKORO_SKIP_APP_INIT enabled; launch skipped")
        return
    logger.info("Launching Gradio app")
    queue_kwargs = {"api_open": API_OPEN}
    if CONFIG.default_concurrency_limit is not None:
        queue_kwargs["default_concurrency_limit"] = CONFIG.default_concurrency_limit
    queued = app.queue(**queue_kwargs)
    launch_params = inspect.signature(queued.launch).parameters
    launch_kwargs = {}

    # Gradio API visibility changed across major versions.
    if "show_api" in launch_params:
        launch_kwargs["show_api"] = API_OPEN
    elif "footer_links" in launch_params:
        launch_kwargs["footer_links"] = (
            ["api", "gradio", "settings"] if API_OPEN else ["gradio", "settings"]
        )

    if "ssr_mode" in launch_params:
        launch_kwargs["ssr_mode"] = SSR_MODE

    if "theme" in launch_params:
        launch_kwargs["theme"] = APP_THEME
    if "css" in launch_params:
        launch_kwargs["css"] = APP_CSS

    queued.launch(**launch_kwargs)


if __name__ == "__main__":
    launch()
