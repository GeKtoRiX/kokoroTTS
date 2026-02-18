"""Configuration loading from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .utils import parse_float_env, parse_int_env, resolve_path


@dataclass(frozen=True)
class AppConfig:
    log_level: str
    file_log_level: str
    log_dir: str
    log_file: str
    repo_id: str
    output_dir: str
    output_dir_abs: str
    max_chunk_chars: int
    history_limit: int
    normalize_times: bool
    normalize_numbers: bool
    default_output_format: str
    default_concurrency_limit: Optional[int]
    log_segment_every: int
    morph_db_enabled: bool
    morph_db_path: str
    morph_db_table_prefix: str
    space_id: str
    is_duplicate: bool
    char_limit: Optional[int]
    morph_local_expressions_enabled: bool = False
    torch_num_threads: Optional[int] = None
    torch_num_interop_threads: Optional[int] = None
    tts_prewarm_enabled: bool = True
    tts_prewarm_async: bool = False
    tts_prewarm_voice: str = "af_heart"
    tts_prewarm_style: str = "neutral"
    morph_async_ingest: bool = False
    morph_async_max_pending: int = 8
    postfx_enabled: bool = False
    postfx_trim_enabled: bool = True
    postfx_trim_threshold_db: float = -42.0
    postfx_trim_keep_ms: int = 25
    postfx_fade_in_ms: int = 12
    postfx_fade_out_ms: int = 40
    postfx_crossfade_ms: int = 25
    postfx_loudness_enabled: bool = True
    postfx_loudness_target_lufs: float = -16.0
    postfx_loudness_true_peak_db: float = -1.0


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def load_config() -> AppConfig:
    base_dir = str(Path(__file__).resolve().parents[1])
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    file_log_level = os.getenv("FILE_LOG_LEVEL", "DEBUG").upper()
    log_dir = resolve_path(os.getenv("LOG_DIR", "logs"), base_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
    )
    repo_id = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
    output_dir = resolve_path(os.getenv("OUTPUT_DIR", "outputs"), base_dir)
    output_dir_abs = os.path.abspath(output_dir)
    max_chunk_chars = parse_int_env("MAX_CHUNK_CHARS", 500, min_value=50, max_value=2000)
    history_limit = parse_int_env("HISTORY_LIMIT", 5, min_value=0, max_value=20)
    normalize_times = os.getenv("NORMALIZE_TIMES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    normalize_numbers = os.getenv("NORMALIZE_NUMBERS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    default_output_format = os.getenv("DEFAULT_OUTPUT_FORMAT", "wav").strip().lower()
    default_concurrency_limit = parse_int_env("DEFAULT_CONCURRENCY_LIMIT", 0, min_value=0)
    if default_concurrency_limit == 0:
        default_concurrency_limit = None
    log_segment_every = parse_int_env(
        "LOG_EVERY_N_SEGMENTS", 10, min_value=1, max_value=1000
    )
    morph_db_enabled = os.getenv("MORPH_DB_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    morph_db_path = resolve_path(
        os.getenv("MORPH_DB_PATH", "data/morphology.sqlite3").strip(),
        base_dir,
    )
    morph_db_table_prefix = os.getenv("MORPH_DB_TABLE_PREFIX", "morph_").strip()
    space_id = os.getenv("SPACE_ID", "")
    is_duplicate = not space_id.startswith("hexgrad/")
    char_limit = None if is_duplicate else 5000
    morph_local_expressions_enabled = os.getenv(
        "MORPH_LOCAL_EXPRESSIONS_ENABLED", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    torch_num_threads = parse_int_env("TORCH_NUM_THREADS", 0, min_value=0, max_value=512)
    if torch_num_threads == 0:
        torch_num_threads = None
    torch_num_interop_threads = parse_int_env(
        "TORCH_NUM_INTEROP_THREADS",
        0,
        min_value=0,
        max_value=512,
    )
    if torch_num_interop_threads == 0:
        torch_num_interop_threads = None
    tts_prewarm_enabled = _env_flag("TTS_PREWARM_ENABLED", "1")
    tts_prewarm_async = _env_flag("TTS_PREWARM_ASYNC", "0")
    tts_prewarm_voice = os.getenv("TTS_PREWARM_VOICE", "af_heart").strip() or "af_heart"
    tts_prewarm_style = os.getenv("TTS_PREWARM_STYLE", "neutral").strip().lower() or "neutral"
    morph_async_ingest = _env_flag("MORPH_ASYNC_INGEST", "0")
    morph_async_max_pending = parse_int_env(
        "MORPH_ASYNC_MAX_PENDING",
        8,
        min_value=1,
        max_value=256,
    )
    postfx_enabled = _env_flag("POSTFX_ENABLED", "0")
    postfx_trim_enabled = _env_flag("POSTFX_TRIM_ENABLED", "1")
    postfx_trim_threshold_db = parse_float_env(
        "POSTFX_TRIM_THRESHOLD_DB",
        -42.0,
        min_value=-90.0,
        max_value=-1.0,
    )
    postfx_trim_keep_ms = parse_int_env(
        "POSTFX_TRIM_KEEP_MS",
        25,
        min_value=0,
        max_value=2000,
    )
    postfx_fade_in_ms = parse_int_env(
        "POSTFX_FADE_IN_MS",
        12,
        min_value=0,
        max_value=5000,
    )
    postfx_fade_out_ms = parse_int_env(
        "POSTFX_FADE_OUT_MS",
        40,
        min_value=0,
        max_value=5000,
    )
    postfx_crossfade_ms = parse_int_env(
        "POSTFX_CROSSFADE_MS",
        25,
        min_value=0,
        max_value=2000,
    )
    postfx_loudness_enabled = _env_flag("POSTFX_LOUDNESS_ENABLED", "1")
    postfx_loudness_target_lufs = parse_float_env(
        "POSTFX_LOUDNESS_TARGET_LUFS",
        -16.0,
        min_value=-40.0,
        max_value=-5.0,
    )
    postfx_loudness_true_peak_db = parse_float_env(
        "POSTFX_LOUDNESS_TRUE_PEAK_DB",
        -1.0,
        min_value=-9.0,
        max_value=0.0,
    )
    return AppConfig(
        log_level=log_level,
        file_log_level=file_log_level,
        log_dir=log_dir,
        log_file=log_file,
        repo_id=repo_id,
        output_dir=output_dir,
        output_dir_abs=output_dir_abs,
        max_chunk_chars=max_chunk_chars,
        history_limit=history_limit,
        normalize_times=normalize_times,
        normalize_numbers=normalize_numbers,
        default_output_format=default_output_format,
        default_concurrency_limit=default_concurrency_limit,
        log_segment_every=log_segment_every,
        morph_db_enabled=morph_db_enabled,
        morph_db_path=morph_db_path,
        morph_db_table_prefix=morph_db_table_prefix,
        space_id=space_id,
        is_duplicate=is_duplicate,
        char_limit=char_limit,
        morph_local_expressions_enabled=morph_local_expressions_enabled,
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
        tts_prewarm_enabled=tts_prewarm_enabled,
        tts_prewarm_async=tts_prewarm_async,
        tts_prewarm_voice=tts_prewarm_voice,
        tts_prewarm_style=tts_prewarm_style,
        morph_async_ingest=morph_async_ingest,
        morph_async_max_pending=morph_async_max_pending,
        postfx_enabled=postfx_enabled,
        postfx_trim_enabled=postfx_trim_enabled,
        postfx_trim_threshold_db=postfx_trim_threshold_db,
        postfx_trim_keep_ms=postfx_trim_keep_ms,
        postfx_fade_in_ms=postfx_fade_in_ms,
        postfx_fade_out_ms=postfx_fade_out_ms,
        postfx_crossfade_ms=postfx_crossfade_ms,
        postfx_loudness_enabled=postfx_loudness_enabled,
        postfx_loudness_target_lufs=postfx_loudness_target_lufs,
        postfx_loudness_true_peak_db=postfx_loudness_true_peak_db,
    )
