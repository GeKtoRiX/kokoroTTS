"""Configuration loading from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .utils import parse_float_env, parse_int_env, resolve_path


def _parse_timeout_seconds_env(name: str, default: int) -> int:
    value = parse_int_env(name, default, min_value=0, max_value=86400)
    if value != 0 and value < 5:
        return 5
    return value


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
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str = ""
    lm_studio_timeout_seconds: int = 120
    lm_studio_temperature: float = 0.3
    lm_studio_max_tokens: int = 2048
    lm_verify_enabled: bool = False
    lm_verify_base_url: str = "http://127.0.0.1:1234/v1"
    lm_verify_api_key: str = "lm-studio"
    lm_verify_model: str = ""
    lm_verify_timeout_seconds: int = 30
    lm_verify_temperature: float = 0.0
    lm_verify_max_tokens: int = 2048
    lm_verify_max_retries: int = 2
    lm_verify_workers: int = 1
    morph_local_expressions_enabled: bool = False


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
    lm_studio_base_url = os.getenv(
        "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"
    ).strip()
    lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio").strip() or "lm-studio"
    lm_studio_model = os.getenv("LM_STUDIO_MODEL", "").strip()
    lm_studio_timeout_seconds = _parse_timeout_seconds_env(
        "LM_STUDIO_TIMEOUT_SECONDS",
        120,
    )
    lm_studio_temperature = parse_float_env(
        "LM_STUDIO_TEMPERATURE",
        0.3,
        min_value=0.0,
        max_value=2.0,
    )
    lm_studio_max_tokens = parse_int_env(
        "LM_STUDIO_MAX_TOKENS",
        2048,
        min_value=64,
        max_value=32768,
    )
    lm_verify_enabled = os.getenv("LM_VERIFY_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    lm_verify_base_url = os.getenv(
        "LM_VERIFY_BASE_URL", "http://127.0.0.1:1234/v1"
    ).strip()
    lm_verify_api_key = os.getenv("LM_VERIFY_API_KEY", "lm-studio").strip() or "lm-studio"
    lm_verify_model = os.getenv("LM_VERIFY_MODEL", "").strip()
    lm_verify_timeout_seconds = _parse_timeout_seconds_env(
        "LM_VERIFY_TIMEOUT_SECONDS",
        30,
    )
    lm_verify_temperature = parse_float_env(
        "LM_VERIFY_TEMPERATURE",
        0.0,
        min_value=0.0,
        max_value=2.0,
    )
    lm_verify_max_tokens = parse_int_env(
        "LM_VERIFY_MAX_TOKENS",
        2048,
        min_value=64,
        max_value=32768,
    )
    lm_verify_max_retries = parse_int_env(
        "LM_VERIFY_MAX_RETRIES",
        2,
        min_value=0,
        max_value=5,
    )
    lm_verify_workers = parse_int_env(
        "LM_VERIFY_WORKERS",
        1,
        min_value=1,
        max_value=4,
    )
    morph_local_expressions_enabled = os.getenv(
        "MORPH_LOCAL_EXPRESSIONS_ENABLED", "0"
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
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
        lm_studio_base_url=lm_studio_base_url,
        lm_studio_api_key=lm_studio_api_key,
        lm_studio_model=lm_studio_model,
        lm_studio_timeout_seconds=lm_studio_timeout_seconds,
        lm_studio_temperature=lm_studio_temperature,
        lm_studio_max_tokens=lm_studio_max_tokens,
        lm_verify_enabled=lm_verify_enabled,
        lm_verify_base_url=lm_verify_base_url,
        lm_verify_api_key=lm_verify_api_key,
        lm_verify_model=lm_verify_model,
        lm_verify_timeout_seconds=lm_verify_timeout_seconds,
        lm_verify_temperature=lm_verify_temperature,
        lm_verify_max_tokens=lm_verify_max_tokens,
        lm_verify_max_retries=lm_verify_max_retries,
        lm_verify_workers=lm_verify_workers,
        morph_local_expressions_enabled=morph_local_expressions_enabled,
    )
