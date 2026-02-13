import os

from kokoro_tts.config import load_config
from kokoro_tts.utils import parse_int_env, resolve_path


def test_resolve_path_handles_relative_and_absolute(tmp_path):
    base_dir = str(tmp_path)
    relative = "nested/file.txt"
    absolute = str(tmp_path / "absolute.txt")

    assert resolve_path(relative, base_dir) == os.path.join(base_dir, relative)
    assert resolve_path(absolute, base_dir) == absolute


def test_parse_int_env_applies_default_and_bounds(monkeypatch):
    monkeypatch.setenv("INT_ENV_TEST", "not-a-number")
    assert parse_int_env("INT_ENV_TEST", 7, min_value=1, max_value=10) == 7

    monkeypatch.setenv("INT_ENV_TEST", "100")
    assert parse_int_env("INT_ENV_TEST", 7, min_value=1, max_value=10) == 10

    monkeypatch.setenv("INT_ENV_TEST", "-5")
    assert parse_int_env("INT_ENV_TEST", 7, min_value=1, max_value=10) == 1


def test_load_config_reads_env_and_sets_duplicate_limits(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("FILE_LOG_LEVEL", "error")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("KOKORO_REPO_ID", "repo/test")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("MAX_CHUNK_CHARS", "42")  # below min -> clamped
    monkeypatch.setenv("HISTORY_LIMIT", "100")  # above max -> clamped
    monkeypatch.setenv("NORMALIZE_TIMES", "0")
    monkeypatch.setenv("NORMALIZE_NUMBERS", "false")
    monkeypatch.setenv("DEFAULT_OUTPUT_FORMAT", "mp3")
    monkeypatch.setenv("DEFAULT_CONCURRENCY_LIMIT", "3")
    monkeypatch.setenv("LOG_EVERY_N_SEGMENTS", "0")  # below min -> clamped
    monkeypatch.setenv("MORPH_DB_ENABLED", "yes")
    monkeypatch.setenv("MORPH_DB_PATH", str(tmp_path / "data" / "morph.sqlite3"))
    monkeypatch.setenv("MORPH_DB_TABLE_PREFIX", "prefix_")
    monkeypatch.setenv("SPACE_ID", "hexgrad/Kokoro-TTS")

    config = load_config()

    assert config.log_level == "WARNING"
    assert config.file_log_level == "ERROR"
    assert os.path.isdir(config.log_dir)
    assert config.repo_id == "repo/test"
    assert config.output_dir_abs == os.path.abspath(str(tmp_path / "outputs"))
    assert config.max_chunk_chars == 50
    assert config.history_limit == 20
    assert config.normalize_times is False
    assert config.normalize_numbers is False
    assert config.default_output_format == "mp3"
    assert config.default_concurrency_limit == 3
    assert config.log_segment_every == 1
    assert config.morph_db_enabled is True
    assert config.morph_db_table_prefix == "prefix_"
    assert config.is_duplicate is False
    assert config.char_limit == 5000


def test_load_config_for_duplicate_space_has_no_char_limit(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "someone/else")
    monkeypatch.setenv("DEFAULT_CONCURRENCY_LIMIT", "0")

    config = load_config()

    assert config.is_duplicate is True
    assert config.char_limit is None
    assert config.default_concurrency_limit is None
