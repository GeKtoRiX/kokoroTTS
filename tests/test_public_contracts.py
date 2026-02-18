import importlib.util
import inspect
import os
from pathlib import Path
import pytest

os.environ.setdefault("KOKORO_SKIP_APP_INIT", "1")

import app
from kokoro_tts.config import AppConfig
from kokoro_tts.application import bootstrap as app_bootstrap


def test_app_public_wrapper_signatures_are_stable():
    expected_parameters = {
        "generate_first": [
            "text",
            "voice",
            "mix_enabled",
            "voice_mix",
            "speed",
            "use_gpu",
            "pause_seconds",
            "output_format",
            "normalize_times_enabled",
            "normalize_numbers_enabled",
            "style_preset",
        ],
        "predict": [
            "text",
            "voice",
            "mix_enabled",
            "voice_mix",
            "speed",
            "normalize_times_enabled",
            "normalize_numbers_enabled",
            "style_preset",
        ],
        "tokenize_first": [
            "text",
            "voice",
            "mix_enabled",
            "voice_mix",
            "speed",
            "normalize_times_enabled",
            "normalize_numbers_enabled",
            "style_preset",
        ],
        "generate_all": [
            "text",
            "voice",
            "mix_enabled",
            "voice_mix",
            "speed",
            "use_gpu",
            "pause_seconds",
            "normalize_times_enabled",
            "normalize_numbers_enabled",
            "style_preset",
        ],
    }
    for function_name, expected in expected_parameters.items():
        function = getattr(app, function_name)
        actual = list(inspect.signature(function).parameters)
        assert actual == expected

    assert list(inspect.signature(app.launch).parameters) == []


def test_removed_llm_public_surfaces_are_absent():
    assert not hasattr(app, "build_lesson_for_tts")
    assert not hasattr(app, "set_llm_only_mode")
    assert not hasattr(app_bootstrap, "build_lm_verifier")
    with pytest.raises(ModuleNotFoundError):
        __import__("kokoro_tts.integrations.lm_studio")

    removed_config_fields = {
        "lm_studio_base_url",
        "lm_studio_api_key",
        "lm_studio_model",
        "lm_studio_timeout_seconds",
        "lm_studio_temperature",
        "lm_studio_max_tokens",
        "lm_verify_enabled",
        "lm_verify_base_url",
        "lm_verify_api_key",
        "lm_verify_model",
        "lm_verify_timeout_seconds",
        "lm_verify_temperature",
        "lm_verify_max_tokens",
        "lm_verify_max_retries",
        "lm_verify_workers",
    }
    assert removed_config_fields.isdisjoint(set(AppConfig.__annotations__))


def test_check_english_lexemes_cli_flags_are_stable():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_english_lexemes.py"
    )
    spec = importlib.util.spec_from_file_location("check_english_lexemes", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_parser()
    help_text = parser.format_help()
    assert "--text" in help_text
    assert "--verify-llm" not in help_text
    assert "--json" in help_text
