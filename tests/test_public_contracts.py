import importlib.util
import inspect
import os
from pathlib import Path

os.environ.setdefault("KOKORO_SKIP_APP_INIT", "1")

import app


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
        "build_lesson_for_tts": [
            "raw_text",
            "llm_base_url",
            "llm_api_key",
            "llm_model",
            "llm_temperature",
            "llm_max_tokens",
            "llm_timeout_seconds",
            "llm_extra_instructions",
        ],
    }
    for function_name, expected in expected_parameters.items():
        function = getattr(app, function_name)
        actual = list(inspect.signature(function).parameters)
        assert actual == expected

    assert list(inspect.signature(app.launch).parameters) == []


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
    assert "--verify-llm" in help_text
    assert "--json" in help_text
