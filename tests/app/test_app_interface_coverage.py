import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("KOKORO_SKIP_APP_INIT", "1")

if "torch" not in sys.modules:
    sys.modules["torch"] = SimpleNamespace(
        __version__="0.0-test",
        cuda=SimpleNamespace(is_available=lambda: False),
    )

import app


class _Logger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.exceptions = []

    def info(self, message, *args):
        self.infos.append(message % args if args else message)

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)


def test_configure_torch_runtime_success_and_exception_paths(monkeypatch):
    logger = _Logger()
    calls = {"threads": 0, "interop": 0}

    monkeypatch.setattr(app, "logger", logger)
    monkeypatch.setattr(
        app,
        "CONFIG",
        SimpleNamespace(torch_num_threads=4, torch_num_interop_threads=2),
    )
    monkeypatch.setattr(
        app,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda _value: calls.__setitem__("threads", calls["threads"] + 1),
            set_num_interop_threads=lambda _value: calls.__setitem__(
                "interop", calls["interop"] + 1
            ),
        ),
    )
    app._configure_torch_runtime()
    assert calls == {"threads": 1, "interop": 1}
    assert any("Torch CPU threads set" in line for line in logger.infos)
    assert any("Torch interop threads set" in line for line in logger.infos)

    monkeypatch.setattr(
        app,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda _value: (_ for _ in ()).throw(RuntimeError("threads-fail")),
            set_num_interop_threads=lambda _value: (_ for _ in ()).throw(
                RuntimeError("interop-fail")
            ),
        ),
    )
    app._configure_torch_runtime()
    assert any("TORCH_NUM_THREADS" in line for line in logger.exceptions)
    assert any("TORCH_NUM_INTEROP_THREADS" in line for line in logger.exceptions)


def test_current_legacy_wrappers_cover_all_accessors(monkeypatch):
    context = SimpleNamespace(
        model_manager="context_model",
        text_normalizer="context_normalizer",
        audio_writer="context_writer",
        history_repository="context_history_repo",
        history_service="context_history_service",
        morphology_repository="context_morph_repo",
        pronunciation_repository="context_pron_repo",
        app_state="context_state",
        app="context_app",
        morph_default_expression_extractor="context_extractor",
        tts_port=None,
    )
    monkeypatch.setattr(app, "APP_CONTEXT", context)
    monkeypatch.setattr(app, "MODEL_MANAGER", "legacy_model")
    monkeypatch.setattr(app, "TEXT_NORMALIZER", "legacy_normalizer")
    monkeypatch.setattr(app, "AUDIO_WRITER", "legacy_writer")
    monkeypatch.setattr(app, "HISTORY_REPOSITORY", "legacy_history_repo")
    monkeypatch.setattr(app, "HISTORY_SERVICE", "legacy_history_service")
    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", "legacy_morph_repo")
    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", "legacy_pron_repo")
    monkeypatch.setattr(app, "APP_STATE", "legacy_state")
    monkeypatch.setattr(app, "app", "legacy_app")
    monkeypatch.setattr(app, "MORPH_DEFAULT_EXPRESSION_EXTRACTOR", "legacy_extractor")

    assert app._current_model_manager() == "legacy_model"
    assert app._current_text_normalizer() == "legacy_normalizer"
    assert app._current_audio_writer() == "legacy_writer"
    assert app._current_history_repository() == "legacy_history_repo"
    assert app._current_history_service() == "legacy_history_service"
    assert app._current_morphology_repository() == "legacy_morph_repo"
    assert app._current_pronunciation_repository() == "legacy_pron_repo"
    assert app._current_app_state() == "legacy_state"
    assert app._current_desktop_app() == "legacy_app"
    assert app._current_morph_default_extractor() == "legacy_extractor"


def test_prewarm_runtime_covers_guard_async_success_and_failure(monkeypatch):
    logger = _Logger()
    monkeypatch.setattr(app, "logger", logger)
    monkeypatch.setattr(app, "SKIP_APP_INIT", True)
    monkeypatch.setattr(app, "_PREWARM_STARTED", False)
    monkeypatch.setattr(
        app, "CONFIG", SimpleNamespace(tts_prewarm_enabled=True, tts_prewarm_async=False)
    )
    app._prewarm_runtime()
    assert app._PREWARM_STARTED is False

    monkeypatch.setattr(app, "SKIP_APP_INIT", False)
    monkeypatch.setattr(app, "_PREWARM_STARTED", False)
    monkeypatch.setattr(
        app, "CONFIG", SimpleNamespace(tts_prewarm_enabled=False, tts_prewarm_async=False)
    )
    app._prewarm_runtime()
    assert app._PREWARM_STARTED is False

    monkeypatch.setattr(
        app, "CONFIG", SimpleNamespace(tts_prewarm_enabled=True, tts_prewarm_async=False)
    )
    monkeypatch.setattr(app, "_PREWARM_STARTED", True)
    app._prewarm_runtime()
    assert app._PREWARM_STARTED is True

    monkeypatch.setattr(app, "_PREWARM_STARTED", False)
    monkeypatch.setattr(app, "_current_app_state", lambda: None)
    app._prewarm_runtime()
    assert app._PREWARM_STARTED is True

    monkeypatch.setattr(app, "_PREWARM_STARTED", False)
    monkeypatch.setattr(app, "_current_app_state", lambda: SimpleNamespace(prewarm_inference=None))
    app._prewarm_runtime()
    assert app._PREWARM_STARTED is True

    calls = []
    monkeypatch.setattr(app, "_PREWARM_STARTED", False)
    monkeypatch.setattr(
        app,
        "CONFIG",
        SimpleNamespace(
            tts_prewarm_enabled=True,
            tts_prewarm_async=False,
            tts_prewarm_voice="af_heart",
            tts_prewarm_style="narrator",
        ),
    )
    monkeypatch.setattr(app, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(app, "normalize_voice_input", lambda value: f"normalized:{value}")
    monkeypatch.setattr(
        app,
        "_current_app_state",
        lambda: SimpleNamespace(
            prewarm_inference=lambda **kwargs: calls.append(kwargs),
        ),
    )
    app._prewarm_runtime()
    assert calls and calls[0]["voice"] == "normalized:af_heart"
    assert any("TTS prewarm complete" in line for line in logger.infos)

    monkeypatch.setattr(app, "_PREWARM_STARTED", False)
    monkeypatch.setattr(
        app,
        "_current_app_state",
        lambda: SimpleNamespace(
            prewarm_inference=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("prewarm-fail")),
        ),
    )
    app._prewarm_runtime()
    assert any("TTS prewarm failed" in line for line in logger.exceptions)

    class _InlineThread:
        def __init__(self, target, name=None, daemon=None):
            self.target = target
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True
            self.target()

    monkeypatch.setattr(app, "_PREWARM_STARTED", False)
    monkeypatch.setattr(
        app,
        "CONFIG",
        SimpleNamespace(
            tts_prewarm_enabled=True,
            tts_prewarm_async=True,
            tts_prewarm_voice="af_heart",
            tts_prewarm_style="neutral",
        ),
    )
    monkeypatch.setattr(app.threading, "Thread", _InlineThread)
    monkeypatch.setattr(
        app,
        "_current_app_state",
        lambda: SimpleNamespace(prewarm_inference=lambda **kwargs: calls.append(kwargs)),
    )
    app._prewarm_runtime()
    assert any("Started asynchronous TTS prewarm" in line for line in logger.infos)


def test_refresh_pronunciation_rules_shared_unchanged_and_changed(monkeypatch):
    set_calls = []

    class _Manager:
        def set_pronunciation_rules(self, rules):
            set_calls.append(rules)

    manager = _Manager()

    monkeypatch.setattr(app, "MODEL_MANAGER", manager)
    monkeypatch.setattr(app, "APP_STATE", None)
    monkeypatch.setattr(
        app,
        "PRONUNCIATION_REPOSITORY",
        SimpleNamespace(load_rules_shared=lambda: ({"a": {"x": "y"}}, False)),
    )
    app._refresh_pronunciation_rules_from_store()
    assert set_calls == []

    monkeypatch.setattr(
        app,
        "PRONUNCIATION_REPOSITORY",
        SimpleNamespace(load_rules_shared=lambda: ({"a": {"x": "y"}}, True)),
    )
    app._refresh_pronunciation_rules_from_store()
    assert set_calls == [{"a": {"x": "y"}}]


def test_shutdown_runtime_handles_absent_and_failing_state(monkeypatch):
    logger = _Logger()
    monkeypatch.setattr(app, "logger", logger)

    monkeypatch.setattr(app, "_current_app_state", lambda: None)
    app._shutdown_runtime()

    called = {"wait": None}
    monkeypatch.setattr(
        app,
        "_current_app_state",
        lambda: SimpleNamespace(shutdown=lambda *, wait: called.__setitem__("wait", wait)),
    )
    app._shutdown_runtime()
    assert called["wait"] is True

    monkeypatch.setattr(
        app,
        "_current_app_state",
        lambda: SimpleNamespace(
            shutdown=lambda *, wait: (_ for _ in ()).throw(RuntimeError("shutdown-fail"))
        ),
    )
    app._shutdown_runtime()
    assert any("Runtime shutdown failed" in line for line in logger.exceptions)


def test_forward_gpu_and_launch_raise_when_runtime_not_initialized(monkeypatch):
    monkeypatch.setattr(app, "APP_STATE", None)
    monkeypatch.setattr(app.APP_CONTEXT, "app_state", None)
    with pytest.raises(RuntimeError, match="App state is not initialized"):
        app.forward_gpu("ps", "ref", 1.0)

    monkeypatch.setattr(app, "SKIP_APP_INIT", False)
    monkeypatch.setattr(app, "app", None)
    monkeypatch.setattr(app.APP_CONTEXT, "app", None)
    with pytest.raises(RuntimeError, match="Desktop app is not initialized"):
        app.launch()
