import sys
from types import SimpleNamespace

if "torch" not in sys.modules:
    sys.modules["torch"] = SimpleNamespace(
        __version__="0.0-test",
        cuda=SimpleNamespace(is_available=lambda: False),
    )

from kokoro_tts.application import bootstrap


def _minimal_config(**overrides):
    base = {
        "repo_id": "hexgrad/Kokoro-82M",
        "char_limit": 5000,
        "normalize_times": True,
        "normalize_numbers": True,
        "output_dir": "outputs",
        "max_chunk_chars": 500,
        "log_segment_every": 10,
        "output_dir_abs": "outputs_abs",
        "history_limit": 5,
        "morph_db_enabled": True,
        "morph_db_path": "data/morphology.sqlite3",
        "morph_db_table_prefix": "morph_",
        "morph_local_expressions_enabled": True,
        "morph_async_ingest": True,
        "morph_async_max_pending": 8,
        "default_output_format": "wav",
        "space_id": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _Logger:
    def __init__(self):
        self.warning_messages = []
        self.info_messages = []

    def warning(self, message, *args):
        self.warning_messages.append(message % args if args else message)

    def info(self, message, *args):
        self.info_messages.append(message % args if args else message)


def test_initialize_app_services_builds_bundle(monkeypatch):
    calls = {}

    class FakePronRepo:
        def __init__(self, path, logger_instance=None):
            calls["pron_repo_init"] = (path, logger_instance)
            self.path = path

        def load_rules(self):
            return {"a": {"OpenAI": "phoneme"}}

    class FakeModelManager:
        def __init__(self, repo_id, cuda_available, logger, pronunciation_rules=None):
            calls["model_manager_init"] = (
                repo_id,
                cuda_available,
                logger,
                pronunciation_rules,
            )

    class FakeAudioWriter:
        def __init__(self, output_dir, sample_rate, logger):
            calls["audio_writer_init"] = (output_dir, sample_rate, logger)

    class FakeMorphRepo:
        def __init__(self, **kwargs):
            calls["morph_repo_init"] = kwargs

    class FakeUiHooks:
        def __init__(self, **kwargs):
            calls["ui_hooks_init"] = kwargs

    class FakeState:
        def __init__(self, *args, **kwargs):
            calls["state_init"] = (args, kwargs)
            self.last_saved_paths = []

    class FakeHistoryRepo:
        def __init__(self, output_dir_abs, logger):
            calls["history_repo_init"] = (output_dir_abs, logger)

    class FakeHistoryService:
        def __init__(self, history_limit, repository, state, logger):
            calls["history_service_init"] = (history_limit, repository, state, logger)

    def fake_create_tkinter_app(**kwargs):
        calls["create_tkinter_app"] = kwargs
        return "APP"

    monkeypatch.setattr(bootstrap, "PronunciationRepository", FakePronRepo)
    monkeypatch.setattr(bootstrap, "ModelManager", FakeModelManager)
    monkeypatch.setattr(bootstrap, "AudioWriter", FakeAudioWriter)
    monkeypatch.setattr(bootstrap, "MorphologyRepository", FakeMorphRepo)
    monkeypatch.setattr(bootstrap, "UiHooks", FakeUiHooks)
    monkeypatch.setattr(bootstrap, "KokoroState", FakeState)
    monkeypatch.setattr(bootstrap, "HistoryRepository", FakeHistoryRepo)
    monkeypatch.setattr(bootstrap, "HistoryService", FakeHistoryService)
    monkeypatch.setattr(bootstrap, "create_tkinter_app", fake_create_tkinter_app)
    monkeypatch.setattr(bootstrap, "build_forward_gpu", lambda _model_manager: "forward_gpu")

    logger = _Logger()
    config = _minimal_config()
    handlers = {
        "generate_first": lambda *args, **kwargs: None,
        "tokenize_first": lambda *args, **kwargs: None,
        "generate_all": lambda *args, **kwargs: iter(()),
        "predict": lambda *args, **kwargs: None,
        "export_morphology_sheet": lambda *args, **kwargs: None,
        "morphology_db_view": lambda *args, **kwargs: None,
        "load_pronunciation_rules": lambda *args, **kwargs: None,
        "apply_pronunciation_rules": lambda *args, **kwargs: None,
        "import_pronunciation_rules": lambda *args, **kwargs: None,
        "export_pronunciation_rules": lambda *args, **kwargs: None,
    }

    bundle = bootstrap.initialize_app_services(
        config=config,
        cuda_available=False,
        logger=logger,
        pronunciation_rules_path="data/pronunciation_rules.json",
        choices={"Voice Label": "af_heart"},
        **handlers,
    )

    assert bundle.app == "APP"
    assert calls["model_manager_init"][0] == "hexgrad/Kokoro-82M"
    assert calls["audio_writer_init"][0] == "outputs"
    postfx_settings = calls["state_init"][1].get("postfx_settings")
    assert postfx_settings is not None
    assert postfx_settings.enabled is False
    assert "create_tkinter_app" in calls
