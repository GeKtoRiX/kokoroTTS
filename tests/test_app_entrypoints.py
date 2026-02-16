import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KOKORO_SKIP_APP_INIT", "1")

if "torch" not in sys.modules:
    sys.modules["torch"] = SimpleNamespace(
        __version__="0.0-test",
        cuda=SimpleNamespace(is_available=lambda: False),
    )

import app
from kokoro_tts import main as app_main


class _State:
    def __init__(self):
        self.calls = []
        self.aux_features = []
        self.model_manager = SimpleNamespace(
            get_model=lambda _use_gpu: (lambda ps, ref_s, speed: (ps, ref_s, speed))
        )

    def generate_first(self, **kwargs):
        self.calls.append(("generate_first", kwargs))
        return ((24000, [0.1, 0.2]), "ps")

    def tokenize_first(self, **kwargs):
        self.calls.append(("tokenize_first", kwargs))
        return "tokens"

    def generate_all(self, **kwargs):
        self.calls.append(("generate_all", kwargs))
        yield (24000, [1.0])

    def set_aux_features_enabled(self, enabled):
        self.aux_features.append(bool(enabled))


def test_app_wrappers_delegate_to_state(monkeypatch):
    state = _State()
    monkeypatch.setattr(app, "APP_STATE", state)

    output, ps = app.generate_first(
        "hello",
        voice="af_heart",
        use_gpu=False,
        style_preset="narrator",
    )
    assert output[0] == 24000
    assert ps == "ps"

    predicted = app.predict("hello", voice="af_heart", style_preset="energetic")
    assert predicted[0] == 24000

    tokens = app.tokenize_first("hello", voice="af_heart", style_preset="neutral")
    assert tokens == "tokens"

    streamed = list(app.generate_all("hello", voice="af_heart", use_gpu=False))
    assert streamed == [(24000, [1.0])]

    generate_calls = [kwargs for name, kwargs in state.calls if name == "generate_first"]
    assert generate_calls[0]["style_preset"] == "narrator"
    assert generate_calls[1]["style_preset"] == "energetic"
    tokenize_call = next(kwargs for name, kwargs in state.calls if name == "tokenize_first")
    stream_call = next(kwargs for name, kwargs in state.calls if name == "generate_all")
    assert tokenize_call["style_preset"] == "neutral"
    assert stream_call["style_preset"] == "neutral"


def test_export_morphology_sheet_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", None)
    file_path, status = app.export_morphology_sheet()
    assert file_path is None
    assert "not configured" in status

    class Repo:
        def export_csv(self, dataset, output_dir):
            _ = (dataset, output_dir)
            return str(tmp_path / "sheet.csv")

        def export_txt(self, dataset, output_dir):
            _ = (dataset, output_dir)
            return str(tmp_path / "sheet.txt")

        def export_excel(self, dataset, output_dir):
            _ = (dataset, output_dir)
            return str(tmp_path / "sheet.xlsx")

        def export_spreadsheet(self, dataset, output_dir):
            _ = (dataset, output_dir)
            return str(tmp_path / "sheet.ods")

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", Repo())
    file_path, status = app.export_morphology_sheet("lexemes", "ods")
    assert file_path and file_path.endswith(".ods")
    assert "Export ready" in status

    file_path, status = app.export_morphology_sheet("lexemes", "csv")
    assert file_path and file_path.endswith(".csv")
    assert "Export ready" in status

    file_path, status = app.export_morphology_sheet("lexemes", "txt")
    assert file_path and file_path.endswith(".txt")
    assert "Export ready" in status

    file_path, status = app.export_morphology_sheet("lexemes", "xlsx")
    assert file_path and file_path.endswith(".xlsx")
    assert "Export ready" in status

    file_path, status = app.export_morphology_sheet("lexemes", "word")
    assert file_path and file_path.endswith(".xlsx")
    assert "Export ready" in status

    file_path, status = app.export_morphology_sheet("lexemes", "docx")
    assert file_path and file_path.endswith(".xlsx")
    assert "Export ready" in status

    file_path, status = app.export_morphology_sheet("lexemes", "unknown")
    assert file_path is None
    assert "Unsupported export format" in status

    class FailingRepo:
        def export_spreadsheet(self, dataset, output_dir):
            _ = (dataset, output_dir)
            raise RuntimeError("boom")

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", FailingRepo())
    file_path, status = app.export_morphology_sheet("lexemes", "ods")
    assert file_path is None
    assert "Export failed" in status


def test_morphology_db_wrappers_block_reviews_dataset(monkeypatch):
    class Repo:
        def list_rows(self, dataset, limit, offset):
            _ = (dataset, limit, offset)
            return ["id", "status"], [["1", "success"]]

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", Repo())

    _, status_add = app.morphology_db_add("reviews", '{"source":"manual"}', 20, 0)
    _, status_update = app.morphology_db_update("reviews", "1", '{"status":"failed"}', 20, 0)
    _, status_delete = app.morphology_db_delete("reviews", "1", 20, 0)

    assert "read-only" in status_add.lower()
    assert "read-only" in status_update.lower()
    assert "read-only" in status_delete.lower()


def test_launch_handles_skip_mode(monkeypatch):
    monkeypatch.setattr(app, "SKIP_APP_INIT", True)
    assert app.launch() is None


def test_launch_delegates_to_desktop_app(monkeypatch):
    launched = {}

    class DesktopApp:
        def launch(self):
            launched["ok"] = True

    monkeypatch.setattr(app, "SKIP_APP_INIT", False)
    monkeypatch.setattr(app, "app", DesktopApp())
    app.launch()
    assert launched["ok"] is True


def test_forward_gpu_and_main_delegate(monkeypatch):
    state = _State()
    monkeypatch.setattr(app, "APP_STATE", state)
    result = app.forward_gpu("ps", "ref", 1.0)
    assert result == ("ps", "ref", 1.0)

    called = {}
    monkeypatch.setattr(app, "launch", lambda: called.setdefault("ok", True))
    app_main.main()
    assert called["ok"] is True


def test_pronunciation_dictionary_handlers(monkeypatch, tmp_path):
    class Repo:
        def __init__(self):
            self.saved = {}

        def load_rules(self):
            return {"a": {"OpenAI": "oʊpənˈeɪ aɪ"}}

        def parse_rules_json(self, raw_json):
            _ = raw_json
            return {"a": {"Kokoro": "kˈOkəɹO"}}

        def save_rules(self, rules):
            self.saved = dict(rules)
            return dict(rules)

        def to_pretty_json(self, rules=None):
            payload = rules if rules is not None else self.load_rules()
            return str(payload)

        def load_rules_from_file(self, file_path):
            _ = file_path
            return {"b": {"OpenAI": "əʊpənˈeɪ aɪ"}}

    class Manager:
        def __init__(self):
            self.calls = []

        def set_pronunciation_rules(self, rules):
            self.calls.append(rules)
            return sum(len(items) for items in rules.values())

    repo = Repo()
    manager = Manager()
    monkeypatch.setattr(app, "PRONUNCIATION_REPOSITORY", repo)
    monkeypatch.setattr(app, "MODEL_MANAGER", manager)
    monkeypatch.setattr(app, "APP_STATE", None)
    monkeypatch.setattr(
        app,
        "CONFIG",
        SimpleNamespace(output_dir_abs=str(tmp_path)),
    )

    text, status = app.load_pronunciation_rules_json()
    assert "OpenAI" in text
    assert "Loaded 1 rule" in status

    text, status = app.apply_pronunciation_rules_json('{"a":{"x":"y"}}')
    assert "Kokoro" in text
    assert "Applied 1 rule" in status

    text, status = app.import_pronunciation_rules_json("import.json")
    assert "OpenAI" in text
    assert "Imported 1 rule" in status

    export_path, status = app.export_pronunciation_rules_json()
    assert export_path is not None
    assert Path(export_path).is_file()
    assert "Export ready" in status


def test_tts_only_mode_blocks_db_and_llm_features(monkeypatch):
    state = _State()
    monkeypatch.setattr(app, "APP_STATE", state)
    monkeypatch.setattr(app, "TTS_ONLY_MODE", False)
    monkeypatch.setattr(app, "LLM_ONLY_MODE", False)

    status = app.set_tts_only_mode(True)
    assert "enabled" in status.lower()
    assert state.aux_features and state.aux_features[-1] is False

    _, db_status = app.morphology_db_view()
    assert "tts-only mode" in db_status.lower()

    lesson_text, lesson_status = app.build_lesson_for_tts("hello world")
    assert lesson_text == ""
    assert "disabled" in lesson_status.lower()


def test_llm_only_mode_disables_llm_and_keeps_morphology_enabled(monkeypatch):
    state = _State()

    class Repo:
        def __init__(self):
            self.lm_verify_enabled = True
            self.lm_verifier = lambda payload: payload
            self.expression_extractor = lambda _text: []

    repo = Repo()
    default_extractor = repo.expression_extractor

    monkeypatch.setattr(app, "APP_STATE", state)
    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", repo)
    monkeypatch.setattr(app, "MORPH_DEFAULT_EXPRESSION_EXTRACTOR", default_extractor)
    monkeypatch.setattr(app, "MORPH_DEFAULT_LM_VERIFY_ENABLED", True)
    monkeypatch.setattr(app, "TTS_ONLY_MODE", False)
    monkeypatch.setattr(app, "LLM_ONLY_MODE", False)

    status = app.set_llm_only_mode(True)
    assert "enabled" in status.lower()
    assert state.aux_features and state.aux_features[-1] is True
    assert repo.lm_verify_enabled is False
    assert repo.expression_extractor is app.extract_english_expressions

    lesson_text, lesson_status = app.build_lesson_for_tts("hello world")
    assert lesson_text == ""
    assert "disabled" in lesson_status.lower()

    status = app.set_llm_only_mode(False)
    assert "disabled" in status.lower() or "enabled" in status.lower()
    assert repo.lm_verify_enabled is True
    assert repo.expression_extractor is default_extractor
