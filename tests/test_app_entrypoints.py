import os
from types import SimpleNamespace

os.environ.setdefault("KOKORO_SKIP_APP_INIT", "1")

import app
from kokoro_tts import main as app_main


class _State:
    def __init__(self):
        self.calls = []
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


def test_app_wrappers_delegate_to_state(monkeypatch):
    state = _State()
    monkeypatch.setattr(app, "APP_STATE", state)

    output, ps = app.generate_first("hello", voice="af_heart", use_gpu=False)
    assert output[0] == 24000
    assert ps == "ps"

    predicted = app.predict("hello", voice="af_heart")
    assert predicted[0] == 24000

    tokens = app.tokenize_first("hello", voice="af_heart")
    assert tokens == "tokens"

    streamed = list(app.generate_all("hello", voice="af_heart", use_gpu=False))
    assert streamed == [(24000, [1.0])]


def test_export_morphology_sheet_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", None)
    file_path, status = app.export_morphology_sheet()
    assert file_path is None
    assert "not configured" in status

    class Repo:
        def export_spreadsheet(self, dataset, output_dir):
            _ = (dataset, output_dir)
            return str(tmp_path / "sheet.ods")

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", Repo())
    file_path, status = app.export_morphology_sheet("lexemes")
    assert file_path and file_path.endswith(".ods")
    assert "Export ready" in status

    class FailingRepo:
        def export_spreadsheet(self, dataset, output_dir):
            _ = (dataset, output_dir)
            raise RuntimeError("boom")

    monkeypatch.setattr(app, "MORPHOLOGY_REPOSITORY", FailingRepo())
    file_path, status = app.export_morphology_sheet("lexemes")
    assert file_path is None
    assert "Export failed" in status


def test_launch_handles_skip_mode(monkeypatch):
    monkeypatch.setattr(app, "SKIP_APP_INIT", True)
    assert app.launch() is None


def test_launch_builds_queue_kwargs_for_gradio_versions(monkeypatch):
    launched = {}

    class QueuedShowApi:
        def launch(self, show_api=None, ssr_mode=None):
            launched["show_api"] = show_api
            launched["ssr_mode"] = ssr_mode

    class AppA:
        def queue(self, **kwargs):
            launched["queue_a"] = kwargs
            return QueuedShowApi()

    monkeypatch.setattr(app, "SKIP_APP_INIT", False)
    monkeypatch.setattr(app, "app", AppA())
    monkeypatch.setattr(app, "API_OPEN", True)
    monkeypatch.setattr(app, "SSR_MODE", True)
    monkeypatch.setattr(app, "CONFIG", SimpleNamespace(default_concurrency_limit=3))
    app.launch()
    assert launched["queue_a"]["api_open"] is True
    assert launched["queue_a"]["default_concurrency_limit"] == 3
    assert launched["show_api"] is True
    assert launched["ssr_mode"] is True

    class QueuedFooter:
        def launch(self, footer_links=None):
            launched["footer_links"] = footer_links

    class AppB:
        def queue(self, **kwargs):
            launched["queue_b"] = kwargs
            return QueuedFooter()

    monkeypatch.setattr(app, "app", AppB())
    monkeypatch.setattr(app, "API_OPEN", False)
    monkeypatch.setattr(app, "CONFIG", SimpleNamespace(default_concurrency_limit=None))
    app.launch()
    assert launched["queue_b"] == {"api_open": False}
    assert launched["footer_links"] == ["gradio", "settings"]


def test_forward_gpu_and_main_delegate(monkeypatch):
    state = _State()
    monkeypatch.setattr(app, "APP_STATE", state)
    result = app.forward_gpu("ps", "ref", 1.0)
    assert result == ("ps", "ref", 1.0)

    called = {}
    monkeypatch.setattr(app, "launch", lambda: called.setdefault("ok", True))
    app_main.main()
    assert called["ok"] is True
