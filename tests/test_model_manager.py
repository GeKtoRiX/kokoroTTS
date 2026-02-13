from kokoro_tts.integrations.model_manager import ModelManager


class _Logger:
    def __init__(self):
        self.infos = []
        self.debugs = []
        self.exceptions = []

    def info(self, message, *args):
        self.infos.append(message % args if args else message)

    def debug(self, message, *args):
        self.debugs.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)


def test_model_manager_caches_models_and_pipelines(monkeypatch):
    class FakeModel:
        def __init__(self, repo_id):
            self.repo_id = repo_id
            self.device = None
            self.evaluated = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    class FakeLexicon:
        def __init__(self):
            self.golds = {}

    class FakeG2P:
        def __init__(self):
            self.lexicon = FakeLexicon()

    class FakePipeline:
        def __init__(self, lang_code, repo_id, model):
            self.lang_code = lang_code
            self.repo_id = repo_id
            self.model = model
            self.g2p = FakeG2P()
            self.loaded = []

        def load_voice(self, voice):
            self.loaded.append(voice)
            return {"voice": voice}

    monkeypatch.setattr("kokoro_tts.integrations.model_manager.KModel", FakeModel)
    monkeypatch.setattr("kokoro_tts.integrations.model_manager.KPipeline", FakePipeline)

    manager = ModelManager(repo_id="repo/x", cuda_available=True, logger=_Logger())

    cpu_model_a = manager.get_model(False)
    cpu_model_b = manager.get_model(False)
    gpu_model = manager.get_model(True)
    assert cpu_model_a is cpu_model_b
    assert cpu_model_a.device == "cpu"
    assert gpu_model.device == "cuda"
    assert cpu_model_a.evaluated is True

    pipe_a1 = manager.get_pipeline("af_heart")
    pipe_a2 = manager.get_pipeline("af_bella")
    pipe_b = manager.get_pipeline("bf_emma")
    assert pipe_a1 is pipe_a2
    assert pipe_a1.g2p.lexicon.golds["kokoro"]
    assert pipe_b.g2p.lexicon.golds["kokoro"]

    pack_1 = manager.get_voice_pack("af_heart")
    pack_2 = manager.get_voice_pack("af_heart")
    assert pack_1 is pack_2
    assert pipe_a1.loaded == ["af_heart"]


def test_model_manager_propagates_pipeline_and_voice_load_errors(monkeypatch):
    class FakeModel:
        def __init__(self, repo_id):
            self.repo_id = repo_id

        def to(self, _device):
            return self

        def eval(self):
            return self

    class FailingPipeline:
        def __init__(self, lang_code, repo_id, model):
            _ = (lang_code, repo_id, model)
            raise RuntimeError("pipeline init failed")

    monkeypatch.setattr("kokoro_tts.integrations.model_manager.KModel", FakeModel)
    monkeypatch.setattr("kokoro_tts.integrations.model_manager.KPipeline", FailingPipeline)

    manager = ModelManager(repo_id="repo/x", cuda_available=False, logger=_Logger())
    _ = manager.get_model(True)  # forced CPU because cuda unavailable

    try:
        manager.get_pipeline("af_heart")
    except RuntimeError as exc:
        assert "pipeline init failed" in str(exc)
    else:
        raise AssertionError("Expected pipeline init failure")


def test_model_manager_voice_pack_failure(monkeypatch):
    class FakeModel:
        def __init__(self, repo_id):
            self.repo_id = repo_id

        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeLexicon:
        def __init__(self):
            self.golds = {}

    class FakeG2P:
        def __init__(self):
            self.lexicon = FakeLexicon()

    class Pipeline:
        def __init__(self, lang_code, repo_id, model):
            _ = (lang_code, repo_id, model)
            self.g2p = FakeG2P()

        def load_voice(self, _voice):
            raise RuntimeError("voice load failed")

    monkeypatch.setattr("kokoro_tts.integrations.model_manager.KModel", FakeModel)
    monkeypatch.setattr("kokoro_tts.integrations.model_manager.KPipeline", Pipeline)

    manager = ModelManager(repo_id="repo/x", cuda_available=True, logger=_Logger())

    try:
        manager.get_voice_pack("af_heart")
    except RuntimeError as exc:
        assert "voice load failed" in str(exc)
    else:
        raise AssertionError("Expected voice load failure")
