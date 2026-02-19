import sys
from types import SimpleNamespace

import pytest

from kokoro_tts.integrations.silero_manager import SileroManager


class _FakeModel:
    def __init__(self):
        self.speakers = ["ru_alpha", "ru_beta"]
        self.calls = []

    def apply_tts(self, *, text, speaker, sample_rate):
        self.calls.append((text, speaker, sample_rate))
        return [0.1, 0.2, 0.3]


def test_silero_manager_discovers_voices_and_synthesizes(monkeypatch, tmp_path):
    model = _FakeModel()

    def _silero_tts(language="ru", speaker="v5_cis_base", **_kwargs):
        assert language == "ru"
        assert speaker == "v5_cis_base"
        return model, "example"

    monkeypatch.setitem(sys.modules, "silero", SimpleNamespace(silero_tts=_silero_tts))
    manager = SileroManager(
        model_id="v5_cis_base",
        cache_dir=str(tmp_path / "torch-cache"),
        cpu_only=True,
        sample_rate=24000,
    )

    voices = manager.discover_voice_items()
    assert len(voices) == 2
    assert voices[0][1].startswith("r_")
    assert manager.supports_voice(voices[0][1]) is True

    audio = manager.synthesize(
        "test",
        voice_id=voices[0][1],
        sample_rate=24000,
    )
    assert int(audio.numel()) == 3
    assert model.calls


def test_silero_manager_unknown_voice_raises(monkeypatch, tmp_path):
    model = _FakeModel()
    monkeypatch.setitem(
        sys.modules,
        "silero",
        SimpleNamespace(silero_tts=lambda **_kwargs: (model, "example")),
    )
    manager = SileroManager(cache_dir=str(tmp_path / "torch-cache"))
    manager.discover_voice_items()
    with pytest.raises(ValueError):
        manager.synthesize("hello", voice_id="r_missing", sample_rate=24000)
