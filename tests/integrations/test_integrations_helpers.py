import os
import sys

import torch

from kokoro_tts.integrations.ffmpeg import configure_ffmpeg
from kokoro_tts.integrations.spaces_gpu import build_forward_gpu


class _Logger:
    def __init__(self):
        self.debugs = []

    def debug(self, message, *args):
        self.debugs.append(message % args if args else message)


def test_configure_ffmpeg_detects_binaries_in_prefix(monkeypatch, tmp_path):
    logger = _Logger()
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "ffmpeg.exe").write_bytes(b"")
    (scripts_dir / "ffprobe.exe").write_bytes(b"")

    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.delenv("FFPROBE_BINARY", raising=False)
    monkeypatch.setenv("PATH", "")

    configure_ffmpeg(logger, str(tmp_path / "project"))

    assert os.environ.get("FFMPEG_BINARY", "").endswith("ffmpeg.exe")
    assert os.environ.get("FFPROBE_BINARY", "").endswith("ffprobe.exe")
    assert str(scripts_dir) in os.environ.get("PATH", "")


def test_configure_ffmpeg_logs_when_missing(monkeypatch, tmp_path):
    logger = _Logger()
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "no-prefix"))
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.delenv("FFPROBE_BINARY", raising=False)

    configure_ffmpeg(logger, str(tmp_path / "project"))

    joined = "\n".join(logger.debugs)
    assert "FFMPEG_BINARY" in joined


def test_build_forward_gpu_wraps_model_call():
    class Manager:
        def get_model(self, use_gpu):
            assert use_gpu is True
            return lambda ps, ref_s, speed: ps + ref_s + speed

    forward = build_forward_gpu(Manager())
    output = forward(torch.tensor([1.0]), torch.tensor([2.0]), 3.0)

    assert torch.equal(output, torch.tensor([6.0]))
