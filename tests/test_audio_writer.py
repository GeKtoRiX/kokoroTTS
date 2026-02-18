import os
import wave
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from kokoro_tts.domain.audio_postfx import AudioPostFxSettings
from kokoro_tts.storage.audio_writer import AudioWriter


class _Logger:
    def __init__(self):
        self.infos = []
        self.debugs = []
        self.warnings = []
        self.exceptions = []

    def info(self, message, *args):
        self.infos.append(message % args if args else message)

    def debug(self, message, *args):
        self.debugs.append(message % args if args else message)

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)


def test_audio_writer_save_wav_and_build_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    logger = _Logger()
    writer = AudioWriter(str(tmp_path), sample_rate=24000, logger=logger)

    wav_path = tmp_path / "sample.wav"
    writer.save_wav(str(wav_path), torch.tensor([0.0, 0.5, -0.5], dtype=torch.float32))

    assert wav_path.is_file()
    with wave.open(str(wav_path), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2

    assert writer.build_output_paths("af_heart", 0, "wav") == []
    single_path = writer.build_output_paths("af_heart", 1, "wav")
    multi_paths = writer.build_output_paths("af_heart", 3, "wav")
    assert len(single_path) == 1
    assert len(multi_paths) == 3

    date_dir = tmp_path / datetime.now().strftime("%Y-%m-%d")
    assert (date_dir / "records").is_dir()
    assert (date_dir / "vocabulary").is_dir()
    assert Path(single_path[0]).parent.name == "records"
    assert all(Path(path).parent.name == "records" for path in multi_paths)


def test_audio_writer_resolve_format_and_conversion_fallback(tmp_path):
    logger = _Logger()
    writer = AudioWriter(str(tmp_path), sample_rate=24000, logger=logger)
    writer.can_convert = True
    writer.ffmpeg_path = "fake-ffmpeg"

    def _boom(_src, _dst):
        raise RuntimeError("conversion failed")

    writer._convert_with_ffmpeg = _boom  # type: ignore[method-assign]
    out_path = writer.save_audio(
        str(tmp_path / "audio.mp3"),
        torch.tensor([0.1, -0.1]),
        output_format="mp3",
    )

    assert out_path.endswith(".wav")
    assert os.path.isfile(out_path)
    assert logger.exceptions


def test_audio_writer_non_wav_without_ffmpeg_falls_back_to_wav(tmp_path):
    logger = _Logger()
    writer = AudioWriter(str(tmp_path), sample_rate=24000, logger=logger)
    writer.can_convert = False
    writer.ffmpeg_path = None

    resolved = writer.resolve_output_format("ogg")
    output = writer.save_audio(
        str(tmp_path / "clip.ogg"),
        torch.tensor([0.2, -0.2]),
        output_format="ogg",
    )

    assert resolved == "wav"
    assert output.endswith(".wav")
    assert os.path.isfile(output)


def test_audio_writer_loudnorm_runs_when_enabled(tmp_path, monkeypatch):
    logger = _Logger()
    writer = AudioWriter(str(tmp_path), sample_rate=24000, logger=logger)
    writer.ffmpeg_path = "ffmpeg"
    writer.can_convert = True
    calls = []

    def _run(cmd, check, stdout, stderr):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("kokoro_tts.storage.audio_writer.subprocess.run", _run)
    settings = AudioPostFxSettings(
        enabled=True,
        loudness_enabled=True,
        loudness_target_lufs=-16.0,
        loudness_true_peak_db=-1.0,
    )
    output = writer.save_audio(
        str(tmp_path / "loud.wav"),
        torch.tensor([0.1, -0.1], dtype=torch.float32),
        output_format="wav",
        postfx_settings=settings,
    )

    assert output.endswith(".wav")
    assert calls
    assert any("-af" in cmd for cmd in calls)
    assert any("loudnorm=I=-16.0:TP=-1.0:LRA=11" in cmd for cmd in calls)


def test_audio_writer_loudnorm_skips_without_ffmpeg(tmp_path, monkeypatch):
    logger = _Logger()
    writer = AudioWriter(str(tmp_path), sample_rate=24000, logger=logger)
    writer.ffmpeg_path = None
    writer.can_convert = False
    monkeypatch.setattr("kokoro_tts.storage.audio_writer.shutil.which", lambda _name: None)
    settings = AudioPostFxSettings(enabled=True, loudness_enabled=True)
    output = writer.save_audio(
        str(tmp_path / "noffmpeg.wav"),
        torch.tensor([0.1, -0.1], dtype=torch.float32),
        output_format="wav",
        postfx_settings=settings,
    )

    assert output.endswith(".wav")
    assert os.path.isfile(output)
    assert any("Loudness normalization skipped" in message for message in logger.warnings)


def test_audio_writer_loudnorm_failure_keeps_output(tmp_path, monkeypatch):
    logger = _Logger()
    writer = AudioWriter(str(tmp_path), sample_rate=24000, logger=logger)
    writer.ffmpeg_path = "ffmpeg"
    writer.can_convert = True

    def _boom(*_args, **_kwargs):
        raise RuntimeError("loudnorm failed")

    monkeypatch.setattr("kokoro_tts.storage.audio_writer.subprocess.run", _boom)
    settings = AudioPostFxSettings(enabled=True, loudness_enabled=True)
    output = writer.save_audio(
        str(tmp_path / "fail.wav"),
        torch.tensor([0.1, -0.1], dtype=torch.float32),
        output_format="wav",
        postfx_settings=settings,
    )

    assert output.endswith(".wav")
    assert os.path.isfile(output)
    assert logger.exceptions
