"""Audio output management and file conversions."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
from datetime import datetime

import torch

from ..constants import OUTPUT_FORMATS
from ..domain.audio_postfx import AudioPostFxSettings


class AudioWriter:
    def __init__(self, output_dir: str, sample_rate: int, logger) -> None:
        self.output_dir = output_dir
        self.output_dir_abs = os.path.abspath(output_dir)
        self.sample_rate = sample_rate
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
        self.ffmpeg_path = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
        self.can_convert = bool(self.ffmpeg_path)
        self.logger.info("Output dir: %s", self.output_dir)
        if self.can_convert:
            self.logger.debug("FFmpeg available: %s", self.ffmpeg_path)
        else:
            self.logger.warning(
                "FFmpeg not found; output formats other than wav will fall back to wav"
            )

    def _sanitize_voice_id(self, voice: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", voice).strip("_")
        return safe or "voice"

    def _dated_output_dirs(self) -> tuple[str, str]:
        date_dir = os.path.join(self.output_dir, datetime.now().strftime("%Y-%m-%d"))
        records_dir = os.path.join(date_dir, "records")
        vocabulary_dir = os.path.join(date_dir, "vocabulary")
        os.makedirs(records_dir, exist_ok=True)
        os.makedirs(vocabulary_dir, exist_ok=True)
        return records_dir, vocabulary_dir

    def resolve_output_format(self, output_format: str) -> str:
        fmt = (output_format or "wav").strip().lower().lstrip(".")
        if fmt not in OUTPUT_FORMATS:
            fmt = "wav"
        if fmt != "wav" and not self.can_convert:
            self.logger.warning(
                "Requested output format %s but ffmpeg not available; falling back to wav",
                fmt,
            )
            fmt = "wav"
        return fmt

    def _ensure_extension(self, path: str, output_format: str) -> str:
        base, ext = os.path.splitext(path)
        if ext.lower().lstrip(".") != output_format:
            return f"{base}.{output_format}"
        return path

    def save_wav(self, path: str, audio_tensor) -> None:
        tensor = audio_tensor if isinstance(audio_tensor, torch.Tensor) else torch.as_tensor(audio_tensor)
        tensor = tensor.detach().cpu().flatten()
        tensor = torch.clamp(tensor, -1.0, 1.0)
        int16 = (tensor * 32767.0).to(torch.int16).numpy()
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16.tobytes())

    def _convert_with_ffmpeg(self, src_path: str, dst_path: str) -> None:
        ffmpeg = self.ffmpeg_path or shutil.which("ffmpeg")
        if not ffmpeg:
            raise FileNotFoundError("ffmpeg not available")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", src_path, dst_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _apply_loudness_normalization(
        self,
        path: str,
        postfx_settings: AudioPostFxSettings | None,
    ) -> str:
        if postfx_settings is None:
            self.logger.debug("Loudness normalization skipped: postfx settings are not provided")
            return path
        if not bool(postfx_settings.enabled):
            self.logger.debug("Loudness normalization skipped: postfx is disabled")
            return path
        if not bool(postfx_settings.loudness_enabled):
            self.logger.debug("Loudness normalization skipped: loudness step is disabled")
            return path

        ffmpeg = self.ffmpeg_path or shutil.which("ffmpeg")
        if not ffmpeg:
            self.logger.warning("Loudness normalization skipped: ffmpeg is not available")
            return path

        target_lufs = float(postfx_settings.loudness_target_lufs)
        true_peak = float(postfx_settings.loudness_true_peak_db)
        filter_expr = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11"
        output_dir = os.path.dirname(path) or self.output_dir
        _, ext = os.path.splitext(path)
        tmp_handle = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext or ".wav",
            dir=output_dir,
        )
        tmp_path = tmp_handle.name
        tmp_handle.close()
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-af",
                    filter_expr,
                    tmp_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            os.replace(tmp_path, path)
            self.logger.debug(
                "Loudness normalization complete: target_lufs=%s true_peak=%s path=%s",
                target_lufs,
                true_peak,
                path,
            )
        except Exception:
            self.logger.exception("Loudness normalization failed: %s", path)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return path

    def save_audio(
        self,
        path: str,
        audio_tensor,
        output_format: str,
        *,
        postfx_settings: AudioPostFxSettings | None = None,
    ) -> str:
        output_format = self.resolve_output_format(output_format)
        path = self._ensure_extension(path, output_format)
        if output_format == "wav":
            self.save_wav(path, audio_tensor)
            return self._apply_loudness_normalization(path, postfx_settings)
        tmp_handle = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
            dir=self.output_dir,
        )
        tmp_path = tmp_handle.name
        tmp_handle.close()
        self.save_wav(tmp_path, audio_tensor)
        try:
            self._convert_with_ffmpeg(tmp_path, path)
        except Exception:
            self.logger.exception(
                "Failed to convert audio to %s; keeping WAV",
                output_format,
            )
            fallback_path = os.path.splitext(path)[0] + ".wav"
            os.replace(tmp_path, fallback_path)
            return self._apply_loudness_normalization(fallback_path, postfx_settings)
        os.remove(tmp_path)
        return self._apply_loudness_normalization(path, postfx_settings)

    def build_output_paths(
        self,
        voice: str,
        parts_count: int,
        output_format: str,
    ) -> list[str]:
        output_format = self.resolve_output_format(output_format)
        records_dir, _ = self._dated_output_dirs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_voice = self._sanitize_voice_id(voice)
        suffix = uuid.uuid4().hex[:8]
        if parts_count <= 0:
            return []
        if parts_count == 1:
            return [os.path.join(records_dir, f"{timestamp}_{safe_voice}_{suffix}.{output_format}")]
        return [
            os.path.join(
                records_dir,
                f"{timestamp}_{safe_voice}_{suffix}_part{index:02d}.{output_format}",
            )
            for index in range(1, parts_count + 1)
        ]
