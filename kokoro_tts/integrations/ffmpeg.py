"""FFmpeg detection and environment setup."""

from __future__ import annotations

import os
import sys
from typing import Optional


def configure_ffmpeg(logger, base_dir: str) -> None:
    ffmpeg_env = os.getenv("FFMPEG_BINARY")
    ffprobe_env = os.getenv("FFPROBE_BINARY")
    candidates: list[str] = []
    if sys.prefix:
        candidates.append(os.path.join(sys.prefix, "Scripts"))
    candidates.append(os.path.join(base_dir, ".venv", "Scripts"))
    for folder in candidates:
        if not folder:
            continue
        ffmpeg_path = os.path.join(folder, "ffmpeg.exe")
        ffprobe_path = os.path.join(folder, "ffprobe.exe")
        updated = False
        if not ffmpeg_env and os.path.isfile(ffmpeg_path):
            os.environ["FFMPEG_BINARY"] = ffmpeg_path
            ffmpeg_env = ffmpeg_path
            updated = True
        if not ffprobe_env and os.path.isfile(ffprobe_path):
            os.environ["FFPROBE_BINARY"] = ffprobe_path
            ffprobe_env = ffprobe_path
            updated = True
        if updated and folder not in os.environ.get("PATH", ""):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    if ffmpeg_env:
        logger.debug("FFMPEG_BINARY=%s", ffmpeg_env)
    else:
        logger.debug("FFMPEG_BINARY not set; streaming audio may fail without ffmpeg")
    if ffprobe_env:
        logger.debug("FFPROBE_BINARY=%s", ffprobe_env)
