"""Integrations for external services and libraries."""

from .ffmpeg import configure_ffmpeg
from .lm_studio import (
    LmStudioError,
    LessonRequest,
    PosVerifyRequest,
    generate_lesson_text,
    verify_pos_with_context,
)
from .model_manager import ModelManager
from .spaces_gpu import build_forward_gpu

__all__ = [
    "ModelManager",
    "build_forward_gpu",
    "configure_ffmpeg",
    "generate_lesson_text",
    "verify_pos_with_context",
    "LessonRequest",
    "PosVerifyRequest",
    "LmStudioError",
]
