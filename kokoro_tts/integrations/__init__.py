"""Integrations for external services and libraries."""

from .ffmpeg import configure_ffmpeg
from .model_manager import ModelManager
from .silero_manager import SileroManager
from .spaces_gpu import build_forward_gpu

__all__ = [
    "ModelManager",
    "SileroManager",
    "build_forward_gpu",
    "configure_ffmpeg",
]
