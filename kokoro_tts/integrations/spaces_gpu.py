"""Hugging Face Spaces GPU wrapper."""
from __future__ import annotations

from typing import Callable

import spaces
import torch

from .model_manager import ModelManager


def build_forward_gpu(
    model_manager: ModelManager,
) -> Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]:
    @spaces.GPU(duration=30)
    def forward_gpu(ps: torch.Tensor, ref_s: torch.Tensor, speed: float) -> torch.Tensor:
        return model_manager.get_model(True)(ps, ref_s, speed)

    return forward_gpu
