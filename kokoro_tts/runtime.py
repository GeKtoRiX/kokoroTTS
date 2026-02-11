"""Runtime environment detection."""
from __future__ import annotations

import torch

CUDA_AVAILABLE = torch.cuda.is_available()
