"""Audio player runtime state container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class AudioPlayerRuntimeState:
    """Mutable runtime state used by the audio player feature."""

    loaded_path: Path | None = None
    pcm_data: np.ndarray | None = None
    sample_rate: int = 0
    total_frames: int = 0
    current_frame: int = 0
    media_length_ms: int = 0
    backend: str = "vlc"
    sd_start_frame: int = 0
    sd_started_at: float = 0.0
    is_playing: bool = False
    is_paused: bool = False
    tick_job: str | None = None
    seek_dragging: bool = False
    seek_programmatic: bool = False
    queue_index: int | None = None
    waveform: np.ndarray | None = None
    state_path: Path | Any = Path(".audio_player_state.json")
    restore_path: Path | None = None
    restore_position_seconds: float = 0.0
    shortcuts_bound: bool = False
    seek_step_seconds: float = 5.0
