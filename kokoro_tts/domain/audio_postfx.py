"""Audio post-processing helpers for generated waveforms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AudioPostFxSettings:
    """Post-processing controls for generated audio."""

    enabled: bool = False
    trim_enabled: bool = True
    trim_threshold_db: float = -42.0
    trim_keep_ms: int = 25
    fade_in_ms: int = 12
    fade_out_ms: int = 40
    crossfade_ms: int = 25
    loudness_enabled: bool = True
    loudness_target_lufs: float = -16.0
    loudness_true_peak_db: float = -1.0


def _audio_tensor(audio: torch.Tensor | object) -> torch.Tensor:
    if isinstance(audio, torch.Tensor):
        tensor = audio.detach().flatten()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        return tensor
    return torch.as_tensor(audio, dtype=torch.float32).flatten()


def trim_silence(
    audio: torch.Tensor | object,
    sample_rate: int,
    threshold_db: float,
    keep_ms: int,
) -> torch.Tensor:
    """Trim leading/trailing silence and preserve a safety margin."""

    tensor = _audio_tensor(audio)
    if tensor.numel() == 0 or int(sample_rate) <= 0:
        return tensor
    try:
        threshold = float(threshold_db)
    except (TypeError, ValueError):
        return tensor
    amplitude = 10.0 ** (threshold / 20.0)
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        return tensor

    non_silent = torch.nonzero(torch.abs(tensor) > amplitude, as_tuple=False).flatten()
    if non_silent.numel() == 0:
        return tensor

    keep_samples = max(0, int(round(max(0.0, float(keep_ms)) * int(sample_rate) / 1000.0)))
    start = max(0, int(non_silent[0].item()) - keep_samples)
    end = min(int(tensor.numel()), int(non_silent[-1].item()) + 1 + keep_samples)
    if end <= start:
        return tensor

    trimmed = tensor[start:end]
    if trimmed.numel() == 0:
        return tensor
    return trimmed


def apply_fade(
    audio: torch.Tensor | object,
    sample_rate: int,
    fade_in_ms: int,
    fade_out_ms: int,
) -> torch.Tensor:
    """Apply linear fade in/out to the waveform edges."""

    tensor = _audio_tensor(audio)
    if tensor.numel() == 0 or int(sample_rate) <= 0:
        return tensor

    fade_in_samples = max(0, int(round(max(0.0, float(fade_in_ms)) * int(sample_rate) / 1000.0)))
    fade_out_samples = max(0, int(round(max(0.0, float(fade_out_ms)) * int(sample_rate) / 1000.0)))
    if fade_in_samples <= 0 and fade_out_samples <= 0:
        return tensor

    out = tensor.clone()
    total = int(out.numel())
    if fade_in_samples > 0:
        count = min(total, fade_in_samples)
        if count == 1:
            ramp = torch.ones(1, dtype=out.dtype, device=out.device)
        else:
            ramp = torch.linspace(0.0, 1.0, steps=count, dtype=out.dtype, device=out.device)
        out[:count] = out[:count] * ramp
    if fade_out_samples > 0:
        count = min(total, fade_out_samples)
        ramp = torch.linspace(1.0, 0.0, steps=count, dtype=out.dtype, device=out.device)
        out[-count:] = out[-count:] * ramp
    return out


def crossfade_join(
    previous_audio: torch.Tensor | object,
    next_audio: torch.Tensor | object,
    crossfade_samples: int,
) -> torch.Tensor:
    """Join two waveforms with an equal-power crossfade."""

    previous = _audio_tensor(previous_audio)
    current = _audio_tensor(next_audio)
    if previous.numel() == 0:
        return current
    if current.numel() == 0:
        return previous
    if current.device != previous.device or current.dtype != previous.dtype:
        current = current.to(device=previous.device, dtype=previous.dtype)
    overlap = min(
        int(crossfade_samples),
        int(previous.numel()),
        int(current.numel()),
    )
    if overlap < 2:
        return torch.cat([previous, current])

    phase = torch.linspace(
        0.0,
        math.pi / 2.0,
        steps=overlap,
        dtype=previous.dtype,
        device=previous.device,
    )
    fade_out = torch.cos(phase)
    fade_in = torch.sin(phase)
    prev_tail = previous[-overlap:]
    cur_head = current[:overlap]
    mixed = prev_tail * fade_out + cur_head * fade_in
    return torch.cat([previous[:-overlap], mixed, current[overlap:]])
