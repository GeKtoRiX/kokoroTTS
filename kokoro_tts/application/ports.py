"""Application-level ports for generation and tokenization flows."""

from __future__ import annotations

from typing import Generator, Protocol


class KokoroTtsPort(Protocol):
    """Port abstraction for local or remote TTS execution."""

    def generate_first(
        self,
        text,
        voice="af_heart",
        mix_enabled=False,
        voice_mix=None,
        speed=1,
        use_gpu=True,
        pause_seconds=0.0,
        output_format="wav",
        normalize_times_enabled=None,
        normalize_numbers_enabled=None,
        style_preset="neutral",
    ): ...

    def predict(
        self,
        text,
        voice="af_heart",
        mix_enabled=False,
        voice_mix=None,
        speed=1,
        normalize_times_enabled=None,
        normalize_numbers_enabled=None,
        style_preset="neutral",
    ): ...

    def tokenize_first(
        self,
        text,
        voice="af_heart",
        mix_enabled=False,
        voice_mix=None,
        speed=1,
        normalize_times_enabled=None,
        normalize_numbers_enabled=None,
        style_preset="neutral",
    ): ...

    def generate_all(
        self,
        text,
        voice="af_heart",
        mix_enabled=False,
        voice_mix=None,
        speed=1,
        use_gpu=True,
        pause_seconds=0.0,
        normalize_times_enabled=None,
        normalize_numbers_enabled=None,
        style_preset="neutral",
    ) -> Generator[tuple[int, object], None, None]: ...
