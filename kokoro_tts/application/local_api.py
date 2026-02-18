"""Local single-process implementation of the application TTS port."""

from __future__ import annotations

from typing import Callable, Generator

from ..domain.voice import resolve_voice
from .ports import KokoroTtsPort


class LocalKokoroApi(KokoroTtsPort):
    """Adapt KokoroState to an explicit application port."""

    def __init__(
        self,
        *,
        state_provider: Callable[[], object | None],
        refresh_pronunciation_rules: Callable[[], None],
        default_use_gpu: bool,
    ) -> None:
        self._state_provider = state_provider
        self._refresh_pronunciation_rules = refresh_pronunciation_rules
        self._default_use_gpu = bool(default_use_gpu)

    def _state(self):
        state = self._state_provider()
        if state is None:
            raise RuntimeError("App state is not initialized.")
        return state

    def generate_first(
        self,
        text,
        voice="af_heart",
        mix_enabled=False,
        voice_mix=None,
        speed=1,
        use_gpu=None,
        pause_seconds=0.0,
        output_format="wav",
        normalize_times_enabled=None,
        normalize_numbers_enabled=None,
        style_preset="neutral",
    ):
        self._refresh_pronunciation_rules()
        resolved_voice = resolve_voice(voice, voice_mix, mix_enabled)
        if use_gpu is None:
            use_gpu = self._default_use_gpu
        return self._state().generate_first(
            text=text,
            voice=resolved_voice,
            speed=speed,
            use_gpu=use_gpu,
            pause_seconds=pause_seconds,
            output_format=output_format,
            normalize_times_enabled=normalize_times_enabled,
            normalize_numbers_enabled=normalize_numbers_enabled,
            style_preset=style_preset,
        )

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
    ):
        self._refresh_pronunciation_rules()
        resolved_voice = resolve_voice(voice, voice_mix, mix_enabled)
        return self._state().generate_first(
            text=text,
            voice=resolved_voice,
            speed=speed,
            use_gpu=False,
            normalize_times_enabled=normalize_times_enabled,
            normalize_numbers_enabled=normalize_numbers_enabled,
            save_outputs=False,
            style_preset=style_preset,
        )[0]

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
    ):
        self._refresh_pronunciation_rules()
        resolved_voice = resolve_voice(voice, voice_mix, mix_enabled)
        return self._state().tokenize_first(
            text=text,
            voice=resolved_voice,
            speed=speed,
            normalize_times_enabled=normalize_times_enabled,
            normalize_numbers_enabled=normalize_numbers_enabled,
            style_preset=style_preset,
        )

    def generate_all(
        self,
        text,
        voice="af_heart",
        mix_enabled=False,
        voice_mix=None,
        speed=1,
        use_gpu=None,
        pause_seconds=0.0,
        normalize_times_enabled=None,
        normalize_numbers_enabled=None,
        style_preset="neutral",
    ) -> Generator[tuple[int, object], None, None]:
        self._refresh_pronunciation_rules()
        resolved_voice = resolve_voice(voice, voice_mix, mix_enabled)
        if use_gpu is None:
            use_gpu = self._default_use_gpu
        yield from self._state().generate_all(
            text=text,
            voice=resolved_voice,
            speed=speed,
            use_gpu=use_gpu,
            pause_seconds=pause_seconds,
            normalize_times_enabled=normalize_times_enabled,
            normalize_numbers_enabled=normalize_numbers_enabled,
            style_preset=style_preset,
        )
