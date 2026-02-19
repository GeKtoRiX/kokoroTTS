"""Runtime integration for Russian Silero TTS voices."""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


_PLACEHOLDER_PREWARM_TEXT = "\u041f\u0440\u0438\u0432\u0435\u0442."


class SileroManager:
    """Lazy loader and voice registry for the Silero Russian backend."""

    def __init__(
        self,
        *,
        model_id: str = "v5_cis_base",
        cache_dir: str = "data/cache/torch",
        cpu_only: bool = True,
        logger=None,
        sample_rate: int = 24000,
    ) -> None:
        self.model_id = str(model_id or "v5_cis_base").strip() or "v5_cis_base"
        self.cache_dir = str(cache_dir or "data/cache/torch").strip() or "data/cache/torch"
        self.cpu_only = bool(cpu_only)
        self.logger = logger or logging.getLogger(__name__)
        self.sample_rate = max(8000, int(sample_rate))
        self._lock = threading.Lock()
        self._model = None
        self._voice_id_to_speaker: dict[str, str] = {}
        self._voice_items: list[tuple[str, str]] = []

    def _set_torch_cache(self) -> None:
        cache_path = Path(self.cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_path)

    @staticmethod
    def _slugify_speaker(speaker: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", speaker.lower()).strip("_")
        return slug or "voice"

    def _build_voice_items(self, speakers: list[str]) -> None:
        voice_items: list[tuple[str, str]] = []
        voice_map: dict[str, str] = {}
        used_ids: set[str] = set()
        for speaker in speakers:
            base_slug = self._slugify_speaker(speaker)
            voice_id = f"r_{base_slug}"
            index = 2
            while voice_id in used_ids:
                voice_id = f"r_{base_slug}_{index}"
                index += 1
            used_ids.add(voice_id)
            label = f"Russian {speaker}"
            voice_map[voice_id] = speaker
            voice_items.append((label, voice_id))
        self._voice_id_to_speaker = voice_map
        self._voice_items = voice_items

    @staticmethod
    def _extract_speakers(model: Any) -> list[str]:
        raw_speakers = getattr(model, "speakers", None)
        if raw_speakers is None:
            return []
        speakers = [str(item).strip() for item in list(raw_speakers) if str(item).strip()]
        return speakers

    def _load_model_locked(self) -> None:
        if self._model is not None:
            return
        self._set_torch_cache()
        try:
            from silero import silero_tts
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Silero package is not available.") from exc

        result = silero_tts(language="ru", speaker=self.model_id)
        if not isinstance(result, tuple) or not result:
            raise RuntimeError("Unexpected Silero model response.")
        model = result[0]
        if not hasattr(model, "apply_tts"):
            raise RuntimeError("Silero model does not expose apply_tts().")
        speakers = self._extract_speakers(model)
        if not speakers:
            raise RuntimeError("Silero model does not expose speaker list.")
        self._model = model
        self._build_voice_items(speakers)
        self.logger.info(
            "Silero Russian model loaded: model_id=%s voices=%s cpu_only=%s",
            self.model_id,
            len(self._voice_items),
            self.cpu_only,
        )

    def _ensure_model(self) -> Any:
        model = self._model
        if model is not None:
            return model
        with self._lock:
            self._load_model_locked()
            return self._model

    def discover_voice_items(self) -> list[tuple[str, str]]:
        self._ensure_model()
        return list(self._voice_items)

    def supports_voice(self, voice_id: str) -> bool:
        if not voice_id:
            return False
        if voice_id in self._voice_id_to_speaker:
            return True
        # If called before discovery, ensure map is initialized first.
        self._ensure_model()
        return voice_id in self._voice_id_to_speaker

    def default_voice_id(self) -> str:
        voices = self.discover_voice_items()
        if not voices:
            raise RuntimeError("No Russian voices are available in Silero.")
        return voices[0][1]

    def _speaker_for_voice(self, voice_id: str) -> str:
        speaker = self._voice_id_to_speaker.get(str(voice_id or "").strip())
        if speaker:
            return speaker
        self._ensure_model()
        speaker = self._voice_id_to_speaker.get(str(voice_id or "").strip())
        if not speaker:
            raise ValueError(f"Unknown Russian voice: {voice_id}")
        return speaker

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        model = self._ensure_model()
        cleaned_text = str(text or "").strip()
        if not cleaned_text:
            return torch.zeros(0, dtype=torch.float32)
        speaker = self._speaker_for_voice(voice_id)
        target_rate = int(sample_rate or self.sample_rate)
        if target_rate < 8000:
            target_rate = self.sample_rate
        audio = model.apply_tts(
            text=cleaned_text,
            speaker=speaker,
            sample_rate=target_rate,
        )
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().flatten().to(dtype=torch.float32)
        return torch.as_tensor(np.asarray(audio), dtype=torch.float32).flatten().cpu()

    def prewarm(self, *, voice_id: str | None = None, sample_rate: int | None = None) -> None:
        target_voice = voice_id or self.default_voice_id()
        _ = self.synthesize(
            _PLACEHOLDER_PREWARM_TEXT,
            voice_id=target_voice,
            sample_rate=sample_rate or self.sample_rate,
        )
