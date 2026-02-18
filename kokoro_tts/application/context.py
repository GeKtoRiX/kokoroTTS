"""Runtime dependency container for the desktop application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .ports import KokoroTtsPort

if TYPE_CHECKING:
    from ..config import AppConfig
    from .bootstrap import AppServices


@dataclass
class AppContext:
    """Holds runtime dependencies assembled at startup."""

    config: "AppConfig"
    logger: Any
    cuda_available: bool
    skip_app_init: bool
    tts_only_mode: bool
    pronunciation_rules_path: str
    default_output_format: str
    model_manager: Any = None
    text_normalizer: Any = None
    audio_writer: Any = None
    history_repository: Any = None
    history_service: Any = None
    morphology_repository: Any = None
    pronunciation_repository: Any = None
    app_state: Any = None
    app: Any = None
    morph_default_expression_extractor: Any = None
    tts_port: KokoroTtsPort | None = None

    def bind_services(self, services: "AppServices") -> None:
        self.model_manager = services.model_manager
        self.text_normalizer = services.text_normalizer
        self.audio_writer = services.audio_writer
        self.morphology_repository = services.morphology_repository
        self.pronunciation_repository = services.pronunciation_repository
        self.app_state = services.app_state
        self.history_repository = services.history_repository
        self.history_service = services.history_service
        self.app = services.app
        self.morph_default_expression_extractor = getattr(
            services.morphology_repository,
            "expression_extractor",
            None,
        )
