"""Application bootstrap assembly for model, storage, and UI services."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, cast

import gradio as gr

from ..config import AppConfig
from ..constants import SAMPLE_RATE
from ..domain.expressions import extract_english_expressions
from ..domain.normalization import TextNormalizer
from ..integrations.lm_studio import PosVerifyRequest, verify_pos_with_context
from ..integrations.model_manager import ModelManager
from ..integrations.spaces_gpu import build_forward_gpu
from ..storage.audio_writer import AudioWriter
from ..storage.history_repository import HistoryRepository
from ..storage.morphology_repository import MorphologyRepository
from ..storage.pronunciation_repository import PronunciationRepository
from ..ui.gradio_app import create_gradio_app
from .history_service import HistoryService
from .state import KokoroState
from .ui_hooks import UiHooks

LmVerifier = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class AppServices:
    model_manager: ModelManager
    text_normalizer: TextNormalizer
    audio_writer: AudioWriter
    morphology_repository: MorphologyRepository
    pronunciation_repository: PronunciationRepository
    app_state: KokoroState
    history_repository: HistoryRepository
    history_service: HistoryService
    app: gr.Blocks
    api_open: bool


def build_lm_verifier(config: AppConfig, logger) -> LmVerifier | None:
    """Create an LM verifier callable when verify settings are configured."""
    if config.lm_verify_enabled and config.lm_verify_model:

        def _lm_verifier(payload: dict[str, object]) -> dict[str, object]:
            raw_tokens = payload.get("tokens", [])
            raw_locked = payload.get("locked_expressions", [])
            return verify_pos_with_context(
                PosVerifyRequest(
                    segment_text=str(payload.get("segment_text", "")),
                    tokens=raw_tokens if isinstance(raw_tokens, list) else [],
                    locked_expressions=raw_locked if isinstance(raw_locked, list) else [],
                    base_url=config.lm_verify_base_url,
                    api_key=config.lm_verify_api_key,
                    model=config.lm_verify_model,
                    timeout_seconds=config.lm_verify_timeout_seconds,
                    temperature=config.lm_verify_temperature,
                    max_tokens=config.lm_verify_max_tokens,
                )
            )

        return _lm_verifier

    if config.lm_verify_enabled:
        logger.warning(
            "LM verify is enabled but LM_VERIFY_MODEL is empty; background verification is disabled."
        )
    return None


def initialize_app_services(
    *,
    config: AppConfig,
    cuda_available: bool,
    logger,
    pronunciation_rules_path: str,
    generate_first,
    tokenize_first,
    generate_all,
    predict,
    export_morphology_sheet,
    morphology_db_view,
    morphology_db_add,
    morphology_db_update,
    morphology_db_delete,
    load_pronunciation_rules,
    apply_pronunciation_rules,
    import_pronunciation_rules,
    export_pronunciation_rules,
    build_lesson_for_tts,
    set_tts_only_mode=None,
    set_llm_only_mode=None,
    tts_only_mode_default: bool = False,
    llm_only_mode_default: bool = False,
    choices: Mapping[str, str],
) -> AppServices:
    """Construct all runtime services and return a typed service bundle."""
    pronunciation_repository = PronunciationRepository(
        pronunciation_rules_path,
        logger_instance=logger,
    )
    pronunciation_rules = pronunciation_repository.load_rules()
    logger.info(
        "Pronunciation dictionary path: %s (languages=%s entries=%s)",
        pronunciation_rules_path,
        len(pronunciation_rules),
        sum(len(entries) for entries in pronunciation_rules.values()),
    )

    model_manager = ModelManager(
        config.repo_id,
        cuda_available,
        logger,
        pronunciation_rules=pronunciation_rules,
    )
    text_normalizer = TextNormalizer(
        config.char_limit,
        config.normalize_times,
        config.normalize_numbers,
    )
    audio_writer = AudioWriter(config.output_dir, SAMPLE_RATE, logger)

    lm_verifier = build_lm_verifier(config, logger)

    if config.morph_local_expressions_enabled:
        logger.info("Local expression extraction is enabled.")
    else:
        logger.info(
            "Local expression extraction is disabled; phrasal verbs and idioms are expected from LM verify."
        )

    morphology_repository = MorphologyRepository(
        enabled=config.morph_db_enabled,
        db_path=config.morph_db_path,
        table_prefix=config.morph_db_table_prefix,
        logger_instance=logger,
        expression_extractor=(
            extract_english_expressions
            if config.morph_local_expressions_enabled
            else (lambda _text: [])
        ),
        lm_verifier=lm_verifier,
        lm_verify_enabled=config.lm_verify_enabled,
        lm_verify_model=config.lm_verify_model,
        lm_verify_retries=config.lm_verify_max_retries,
        lm_verify_workers=config.lm_verify_workers,
    )

    forward_gpu = build_forward_gpu(model_manager)
    ui_hooks = UiHooks(
        warn=gr.Warning,
        info=gr.Info,
        error=cast(Callable[[Exception], Exception], gr.Error),
        error_type=gr.exceptions.Error,
    )
    app_state = KokoroState(
        model_manager,
        text_normalizer,
        audio_writer,
        config.max_chunk_chars,
        cuda_available,
        config.log_segment_every,
        logger,
        gpu_forward=forward_gpu,
        ui_hooks=ui_hooks,
        morphology_repository=morphology_repository,
    )
    history_repository = HistoryRepository(config.output_dir_abs, logger)
    history_service = HistoryService(
        config.history_limit,
        history_repository,
        app_state,
        logger,
    )

    app, api_open = create_gradio_app(
        config=config,
        cuda_available=cuda_available,
        logger=logger,
        generate_first=generate_first,
        tokenize_first=tokenize_first,
        generate_all=generate_all,
        predict=predict,
        export_morphology_sheet=export_morphology_sheet,
        morphology_db_view=morphology_db_view,
        morphology_db_add=morphology_db_add,
        morphology_db_update=morphology_db_update,
        morphology_db_delete=morphology_db_delete,
        load_pronunciation_rules=load_pronunciation_rules,
        apply_pronunciation_rules=apply_pronunciation_rules,
        import_pronunciation_rules=import_pronunciation_rules,
        export_pronunciation_rules=export_pronunciation_rules,
        build_lesson_for_tts=build_lesson_for_tts,
        set_tts_only_mode=set_tts_only_mode,
        set_llm_only_mode=set_llm_only_mode,
        tts_only_mode_default=tts_only_mode_default,
        llm_only_mode_default=llm_only_mode_default,
        history_service=history_service,
        choices=choices,
    )

    return AppServices(
        model_manager=model_manager,
        text_normalizer=text_normalizer,
        audio_writer=audio_writer,
        morphology_repository=morphology_repository,
        pronunciation_repository=pronunciation_repository,
        app_state=app_state,
        history_repository=history_repository,
        history_service=history_service,
        app=app,
        api_open=api_open,
    )
