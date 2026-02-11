"""Gradio entrypoint and compatibility facade for Kokoro TTS."""
from __future__ import annotations

import os
import platform
import sys

import gradio as gr
import spaces
import torch

from kokoro_tts.config import AppConfig, load_config
from kokoro_tts.constants import OUTPUT_FORMATS, SAMPLE_RATE
from kokoro_tts.domain.normalization import (
    DECIMAL_RE,
    DIGIT_RE,
    INT_RE,
    MULTI_DOT_NUMBER_RE,
    ORDINAL_RE,
    ORDINAL_WORDS,
    PERCENT_RE,
    TIME_RE,
    ONES,
    TEENS,
    TENS,
    TextNormalizer,
    digits_to_words,
    normalize_numbers,
    normalize_times,
    number_to_words_0_59,
    number_to_words_0_99,
    number_to_words_0_999,
    number_to_words_0_9999,
    ordinal_to_words,
    ordinalize_words,
    time_to_words,
)
from kokoro_tts.domain.splitting import (
    ABBREV_DOTTED,
    ABBREV_TITLES,
    SENTENCE_BREAK_RE,
    SOFT_BREAK_RE,
    smart_split,
    split_parts,
    split_sentences,
)
from kokoro_tts.domain.text_utils import (
    MD_LINK_RE,
    SLASHED_RE,
    _apply_outside_spans,
    _find_skip_spans,
    _is_within_spans,
    _merge_spans,
)
from kokoro_tts.domain.voice import (
    CHOICES,
    VOICE_TAG_RE,
    limit_dialogue_parts,
    normalize_voice_input,
    normalize_voice_tag,
    parse_voice_segments,
    resolve_voice,
    summarize_voice,
)
from kokoro_tts.integrations.ffmpeg import configure_ffmpeg
from kokoro_tts.integrations.model_manager import ModelManager
from kokoro_tts.integrations.spaces_gpu import build_forward_gpu
from kokoro_tts.logging_config import setup_logging
from kokoro_tts.storage.audio_writer import AudioWriter
from kokoro_tts.storage.history_repository import HistoryRepository
from kokoro_tts.application.history_service import HistoryService
from kokoro_tts.application.state import KokoroState
from kokoro_tts.application.ui_hooks import UiHooks
from kokoro_tts.ui.gradio_app import (
    APP_THEME,
    DIALOGUE_NOTE,
    TOKEN_NOTE,
    UI_PRIMARY_HUE,
    create_gradio_app,
)

CONFIG = load_config()
logger = setup_logging(CONFIG)
SKIP_APP_INIT = os.getenv("KOKORO_SKIP_APP_INIT", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

BASE_DIR = os.path.dirname(__file__)
configure_ffmpeg(logger, BASE_DIR)

logger.info("Starting app")
logger.info("Log file: %s", CONFIG.log_file)
logger.debug(
    "Log config: LOG_LEVEL=%s FILE_LOG_LEVEL=%s LOG_DIR=%s OUTPUT_DIR=%s REPO_ID=%s "
    "MAX_CHUNK_CHARS=%s HISTORY_LIMIT=%s NORMALIZE_TIMES=%s NORMALIZE_NUMBERS=%s "
    "DEFAULT_OUTPUT_FORMAT=%s DEFAULT_CONCURRENCY_LIMIT=%s LOG_EVERY_N_SEGMENTS=%s",
    CONFIG.log_level,
    CONFIG.file_log_level,
    CONFIG.log_dir,
    CONFIG.output_dir,
    CONFIG.repo_id,
    CONFIG.max_chunk_chars,
    CONFIG.history_limit,
    CONFIG.normalize_times,
    CONFIG.normalize_numbers,
    CONFIG.default_output_format,
    CONFIG.default_concurrency_limit,
    CONFIG.log_segment_every,
)

CUDA_AVAILABLE = torch.cuda.is_available()
logger.info(
    "Config: SPACE_ID=%s IS_DUPLICATE=%s CHAR_LIMIT=%s CUDA_AVAILABLE=%s",
    CONFIG.space_id,
    CONFIG.is_duplicate,
    CONFIG.char_limit,
    CUDA_AVAILABLE,
)
if os.getenv("HF_TOKEN"):
    logger.debug("HF_TOKEN is set")
else:
    logger.debug("HF_TOKEN is not set; hub requests may be rate-limited")
logger.debug("Python version: %s", sys.version.replace("\n", " "))
logger.debug("Platform: %s", platform.platform())
logger.debug("Torch version: %s", torch.__version__)
if not CONFIG.is_duplicate:
    import kokoro
    import misaki

    logger.debug("Kokoro version: %s", kokoro.__version__)
    logger.debug("Misaki version: %s", misaki.__version__)

DEFAULT_OUTPUT_FORMAT = (
    CONFIG.default_output_format
    if CONFIG.default_output_format in OUTPUT_FORMATS
    else "wav"
)

logger.info("Voices available: %s (lazy loading)", len(CHOICES))

MODEL_MANAGER = None
TEXT_NORMALIZER = None
AUDIO_WRITER = None
HISTORY_REPOSITORY = None
HISTORY_SERVICE = None
APP_STATE = None
app = None
API_OPEN = None


@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed):
    return APP_STATE.model_manager.get_model(True)(ps, ref_s, speed)


def generate_first(
    text,
    voice="af_heart",
    mix_enabled=False,
    voice_mix=None,
    speed=1,
    use_gpu=CUDA_AVAILABLE,
    pause_seconds=0.0,
    output_format="wav",
    normalize_times_enabled=None,
    normalize_numbers_enabled=None,
):
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.generate_first(
        text=text,
        voice=voice,
        speed=speed,
        use_gpu=use_gpu,
        pause_seconds=pause_seconds,
        output_format=output_format,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
    )


# Arena API
def predict(
    text,
    voice="af_heart",
    mix_enabled=False,
    voice_mix=None,
    speed=1,
    normalize_times_enabled=None,
    normalize_numbers_enabled=None,
):
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.generate_first(
        text=text,
        voice=voice,
        speed=speed,
        use_gpu=False,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
        save_outputs=False,
    )[0]


def tokenize_first(
    text,
    voice="af_heart",
    mix_enabled=False,
    voice_mix=None,
    speed=1,
    normalize_times_enabled=None,
    normalize_numbers_enabled=None,
):
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    return APP_STATE.tokenize_first(
        text=text,
        voice=voice,
        speed=speed,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
    )


def generate_all(
    text,
    voice="af_heart",
    mix_enabled=False,
    voice_mix=None,
    speed=1,
    use_gpu=CUDA_AVAILABLE,
    pause_seconds=0.0,
    normalize_times_enabled=None,
    normalize_numbers_enabled=None,
):
    voice = resolve_voice(voice, voice_mix, mix_enabled)
    yield from APP_STATE.generate_all(
        text=text,
        voice=voice,
        speed=speed,
        use_gpu=use_gpu,
        pause_seconds=pause_seconds,
        normalize_times_enabled=normalize_times_enabled,
        normalize_numbers_enabled=normalize_numbers_enabled,
    )


if not SKIP_APP_INIT:
    MODEL_MANAGER = ModelManager(CONFIG.repo_id, CUDA_AVAILABLE, logger)
    TEXT_NORMALIZER = TextNormalizer(
        CONFIG.char_limit,
        CONFIG.normalize_times,
        CONFIG.normalize_numbers,
    )
    AUDIO_WRITER = AudioWriter(CONFIG.output_dir, SAMPLE_RATE, logger)
    forward_gpu = build_forward_gpu(MODEL_MANAGER)
    ui_hooks = UiHooks(
        warn=gr.Warning,
        info=gr.Info,
        error=gr.Error,
        error_type=gr.exceptions.Error,
    )
    APP_STATE = KokoroState(
        MODEL_MANAGER,
        TEXT_NORMALIZER,
        AUDIO_WRITER,
        CONFIG.max_chunk_chars,
        CUDA_AVAILABLE,
        CONFIG.log_segment_every,
        logger,
        gpu_forward=forward_gpu,
        ui_hooks=ui_hooks,
    )
    HISTORY_REPOSITORY = HistoryRepository(CONFIG.output_dir_abs, logger)
    HISTORY_SERVICE = HistoryService(
        CONFIG.history_limit,
        HISTORY_REPOSITORY,
        APP_STATE,
        logger,
    )
    app, API_OPEN = create_gradio_app(
        config=CONFIG,
        cuda_available=CUDA_AVAILABLE,
        logger=logger,
        generate_first=generate_first,
        tokenize_first=tokenize_first,
        generate_all=generate_all,
        predict=predict,
        history_service=HISTORY_SERVICE,
        choices=CHOICES,
    )
else:
    logger.info("KOKORO_SKIP_APP_INIT enabled; skipping model and UI initialization")


def launch() -> None:
    if SKIP_APP_INIT:
        logger.info("KOKORO_SKIP_APP_INIT enabled; launch skipped")
        return
    logger.info("Launching Gradio app")
    queue_kwargs = {"api_open": API_OPEN}
    if CONFIG.default_concurrency_limit is not None:
        queue_kwargs["default_concurrency_limit"] = CONFIG.default_concurrency_limit
    app.queue(**queue_kwargs).launch(show_api=API_OPEN, ssr_mode=True)


if __name__ == "__main__":
    launch()
