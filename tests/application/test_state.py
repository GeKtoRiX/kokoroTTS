from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
import torch

from kokoro_tts.application.state import KokoroState
from kokoro_tts.application.ui_hooks import UiHooks
from kokoro_tts.domain.audio_postfx import AudioPostFxSettings
from kokoro_tts.domain.normalization import TextNormalizer
from kokoro_tts.domain.voice import DialogueSegment


class _Logger:
    def __init__(self):
        self.debugs = []
        self.infos = []
        self.exceptions = []

    def debug(self, message, *args):
        self.debugs.append(message % args if args else message)

    def info(self, message, *args):
        self.infos.append(message % args if args else message)

    def warning(self, message, *args):
        self.infos.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)

    def isEnabledFor(self, _level):
        return True


class _Pack:
    def __getitem__(self, index):
        return torch.tensor([float(index)], dtype=torch.float32)


class _Pipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, text, voice, speed, style_preset=None):
        self.calls.append((text, voice, speed, style_preset))
        for word in text.split():
            yield None, word, None


class _Model:
    def __init__(self):
        self.calls = []
        self.raise_error = None

    def __call__(self, ps, ref_s, speed):
        if self.raise_error is not None:
            raise self.raise_error
        self.calls.append((ps, ref_s, speed))
        return torch.tensor([float(len(str(ps))), float(speed)], dtype=torch.float32)


class _ModelManager:
    def __init__(self, pipeline=None):
        self.cpu_model = _Model()
        self.gpu_model = _Model()
        self.pipeline = pipeline or _Pipeline()
        self.pack = _Pack()

    def get_model(self, use_gpu):
        return self.gpu_model if use_gpu else self.cpu_model

    def get_pipeline(self, voice):
        _ = voice
        return self.pipeline

    def get_voice_pack(self, voice):
        _ = voice
        return self.pack


class _AudioWriter:
    def __init__(self):
        self.output_dir = "outputs"
        self.saved = []
        self.saved_texts = []

    def resolve_output_format(self, output_format):
        return output_format or "wav"

    def build_output_paths(self, voice, parts_count, output_format):
        return [f"{voice}_part{index}.{output_format}" for index in range(parts_count)]

    def save_audio(self, path, audio, output_format, *, postfx_settings=None):
        self.saved.append((path, output_format, int(audio.numel()), postfx_settings))
        return path

    def save_text_sidecar(self, audio_path, text):
        txt_path = f"{audio_path.rsplit('.', 1)[0]}.txt"
        self.saved_texts.append((txt_path, text))
        return txt_path


class _MorphRepo:
    def __init__(self, should_raise=False):
        self.rows = []
        self.should_raise = should_raise

    def ingest_dialogue_parts(self, parts, *, source):
        if self.should_raise:
            raise RuntimeError("db write failed")
        self.rows.append((parts, source))


def _build_state(
    *,
    hooks=None,
    morph_repo=None,
    gpu_forward=None,
    pipeline=None,
    postfx_settings=None,
):
    model_manager = _ModelManager(pipeline=pipeline)
    normalizer = TextNormalizer(char_limit=80, normalize_times=False, normalize_numbers=False)
    audio_writer = _AudioWriter()
    logger = _Logger()
    state = KokoroState(
        model_manager,
        normalizer,
        audio_writer,
        max_chunk_chars=20,
        cuda_available=True,
        log_segment_every=1,
        logger=logger,
        gpu_forward=gpu_forward,
        ui_hooks=hooks,
        morphology_repository=morph_repo,
        postfx_settings=postfx_settings,
    )
    return state, model_manager, audio_writer, logger


def test_tokenize_first_and_prepare_dialogue_parts():
    state, _, _, _ = _build_state()
    parts = state._prepare_dialogue_parts(
        "[voice=af_heart]hello world|[voice=am_michael]second part",
        "af_heart",
        normalize_times_enabled=False,
        normalize_numbers_enabled=False,
        default_style_preset="neutral",
    )
    assert parts == [
        [DialogueSegment("af_heart", "hello world", "neutral", None)],
        [DialogueSegment("am_michael", "second part", "neutral", None)],
    ]
    assert state.tokenize_first("hello world", voice="af_heart") == "hello\nworld"


def test_generate_first_saves_outputs_and_persists_morphology():
    morph = _MorphRepo()
    state, _, writer, _ = _build_state(morph_repo=morph)

    (sample_rate, audio_np), first_ps = state.generate_first(
        "hello world|second part",
        voice="af_heart",
        speed=1.0,
        use_gpu=False,
        pause_seconds=0.1,
        output_format="wav",
    )

    assert sample_rate == 24000
    assert isinstance(audio_np, np.ndarray)
    assert first_ps
    assert writer.saved
    assert writer.saved_texts
    assert writer.saved_texts[0][0].endswith(".txt")
    assert "hello world" in writer.saved_texts[0][1]
    assert writer.saved[0][3] is not None
    assert morph.rows and morph.rows[0][1] == "generate_first"
    assert state.last_saved_paths


def test_generate_first_handles_empty_input():
    state, _, writer, _ = _build_state()
    result, first_ps = state.generate_first("", save_outputs=False)
    assert result is None
    assert first_ps == ""
    assert writer.saved == []


def test_generate_all_streams_audio_and_pause():
    morph = _MorphRepo()
    state, _, _, _ = _build_state(morph_repo=morph)
    chunks = list(
        state.generate_all(
            "hello world",
            voice="af_heart",
            speed=1.0,
            use_gpu=False,
            pause_seconds=0.1,
        )
    )
    assert len(chunks) >= 2
    assert all(rate == 24000 for rate, _ in chunks)
    assert morph.rows and morph.rows[0][1] == "generate_all"


def test_generate_audio_ui_error_fallbacks_to_cpu():
    class UiError(RuntimeError):
        pass

    warnings = []
    infos = []
    hooks = UiHooks(
        warn=lambda msg: warnings.append(msg),
        info=lambda msg: infos.append(msg),
        error=lambda exc: exc,
        error_type=UiError,
    )

    def failing_gpu(_ps, _ref_s, _speed):
        raise UiError("gpu failed")

    state, manager, _, _ = _build_state(hooks=hooks, gpu_forward=failing_gpu)
    audio, first_ps = state._generate_audio_for_text(
        "hello",
        "af_heart",
        speed=1.0,
        style_preset="neutral",
        use_gpu=True,
        pause_seconds=0.0,
    )

    assert audio is not None
    assert first_ps == "hello"
    assert warnings and infos
    assert manager.cpu_model.calls


def test_generate_audio_ui_error_without_gpu_raises_wrapped_error():
    class UiError(RuntimeError):
        pass

    hooks = UiHooks(
        warn=lambda _msg: None,
        info=lambda _msg: None,
        error=lambda exc: RuntimeError(f"wrapped:{exc}"),
        error_type=UiError,
    )
    state, manager, _, _ = _build_state(hooks=hooks)
    manager.cpu_model.raise_error = UiError("cpu failed")

    try:
        state._generate_audio_for_text(
            "hello",
            "af_heart",
            speed=1.0,
            style_preset="neutral",
            use_gpu=False,
            pause_seconds=0.0,
        )
    except RuntimeError as exc:
        assert "wrapped:cpu failed" in str(exc)
    else:
        raise AssertionError("Expected wrapped UI error")


def test_morphology_persist_errors_are_caught():
    state, _, _, logger = _build_state(morph_repo=_MorphRepo(should_raise=True))
    result, _ = state.generate_first("hello", save_outputs=False, use_gpu=False)
    assert result is not None
    assert logger.exceptions


def test_generate_first_applies_style_preset_to_speed_and_pipeline():
    state, manager, _, _ = _build_state()
    state.generate_first(
        "hello world",
        speed=1.0,
        style_preset="narrator",
        use_gpu=False,
        pause_seconds=1.0,
        save_outputs=False,
    )

    assert manager.pipeline.calls
    assert manager.pipeline.calls[0][3] == "narrator"
    assert manager.cpu_model.calls
    assert round(float(manager.cpu_model.calls[0][2]), 2) == 0.92


def test_generate_first_applies_inline_style_and_pause_overrides():
    state, manager, _, _ = _build_state()
    (sample_rate, audio_np), _ = state.generate_first(
        "[style=narrator][pause=0.001]hello [style=energetic][pause=0]world",
        speed=1.0,
        style_preset="neutral",
        pause_seconds=0.5,
        use_gpu=False,
        save_outputs=False,
    )

    assert sample_rate == 24000
    assert manager.pipeline.calls
    assert manager.pipeline.calls[0][3] == "narrator"
    assert manager.pipeline.calls[1][3] == "energetic"
    assert manager.cpu_model.calls
    assert round(float(manager.cpu_model.calls[0][2]), 2) == 0.92
    assert round(float(manager.cpu_model.calls[1][2]), 2) == 1.12
    assert len(audio_np) == 28


def test_generate_first_pause_does_not_split_long_single_sentence():
    state, _, _, _ = _build_state()
    result, _ = state.generate_first(
        "alpha beta gamma delta epsilon",
        pause_seconds=0.1,
        use_gpu=False,
        save_outputs=False,
    )

    assert result is not None
    _sample_rate, audio_np = result
    assert len(audio_np) == 10


def test_generate_first_pause_between_sentences_when_sentence_is_chunked():
    state, _, _, _ = _build_state()
    result, _ = state.generate_first(
        "alpha beta gamma delta epsilon. zeta eta.",
        pause_seconds=0.001,
        use_gpu=False,
        save_outputs=False,
    )

    assert result is not None
    _sample_rate, audio_np = result
    assert len(audio_np) == 38


def test_generate_all_pause_does_not_emit_pause_inside_long_single_sentence():
    state, _, _, _ = _build_state()
    chunks = list(
        state.generate_all(
            "alpha beta gamma delta epsilon",
            pause_seconds=0.001,
            use_gpu=False,
        )
    )

    pause_chunks = [audio for _rate, audio in chunks if len(audio) == 24]
    assert pause_chunks == []


def test_join_sentence_audios_applies_edge_fade_around_pause():
    state, _, _, _ = _build_state()
    pause_tensor = state._pause_tensor(24)
    assert pause_tensor is not None
    joined = state._join_sentence_audios(
        [
            torch.ones(10, dtype=torch.float32),
            torch.ones(10, dtype=torch.float32),
        ],
        pause_tensor,
    )

    assert joined is not None
    assert len(joined) == 44
    assert float(joined[9]) == 0.0
    assert float(joined[34]) == 0.0


def test_generate_first_falls_back_when_pipeline_has_no_style_argument():
    class _PipelineNoStyle:
        def __init__(self):
            self.calls = []

        def __call__(self, text, voice, speed):
            self.calls.append((text, voice, speed))
            for word in text.split():
                yield None, word, None

    pipeline = _PipelineNoStyle()
    state, manager, _, _ = _build_state(pipeline=pipeline)
    state.generate_first(
        "hello",
        speed=1.0,
        style_preset="energetic",
        use_gpu=False,
        save_outputs=False,
    )

    assert pipeline.calls
    assert len(pipeline.calls[0]) == 3
    assert manager.cpu_model.calls
    assert round(float(manager.cpu_model.calls[0][2]), 2) == 1.12


def test_set_aux_features_enabled_disables_morphology_persist():
    morph = _MorphRepo()
    state, _, _, _ = _build_state(morph_repo=morph)
    state.set_aux_features_enabled(False)
    result, _ = state.generate_first("hello world", save_outputs=False, use_gpu=False)
    assert result is not None
    assert morph.rows == []


def test_prewarm_inference_runs_pipeline_and_model():
    state, manager, _, _ = _build_state()
    state.prewarm_inference(voice="af_heart", use_gpu=False, style_preset="neutral")
    assert manager.pipeline.calls
    assert manager.cpu_model.calls


def test_async_morphology_ingest_enqueue_and_flush():
    class _SlowMorphRepo:
        def __init__(self):
            self.rows = []

        def ingest_dialogue_parts(self, parts, *, source):
            time.sleep(0.01)
            self.rows.append((parts, source))

    morph = _SlowMorphRepo()
    state, _, _, _ = _build_state(morph_repo=morph)
    state.morphology_async_ingest = True
    state._morph_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="morph-test",
    )
    state._persist_morphology(
        [[DialogueSegment("af_heart", "hello world", "neutral", None)]],
        source="generate_first",
    )
    state.wait_for_pending_morphology(timeout=2.0)
    state.shutdown(wait=True)
    assert morph.rows


def test_shutdown_clears_async_executor_and_pending_futures():
    class _SlowMorphRepo:
        def __init__(self):
            self.rows = []

        def ingest_dialogue_parts(self, parts, *, source):
            time.sleep(0.01)
            self.rows.append((parts, source))

    morph = _SlowMorphRepo()
    state, _, _, _ = _build_state(morph_repo=morph)
    state.morphology_async_ingest = True
    state._morph_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="morph-test",
    )
    state._persist_morphology(
        [[DialogueSegment("af_heart", "hello world", "neutral", None)]],
        source="generate_first",
    )
    state.shutdown(wait=True)
    assert state._morph_executor is None
    assert state._pending_morph_futures == []
    assert morph.rows


def test_generate_first_postfx_trim_and_fade_enabled_no_crash():
    settings = AudioPostFxSettings(
        enabled=True,
        trim_enabled=True,
        trim_threshold_db=-60.0,
        trim_keep_ms=0,
        fade_in_ms=10,
        fade_out_ms=10,
        crossfade_ms=0,
        loudness_enabled=True,
    )
    state, _, _, _ = _build_state(postfx_settings=settings)
    result, _ = state.generate_first(
        "hello world",
        save_outputs=False,
        use_gpu=False,
        pause_seconds=0.0,
    )
    assert result is not None


def test_generate_first_crossfade_reduces_length_when_pause_is_zero():
    baseline_state, _, _, _ = _build_state(
        postfx_settings=AudioPostFxSettings(
            enabled=False,
            trim_enabled=False,
            fade_in_ms=0,
            fade_out_ms=0,
            crossfade_ms=25,
        )
    )
    crossfade_state, _, _, _ = _build_state(
        postfx_settings=AudioPostFxSettings(
            enabled=True,
            trim_enabled=False,
            fade_in_ms=0,
            fade_out_ms=0,
            crossfade_ms=25,
        )
    )
    base_result, _ = baseline_state.generate_first(
        "hello [style=energetic]world",
        save_outputs=False,
        use_gpu=False,
        pause_seconds=0.0,
    )
    cross_result, _ = crossfade_state.generate_first(
        "hello [style=energetic]world",
        save_outputs=False,
        use_gpu=False,
        pause_seconds=0.0,
    )
    assert base_result is not None
    assert cross_result is not None
    assert len(cross_result[1]) < len(base_result[1])


def test_generate_first_crossfade_not_applied_when_pause_exists():
    settings_off = AudioPostFxSettings(
        enabled=False,
        trim_enabled=False,
        fade_in_ms=0,
        fade_out_ms=0,
        crossfade_ms=25,
    )
    settings_on = AudioPostFxSettings(
        enabled=True,
        trim_enabled=False,
        fade_in_ms=0,
        fade_out_ms=0,
        crossfade_ms=25,
    )
    state_off, _, _, _ = _build_state(postfx_settings=settings_off)
    state_on, _, _, _ = _build_state(postfx_settings=settings_on)
    off_result, _ = state_off.generate_first(
        "hello [style=energetic]world",
        save_outputs=False,
        use_gpu=False,
        pause_seconds=0.1,
    )
    on_result, _ = state_on.generate_first(
        "hello [style=energetic]world",
        save_outputs=False,
        use_gpu=False,
        pause_seconds=0.1,
    )
    assert off_result is not None
    assert on_result is not None
    assert len(on_result[1]) == len(off_result[1])
