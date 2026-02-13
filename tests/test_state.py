import numpy as np
import torch

from kokoro_tts.application.state import KokoroState
from kokoro_tts.application.ui_hooks import UiHooks
from kokoro_tts.domain.normalization import TextNormalizer


class _Logger:
    def __init__(self):
        self.debugs = []
        self.infos = []
        self.exceptions = []

    def debug(self, message, *args):
        self.debugs.append(message % args if args else message)

    def info(self, message, *args):
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

    def resolve_output_format(self, output_format):
        return output_format or "wav"

    def build_output_paths(self, voice, parts_count, output_format):
        return [f"{voice}_part{index}.{output_format}" for index in range(parts_count)]

    def save_audio(self, path, audio, output_format):
        self.saved.append((path, output_format, int(audio.numel())))
        return path


class _MorphRepo:
    def __init__(self, should_raise=False):
        self.rows = []
        self.should_raise = should_raise

    def ingest_dialogue_parts(self, parts, *, source):
        if self.should_raise:
            raise RuntimeError("db write failed")
        self.rows.append((parts, source))


def _build_state(*, hooks=None, morph_repo=None, gpu_forward=None, pipeline=None):
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
    )
    return state, model_manager, audio_writer, logger


def test_tokenize_first_and_prepare_dialogue_parts():
    state, _, _, _ = _build_state()
    parts = state._prepare_dialogue_parts(
        "[voice=af_heart]hello world|[voice=am_michael]second part",
        "af_heart",
        normalize_times_enabled=False,
        normalize_numbers_enabled=False,
    )
    assert parts == [
        [("af_heart", "hello world")],
        [("am_michael", "second part")],
    ]
    assert state.tokenize_first("hello world", voice="af_heart") == "hello"


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
