"""Application state and orchestration for audio generation."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import logging
import threading
from typing import Callable, Generator

import torch

from ..constants import SAMPLE_RATE
from ..domain.audio_postfx import (
    AudioPostFxSettings,
    apply_fade,
    crossfade_join,
    trim_silence,
)
from ..domain.normalization import TextNormalizer
from ..domain.splitting import smart_split, split_parts
from ..domain.style import PIPELINE_STYLE_PARAM_NAMES, resolve_style_runtime
from ..domain.voice import (
    DialogueSegment,
    limit_dialogue_segment_parts,
    parse_dialogue_segments,
    summarize_dialogue_voice,
)
from ..integrations.model_manager import ModelManager
from ..storage.audio_writer import AudioWriter
from .ui_hooks import UiHooks


class KokoroState:
    def __init__(
        self,
        model_manager: ModelManager,
        normalizer: TextNormalizer,
        audio_writer: AudioWriter,
        max_chunk_chars: int,
        cuda_available: bool,
        log_segment_every: int,
        logger,
        *,
        gpu_forward: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] | None = None,
        ui_hooks: UiHooks | None = None,
        morphology_repository=None,
        morphology_async_ingest: bool = False,
        morphology_async_max_pending: int = 8,
        postfx_settings: AudioPostFxSettings | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.normalizer = normalizer
        self.audio_writer = audio_writer
        self.max_chunk_chars = max_chunk_chars
        self.cuda_available = cuda_available
        self.log_segment_every = log_segment_every
        self.logger = logger
        self.gpu_forward = gpu_forward
        self.ui_hooks = ui_hooks
        self.morphology_repository = morphology_repository
        self.morphology_async_ingest = bool(morphology_async_ingest)
        self.morphology_async_max_pending = max(1, int(morphology_async_max_pending))
        self.last_saved_paths: list[str] = []
        self.aux_features_enabled = True
        self._pipeline_style_param_cache: dict[type, str | None] = {}
        self._pause_tensor_cache: dict[int, torch.Tensor] = {}
        self._morph_executor: ThreadPoolExecutor | None = None
        self._pending_morph_futures: list[Future[None]] = []
        self._pending_morph_lock = threading.Lock()
        self.postfx_settings = postfx_settings or AudioPostFxSettings()
        if self.morphology_repository is not None and self.morphology_async_ingest:
            self._morph_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="morphology-ingest",
            )

    def set_aux_features_enabled(self, enabled: bool) -> None:
        self.aux_features_enabled = bool(enabled)

    def _on_morph_ingest_done(self, future: Future[None]) -> None:
        try:
            future.result()
        except Exception:
            self.logger.exception("Morphology DB async write failed")

    def _trim_pending_morph_futures(self) -> None:
        self._pending_morph_futures = [
            future for future in self._pending_morph_futures if not future.done()
        ]

    def wait_for_pending_morphology(self, timeout: float | None = None) -> None:
        futures: list[Future[None]]
        with self._pending_morph_lock:
            self._trim_pending_morph_futures()
            futures = list(self._pending_morph_futures)
        if not futures:
            return
        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception:
                self.logger.exception("Morphology DB async wait failed")

    def _persist_morphology(
        self,
        parts: list[list[DialogueSegment]],
        *,
        source: str,
    ) -> None:
        if not self.aux_features_enabled:
            return
        if self.morphology_repository is None:
            return
        morph_parts = [
            [(segment.voice, segment.text) for segment in segments]
            for segments in parts
            if segments
        ]
        if not morph_parts:
            return

        if self._morph_executor is None:
            try:
                self.morphology_repository.ingest_dialogue_parts(morph_parts, source=source)
            except Exception:
                self.logger.exception("Morphology DB write failed")
            return

        should_write_sync = False
        with self._pending_morph_lock:
            self._trim_pending_morph_futures()
            if len(self._pending_morph_futures) >= self.morphology_async_max_pending:
                should_write_sync = True
            else:
                payload = tuple(tuple(segment for segment in segments) for segments in morph_parts)
                future = self._morph_executor.submit(
                    self.morphology_repository.ingest_dialogue_parts,
                    payload,
                    source=source,
                )
                future.add_done_callback(self._on_morph_ingest_done)
                self._pending_morph_futures.append(future)
        if should_write_sync:
            self.logger.warning(
                "Morphology ingest queue is full (%s pending); writing synchronously",
                self.morphology_async_max_pending,
            )
            try:
                self.morphology_repository.ingest_dialogue_parts(morph_parts, source=source)
            except Exception:
                self.logger.exception("Morphology DB write failed")

    def tokenize_first(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1,
        normalize_times_enabled: bool | None = None,
        normalize_numbers_enabled: bool | None = None,
        style_preset: str = "neutral",
    ) -> str:
        parts = self._prepare_dialogue_parts(
            text,
            voice,
            normalize_times_enabled,
            normalize_numbers_enabled,
            style_preset,
        )
        if not parts:
            return ""
        first_segment = parts[0][0]
        segment_style, segment_speed, _ = resolve_style_runtime(
            first_segment.style_preset,
            speed,
            0.0,
        )
        pipeline = self.model_manager.get_pipeline(first_segment.voice)
        for _, ps, _ in self._run_pipeline(
            first_segment.text,
            first_segment.voice,
            segment_speed,
            segment_style,
            pipeline,
        ):
            return ps
        return ""

    def _preprocess_text(
        self,
        text: str,
        normalize_times_enabled: bool | None = None,
        normalize_numbers_enabled: bool | None = None,
        apply_char_limit: bool = True,
    ) -> str:
        return self.normalizer.preprocess(
            text,
            normalize_times_enabled=normalize_times_enabled,
            normalize_numbers_enabled=normalize_numbers_enabled,
            apply_char_limit=apply_char_limit,
        )

    def _prepare_dialogue_parts(
        self,
        text: str,
        default_voice: str,
        normalize_times_enabled: bool | None,
        normalize_numbers_enabled: bool | None,
        default_style_preset: str,
    ) -> list[list[DialogueSegment]]:
        parts = split_parts(text)
        dialogue_parts: list[list[DialogueSegment]] = []
        for part in parts:
            segments = parse_dialogue_segments(
                part,
                default_voice,
                default_style_preset=default_style_preset,
            )
            if segments:
                dialogue_parts.append(segments)
        if not dialogue_parts:
            return []
        limited_parts, truncated = limit_dialogue_segment_parts(
            dialogue_parts,
            self.normalizer.char_limit,
        )
        if truncated:
            self.logger.info(
                "Input truncated to %s characters (excluding tags)",
                self.normalizer.char_limit,
            )
        normalized_parts: list[list[DialogueSegment]] = []
        for segments in limited_parts:
            normalized_segments: list[DialogueSegment] = []
            for segment in segments:
                segment_text = segment.text.strip()
                if not segment_text:
                    continue
                normalized_text = self._preprocess_text(
                    segment_text,
                    normalize_times_enabled=normalize_times_enabled,
                    normalize_numbers_enabled=normalize_numbers_enabled,
                    apply_char_limit=False,
                )
                if normalized_text.strip():
                    normalized_segments.append(
                        DialogueSegment(
                            voice=segment.voice,
                            text=normalized_text,
                            style_preset=segment.style_preset,
                            pause_seconds=segment.pause_seconds,
                        )
                    )
            if normalized_segments:
                normalized_parts.append(normalized_segments)
        return normalized_parts

    def _pipeline_style_param_name(self, pipeline) -> str | None:
        pipeline_type = type(pipeline)
        if pipeline_type in self._pipeline_style_param_cache:
            return self._pipeline_style_param_cache[pipeline_type]
        param_name = None
        try:
            parameters = inspect.signature(pipeline.__call__).parameters
        except (TypeError, ValueError):
            parameters = {}
        for candidate in PIPELINE_STYLE_PARAM_NAMES:
            if candidate in parameters:
                param_name = candidate
                break
        self._pipeline_style_param_cache[pipeline_type] = param_name
        return param_name

    def _run_pipeline(
        self,
        text: str,
        voice: str,
        speed: float,
        style_preset: str,
        pipeline,
    ):
        style_param_name = self._pipeline_style_param_name(pipeline)
        if style_param_name is None:
            return pipeline(text, voice, speed)
        kwargs = {style_param_name: style_preset}
        try:
            return pipeline(text, voice, speed, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" in message:
                self._pipeline_style_param_cache[type(pipeline)] = None
                self.logger.debug(
                    "Pipeline style argument %s is unsupported at runtime; retrying without style",
                    style_param_name,
                )
                return pipeline(text, voice, speed)
            raise

    def _forward_gpu(self, ps: torch.Tensor, ref_s: torch.Tensor, speed: float) -> torch.Tensor:
        if self.gpu_forward is None:
            return self.model_manager.get_model(True)(ps, ref_s, speed)
        return self.gpu_forward(ps, ref_s, speed)

    def _is_ui_error(self, error: Exception) -> bool:
        return bool(self.ui_hooks and isinstance(error, self.ui_hooks.error_type))

    def _raise_ui_error(self, error: Exception) -> Exception:
        if self.ui_hooks:
            return self.ui_hooks.error(error)
        return error

    def _pause_tensor(self, sample_count: int) -> torch.Tensor | None:
        if sample_count <= 0:
            return None
        cached = self._pause_tensor_cache.get(sample_count)
        if cached is None:
            cached = torch.zeros(sample_count)
            self._pause_tensor_cache[sample_count] = cached
        return cached

    def _pause_audio_numpy(self, sample_count: int):
        pause_tensor = self._pause_tensor(sample_count)
        if pause_tensor is None:
            return None
        return pause_tensor.numpy()

    def _postfx_enabled(self) -> bool:
        return bool(getattr(self.postfx_settings, "enabled", False))

    def _trim_segment_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        if not self._postfx_enabled():
            return audio_tensor
        if not bool(getattr(self.postfx_settings, "trim_enabled", True)):
            return audio_tensor
        return trim_silence(
            audio_tensor,
            SAMPLE_RATE,
            float(getattr(self.postfx_settings, "trim_threshold_db", -42.0)),
            int(getattr(self.postfx_settings, "trim_keep_ms", 25)),
        )

    def _apply_part_fade(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        if not self._postfx_enabled():
            return audio_tensor
        return apply_fade(
            audio_tensor,
            SAMPLE_RATE,
            int(getattr(self.postfx_settings, "fade_in_ms", 12)),
            int(getattr(self.postfx_settings, "fade_out_ms", 40)),
        )

    def _crossfade_samples(self) -> int:
        if not self._postfx_enabled():
            return 0
        crossfade_ms = max(0, int(getattr(self.postfx_settings, "crossfade_ms", 25)))
        return max(0, int(round(crossfade_ms * SAMPLE_RATE / 1000.0)))

    def _generate_audio_for_text(
        self,
        text: str,
        voice: str,
        speed: float,
        style_preset: str,
        use_gpu: bool,
        pause_seconds: float,
    ) -> tuple[torch.Tensor | None, str]:
        keep_sentences = pause_seconds > 0
        sentences = smart_split(text, self.max_chunk_chars, keep_sentences=keep_sentences)
        pipeline = self.model_manager.get_pipeline(voice)
        pack = self.model_manager.get_voice_pack(voice)
        use_gpu = use_gpu and self.cuda_available
        pause_samples = max(0, int(pause_seconds * SAMPLE_RATE))
        pause_tensor = self._pause_tensor(pause_samples)
        segments: list[torch.Tensor] = []
        first_ps = ""
        model_cpu = None
        if not use_gpu:
            model_cpu = self.model_manager.get_model(False)
        with torch.inference_mode():
            for index, sentence in enumerate(sentences):
                for _, ps, _ in self._run_pipeline(sentence, voice, speed, style_preset, pipeline):
                    if not first_ps:
                        first_ps = ps
                    ref_s = pack[len(ps) - 1]
                    try:
                        if use_gpu:
                            audio = self._forward_gpu(ps, ref_s, speed)
                        else:
                            audio = model_cpu(ps, ref_s, speed)
                    except Exception as exc:
                        if self._is_ui_error(exc):
                            if use_gpu and self.ui_hooks:
                                self.ui_hooks.warn(str(exc))
                                self.ui_hooks.info(
                                    "Retrying with CPU. To avoid this error, change Hardware to CPU."
                                )
                                audio = self.model_manager.get_model(False)(ps, ref_s, speed)
                            else:
                                raise self._raise_ui_error(exc)
                        else:
                            raise
                    audio = audio.detach().cpu().flatten()
                    segments.append(audio)
                if pause_tensor is not None and index < len(sentences) - 1:
                    segments.append(pause_tensor)
        if not segments:
            return None, first_ps
        return torch.cat(segments), first_ps

    def prewarm_inference(
        self,
        *,
        voice: str = "af_heart",
        use_gpu: bool | None = None,
        style_preset: str = "neutral",
    ) -> None:
        if use_gpu is None:
            use_gpu = self.cuda_available
        use_gpu = bool(use_gpu and self.cuda_available)
        pipeline = self.model_manager.get_pipeline(voice)
        pack = self.model_manager.get_voice_pack(voice)
        model_cpu = None if use_gpu else self.model_manager.get_model(False)
        warm_text = "Warm up."
        with torch.inference_mode():
            for _, ps, _ in self._run_pipeline(
                warm_text,
                voice,
                1.0,
                style_preset,
                pipeline,
            ):
                ref_s = pack[len(ps) - 1]
                if use_gpu:
                    audio = self._forward_gpu(ps, ref_s, 1.0)
                else:
                    audio = model_cpu(ps, ref_s, 1.0)
                _ = audio.detach().cpu()
                break

    def generate_first(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1,
        use_gpu: bool | None = None,
        pause_seconds: float = 0.0,
        output_format: str = "wav",
        normalize_times_enabled: bool | None = None,
        normalize_numbers_enabled: bool | None = None,
        save_outputs: bool = True,
        style_preset: str = "neutral",
    ) -> tuple[tuple[int, object] | None, str]:
        if use_gpu is None:
            use_gpu = self.cuda_available
        base_speed = speed
        base_pause_seconds = pause_seconds
        style_preset, speed, pause_seconds = resolve_style_runtime(style_preset, speed, pause_seconds)
        parts = self._prepare_dialogue_parts(
            text,
            voice,
            normalize_times_enabled,
            normalize_numbers_enabled,
            style_preset,
        )
        self.logger.debug(
            "Generate start: text_len=%s parts=%s voice=%s style=%s speed=%s use_gpu=%s pause_seconds=%s output_format=%s postfx=%s trim=%s fade_in_ms=%s fade_out_ms=%s crossfade_ms=%s loudness=%s",
            len(text),
            len(parts),
            voice,
            style_preset,
            speed,
            use_gpu,
            pause_seconds,
            output_format,
            self._postfx_enabled(),
            bool(getattr(self.postfx_settings, "trim_enabled", True)),
            int(getattr(self.postfx_settings, "fade_in_ms", 12)),
            int(getattr(self.postfx_settings, "fade_out_ms", 40)),
            int(getattr(self.postfx_settings, "crossfade_ms", 25)),
            bool(getattr(self.postfx_settings, "loudness_enabled", True)),
        )
        self._persist_morphology(parts, source="generate_first")
        default_pause_samples = max(0, int(pause_seconds * SAMPLE_RATE))
        combined_segments: list[torch.Tensor] = []
        first_ps = ""
        output_format = self.audio_writer.resolve_output_format(output_format)
        output_voice = summarize_dialogue_voice(parts, voice)
        output_paths = (
            self.audio_writer.build_output_paths(output_voice, len(parts), output_format)
            if save_outputs
            else []
        )
        saved_paths: list[str] = []
        for index, segments in enumerate(parts):
            part_audio: torch.Tensor | None = None
            last_segment_pause_seconds = pause_seconds
            pending_pause_samples: int | None = None
            crossfade_samples = self._crossfade_samples()
            for segment_index, segment in enumerate(segments):
                segment_style, segment_speed, segment_pause_seconds = resolve_style_runtime(
                    segment.style_preset,
                    base_speed,
                    base_pause_seconds,
                )
                if segment.pause_seconds is not None:
                    segment_pause_seconds = segment.pause_seconds
                audio_tensor, part_ps = self._generate_audio_for_text(
                    segment.text,
                    segment.voice,
                    segment_speed,
                    segment_style,
                    use_gpu,
                    segment_pause_seconds,
                )
                if audio_tensor is None:
                    continue
                audio_tensor = self._trim_segment_audio(audio_tensor)
                if not first_ps and part_ps:
                    first_ps = part_ps
                if part_audio is None:
                    part_audio = audio_tensor
                else:
                    if pending_pause_samples == 0 and crossfade_samples > 0:
                        part_audio = crossfade_join(part_audio, audio_tensor, crossfade_samples)
                    else:
                        if pending_pause_samples:
                            pause_tensor = self._pause_tensor(pending_pause_samples)
                            if pause_tensor is not None:
                                part_audio = torch.cat([part_audio, pause_tensor])
                        part_audio = torch.cat([part_audio, audio_tensor])
                last_segment_pause_seconds = segment_pause_seconds
                if segment_index < len(segments) - 1:
                    pending_pause_samples = max(0, int(segment_pause_seconds * SAMPLE_RATE))
                else:
                    pending_pause_samples = None
            if part_audio is None:
                continue
            part_audio = self._apply_part_fade(part_audio)
            if save_outputs:
                output_path = output_paths[index]
                try:
                    saved_path = self.audio_writer.save_audio(
                        output_path,
                        part_audio,
                        output_format,
                        postfx_settings=self.postfx_settings,
                    )
                    saved_paths.append(saved_path)
                except Exception:
                    self.logger.exception("Failed to save output: %s", output_path)
            combined_segments.append(part_audio)
            if index < len(parts) - 1:
                part_pause_samples = max(0, int(last_segment_pause_seconds * SAMPLE_RATE))
                pause_tensor = self._pause_tensor(part_pause_samples)
                if pause_tensor is not None:
                    combined_segments.append(pause_tensor)
        if not combined_segments:
            self.logger.debug("Generate produced no segments")
            self.last_saved_paths = []
            return None, ""
        full_audio = torch.cat(combined_segments)
        if save_outputs:
            self.logger.info(
                "Saved %s file(s) to %s",
                len(saved_paths),
                self.audio_writer.output_dir,
            )
            self.logger.debug("Saved files: %s", saved_paths)
        self.last_saved_paths = list(saved_paths)
        self.logger.debug(
            "Generate complete: segments=%s pause_samples=%s audio_samples=%s",
            len(combined_segments),
            default_pause_samples,
            full_audio.numel(),
        )
        return (SAMPLE_RATE, full_audio.numpy()), first_ps

    def generate_all(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1,
        use_gpu: bool | None = None,
        pause_seconds: float = 0.0,
        normalize_times_enabled: bool | None = None,
        normalize_numbers_enabled: bool | None = None,
        style_preset: str = "neutral",
    ) -> Generator[tuple[int, object], None, None]:
        if use_gpu is None:
            use_gpu = self.cuda_available
        base_speed = speed
        base_pause_seconds = pause_seconds
        style_preset, speed, pause_seconds = resolve_style_runtime(style_preset, speed, pause_seconds)
        parts = self._prepare_dialogue_parts(
            text,
            voice,
            normalize_times_enabled,
            normalize_numbers_enabled,
            style_preset,
        )
        self.logger.debug(
            "Stream start: text_len=%s parts=%s voice=%s style=%s speed=%s use_gpu=%s pause_seconds=%s",
            len(text),
            len(parts),
            voice,
            style_preset,
            speed,
            use_gpu,
            pause_seconds,
        )
        self._persist_morphology(parts, source="generate_all")
        use_gpu = use_gpu and self.cuda_available
        first = True
        default_pause_samples = max(0, int(pause_seconds * SAMPLE_RATE))
        segment_index = 0
        model_cpu = None
        if not use_gpu:
            model_cpu = self.model_manager.get_model(False)
        with torch.inference_mode():
            for part_index, segments in enumerate(parts):
                for segment_idx, segment in enumerate(segments):
                    segment_style, segment_speed, segment_pause_seconds = resolve_style_runtime(
                        segment.style_preset,
                        base_speed,
                        base_pause_seconds,
                    )
                    if segment.pause_seconds is not None:
                        segment_pause_seconds = segment.pause_seconds
                    keep_sentences = segment_pause_seconds > 0
                    segment_pause_samples = max(0, int(segment_pause_seconds * SAMPLE_RATE))
                    pause_audio = self._pause_audio_numpy(segment_pause_samples)
                    segment_voice = segment.voice
                    pipeline = self.model_manager.get_pipeline(segment_voice)
                    pack = self.model_manager.get_voice_pack(segment_voice)
                    sentences = smart_split(
                        segment.text,
                        self.max_chunk_chars,
                        keep_sentences=keep_sentences,
                    )
                    for sentence_index, sentence in enumerate(sentences):
                        iterator = iter(
                            self._run_pipeline(
                                sentence,
                                segment_voice,
                                segment_speed,
                                segment_style,
                                pipeline,
                            )
                        )
                        current = next(iterator, None)
                        while current is not None:
                            _, ps, _ = current
                            segment_index += 1
                            if self.logger.isEnabledFor(logging.DEBUG) and (
                                segment_index == 1 or segment_index % self.log_segment_every == 0
                            ):
                                self.logger.debug(
                                    "Stream segment=%s tokens=%s",
                                    segment_index,
                                    len(ps),
                                )
                            ref_s = pack[len(ps) - 1]
                            try:
                                if use_gpu:
                                    audio = self._forward_gpu(ps, ref_s, segment_speed)
                                else:
                                    audio = model_cpu(ps, ref_s, segment_speed)
                            except Exception as exc:
                                if self._is_ui_error(exc):
                                    if use_gpu and self.ui_hooks:
                                        self.ui_hooks.warn(str(exc))
                                        self.ui_hooks.info("Switching to CPU")
                                        audio = self.model_manager.get_model(False)(
                                            ps,
                                            ref_s,
                                            segment_speed,
                                        )
                                    else:
                                        raise self._raise_ui_error(exc)
                                else:
                                    raise
                            audio = audio.detach().cpu()
                            yield SAMPLE_RATE, audio.numpy()
                            if first:
                                first = False
                                yield SAMPLE_RATE, torch.zeros(1).numpy()
                            current = next(iterator, None)
                        if pause_audio is not None:
                            has_next_sentence = sentence_index < len(sentences) - 1
                            has_next_segment = segment_idx < len(segments) - 1
                            has_next_part = part_index < len(parts) - 1
                            if has_next_sentence or has_next_segment or has_next_part:
                                yield SAMPLE_RATE, pause_audio
        self.logger.debug(
            "Stream complete: segments=%s pause_samples=%s",
            segment_index,
            default_pause_samples,
        )
