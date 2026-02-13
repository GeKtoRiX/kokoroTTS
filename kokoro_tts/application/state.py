"""Application state and orchestration for audio generation."""
from __future__ import annotations

import inspect
import logging
from typing import Callable, Generator

import torch

from ..constants import SAMPLE_RATE
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
from ..domain.normalization import TextNormalizer


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
        self.last_saved_paths: list[str] = []
        self._pipeline_style_param_cache: dict[type, str | None] = {}

    def _persist_morphology(
        self,
        parts: list[list[DialogueSegment]],
        *,
        source: str,
    ) -> None:
        if self.morphology_repository is None:
            return
        morph_parts = [
            [(segment.voice, segment.text) for segment in segments]
            for segments in parts
            if segments
        ]
        if not morph_parts:
            return
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
        pause_tensor = torch.zeros(pause_samples) if pause_samples else None
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
            "Generate start: text_len=%s parts=%s voice=%s style=%s speed=%s use_gpu=%s pause_seconds=%s output_format=%s",
            len(text),
            len(parts),
            voice,
            style_preset,
            speed,
            use_gpu,
            pause_seconds,
            output_format,
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
            part_segments: list[torch.Tensor] = []
            last_segment_pause_seconds = pause_seconds
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
                if not first_ps and part_ps:
                    first_ps = part_ps
                part_segments.append(audio_tensor)
                last_segment_pause_seconds = segment_pause_seconds
                if segment_index < len(segments) - 1:
                    segment_pause_samples = max(0, int(segment_pause_seconds * SAMPLE_RATE))
                    if segment_pause_samples:
                        part_segments.append(torch.zeros(segment_pause_samples))
            if not part_segments:
                continue
            part_audio = torch.cat(part_segments)
            if save_outputs:
                output_path = output_paths[index]
                try:
                    saved_path = self.audio_writer.save_audio(
                        output_path,
                        part_audio,
                        output_format,
                    )
                    saved_paths.append(saved_path)
                except Exception:
                    self.logger.exception("Failed to save output: %s", output_path)
            combined_segments.append(part_audio)
            if index < len(parts) - 1:
                part_pause_samples = max(0, int(last_segment_pause_seconds * SAMPLE_RATE))
                if part_pause_samples:
                    combined_segments.append(torch.zeros(part_pause_samples))
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
                    pause_audio = (
                        torch.zeros(segment_pause_samples).numpy() if segment_pause_samples else None
                    )
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
