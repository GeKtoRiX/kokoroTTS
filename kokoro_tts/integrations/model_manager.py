"""Model and pipeline management for Kokoro."""
from __future__ import annotations

import threading

from kokoro import KModel, KPipeline


class ModelManager:
    def __init__(self, repo_id: str, cuda_available: bool, logger) -> None:
        self.repo_id = repo_id
        self.cuda_available = cuda_available
        self.logger = logger
        self.models: dict[bool, KModel] = {}
        self.pipelines: dict[str, KPipeline] = {}
        self.voice_cache: dict[str, object] = {}
        self._model_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        self._voice_lock = threading.Lock()
        self.logger.info("Models and pipelines will load on demand")

    def get_model(self, use_gpu: bool) -> KModel:
        use_gpu = bool(use_gpu and self.cuda_available)
        key = use_gpu
        model = self.models.get(key)
        if model is not None:
            return model
        with self._model_lock:
            model = self.models.get(key)
            if model is None:
                device = "cuda" if use_gpu else "cpu"
                self.logger.info("Loading model on %s", device)
                try:
                    model = KModel(repo_id=self.repo_id).to(device).eval()
                except Exception:
                    self.logger.exception("Failed to load model on %s", device)
                    raise
                self.models[key] = model
                self.logger.info("Model ready on %s", device)
        return model

    def _init_pipeline(self, lang_code: str) -> KPipeline:
        self.logger.info("Initializing pipeline for language code=%s", lang_code)
        try:
            pipeline = KPipeline(lang_code=lang_code, repo_id=self.repo_id, model=False)
        except Exception:
            self.logger.exception(
                "Failed to initialize pipeline for language code=%s",
                lang_code,
            )
            raise
        if lang_code == "a":
            pipeline.g2p.lexicon.golds["kokoro"] = "kˈOkəɹO"
        elif lang_code == "b":
            pipeline.g2p.lexicon.golds["kokoro"] = "kˈQkəɹQ"
        self.logger.debug("Updated lexicon entries for kokoro in pipeline=%s", lang_code)
        return pipeline

    def get_pipeline(self, voice: str) -> KPipeline:
        lang_code = voice[0] if voice else "a"
        pipeline = self.pipelines.get(lang_code)
        if pipeline is not None:
            return pipeline
        with self._pipeline_lock:
            pipeline = self.pipelines.get(lang_code)
            if pipeline is None:
                pipeline = self._init_pipeline(lang_code)
                self.pipelines[lang_code] = pipeline
        return pipeline

    def get_voice_pack(self, voice: str):
        pack = self.voice_cache.get(voice)
        if pack is not None:
            return pack
        with self._voice_lock:
            pack = self.voice_cache.get(voice)
            if pack is not None:
                return pack
            pipeline = self.get_pipeline(voice)
            self.logger.debug("Loading voice pack: %s", voice)
            try:
                pack = pipeline.load_voice(voice)
            except Exception:
                self.logger.exception("Failed to load voice pack: %s", voice)
                raise
            self.voice_cache[voice] = pack
            self.logger.debug("Voice pack cached: %s", voice)
        return pack
