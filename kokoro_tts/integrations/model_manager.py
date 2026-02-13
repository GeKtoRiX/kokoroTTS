"""Model and pipeline management for Kokoro."""
from __future__ import annotations

import threading
from typing import Mapping

from kokoro import KModel, KPipeline


class ModelManager:
    def __init__(
        self,
        repo_id: str,
        cuda_available: bool,
        logger,
        pronunciation_rules: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.cuda_available = cuda_available
        self.logger = logger
        self.models: dict[bool, KModel] = {}
        self.pipelines: dict[str, KPipeline] = {}
        self.voice_cache: dict[str, object] = {}
        self.pronunciation_rules: dict[str, dict[str, str]] = {}
        self._applied_pronunciation_keys: dict[str, set[str]] = {}
        self._model_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        self._voice_lock = threading.Lock()
        self.logger.info("Models and pipelines will load on demand")
        if pronunciation_rules:
            self.set_pronunciation_rules(pronunciation_rules, refresh_pipelines=False)

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
        self._apply_pronunciation_rules_to_pipeline(lang_code, pipeline)
        return pipeline

    def _default_lexicon_entries(self, lang_code: str) -> dict[str, str]:
        if lang_code == "a":
            return {"kokoro": "k\u02c8Ok\u0259\u0279O"}
        if lang_code == "b":
            return {"kokoro": "k\u02c8Qk\u0259\u0279Q"}
        return {"kokoro": "k\u02c8Ok\u0259\u0279O"}

    def _rule_variants(self, word: str) -> set[str]:
        variants = {word}
        lowered = word.lower()
        if lowered:
            variants.add(lowered)
        return {entry for entry in variants if entry}

    def _apply_pronunciation_rules_to_pipeline(
        self,
        lang_code: str,
        pipeline: KPipeline,
    ) -> None:
        defaults = self._default_lexicon_entries(lang_code)
        for word, phoneme in defaults.items():
            pipeline.g2p.lexicon.golds[word] = phoneme

        lang_rules = self.pronunciation_rules.get(lang_code, {})
        desired: dict[str, str] = {}
        for word, phoneme in lang_rules.items():
            for variant in self._rule_variants(word):
                desired[variant] = phoneme

        previous = self._applied_pronunciation_keys.get(lang_code, set())
        removable = previous - set(desired) - set(defaults)
        for word in removable:
            if word in pipeline.g2p.lexicon.golds:
                del pipeline.g2p.lexicon.golds[word]

        for word, phoneme in desired.items():
            pipeline.g2p.lexicon.golds[word] = phoneme
        self._applied_pronunciation_keys[lang_code] = set(desired)

        self.logger.debug(
            "Updated lexicon in pipeline=%s custom_entries=%s",
            lang_code,
            len(desired),
        )

    def _normalize_pronunciation_rules(
        self,
        rules: Mapping[str, Mapping[str, str]] | None,
    ) -> dict[str, dict[str, str]]:
        if not rules:
            return {}
        normalized: dict[str, dict[str, str]] = {}
        for raw_lang, raw_entries in rules.items():
            lang_code = str(raw_lang or "").strip().lower()[:1]
            if not lang_code:
                continue
            if not isinstance(raw_entries, Mapping):
                continue
            entries: dict[str, str] = {}
            for raw_word, raw_phoneme in raw_entries.items():
                word = str(raw_word or "").strip()
                phoneme = str(raw_phoneme or "").strip()
                if not word or not phoneme:
                    continue
                entries[word] = phoneme
            if entries:
                normalized[lang_code] = entries
        return normalized

    def set_pronunciation_rules(
        self,
        rules: Mapping[str, Mapping[str, str]] | None,
        *,
        refresh_pipelines: bool = True,
    ) -> int:
        normalized = self._normalize_pronunciation_rules(rules)
        with self._pipeline_lock:
            changed = normalized != self.pronunciation_rules
            self.pronunciation_rules = normalized
            if refresh_pipelines and changed:
                for lang_code, pipeline in self.pipelines.items():
                    self._apply_pronunciation_rules_to_pipeline(lang_code, pipeline)
        total_entries = sum(len(entries) for entries in normalized.values())
        if changed:
            self.logger.info(
                "Pronunciation rules ready: languages=%s entries=%s",
                len(normalized),
                total_entries,
            )
        return total_entries

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
