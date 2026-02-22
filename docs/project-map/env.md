# Environment Variable Catalog

| Variable | Code Defaults | `.env.example` | Referenced In |
| --- | --- | --- | --- |
| `APP_STATE_PATH` | `"data/app_state.json"` | `data\\app_state.json` | `kokoro_tts/config.py` |
| `DEFAULT_CONCURRENCY_LIMIT` | `<none>` | `0` | `<not found>` |
| `DEFAULT_OUTPUT_FORMAT` | `"wav"` | `wav` | `kokoro_tts/config.py` |
| `ESPEAK_DATA_PATH` | `<process-env>` | `tools\\espeak\\runtime\\eSpeak NG\\espeak-ng-data` | `scripts/doctor.ps1` |
| `FFMPEG_BINARY` | `"", <process-env>, <unset>` | `tools\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe` | `kokoro_tts/integrations/ffmpeg.py, kokoro_tts/storage/audio_writer.py, scripts/doctor.ps1, scripts/smoke_full.ps1, tests/integrations/test_integrations_helpers.py` |
| `FFPROBE_BINARY` | `"", <process-env>, <unset>` | `tools\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin\\ffprobe.exe` | `kokoro_tts/integrations/ffmpeg.py, scripts/doctor.ps1, tests/integrations/test_integrations_helpers.py` |
| `FILE_LOG_LEVEL` | `"DEBUG"` | `DEBUG` | `kokoro_tts/config.py` |
| `HF_TOKEN` | `<unset>` | `<not documented>` | `app.py` |
| `HISTORY_LIMIT` | `<none>` | `5` | `<not found>` |
| `KOKORO_REPO_ID` | `"hexgrad/Kokoro-82M"` | `hexgrad/Kokoro-82M` | `kokoro_tts/config.py, scripts/doctor.ps1` |
| `KOKORO_SKIP_APP_INIT` | `<process-env>, <unset>` | `<not documented>` | `scripts/doctor.ps1` |
| `LOG_DIR` | `"logs"` | `logs` | `kokoro_tts/config.py` |
| `LOG_EVERY_N_SEGMENTS` | `<none>` | `10` | `<not found>` |
| `LOG_LEVEL` | `"INFO"` | `INFO` | `kokoro_tts/config.py` |
| `MAX_CHUNK_CHARS` | `<none>` | `500` | `<not found>` |
| `MORPH_ASYNC_INGEST` | `<none>` | `0` | `<not found>` |
| `MORPH_ASYNC_MAX_PENDING` | `<none>` | `8` | `<not found>` |
| `MORPH_DB_ENABLED` | `"0"` | `1` | `kokoro_tts/config.py` |
| `MORPH_DB_PATH` | `"data/morphology.sqlite3"` | `data\\morphology.sqlite3` | `kokoro_tts/config.py` |
| `MORPH_DB_TABLE_PREFIX` | `"morph_"` | `morph_` | `kokoro_tts/config.py` |
| `MORPH_FLAIR_MODELS` | `""` | `<not documented>` | `kokoro_tts/domain/morphology.py` |
| `MORPH_LOCAL_EXPRESSIONS_ENABLED` | `"0"` | `0` | `kokoro_tts/config.py` |
| `MORPH_SPACY_MODELS` | `""` | `<not documented>` | `kokoro_tts/domain/expressions.py, kokoro_tts/domain/morphology.py` |
| `MORPH_STANZA_PACKAGE` | `""` | `<not documented>` | `kokoro_tts/domain/morphology.py` |
| `NORMALIZE_NUMBERS` | `"1"` | `1` | `kokoro_tts/config.py` |
| `NORMALIZE_TIMES` | `"1"` | `1` | `kokoro_tts/config.py` |
| `OUTPUT_DIR` | `"outputs"` | `outputs` | `kokoro_tts/config.py` |
| `PATH` | `"", <process-env>` | `<not documented>` | `kokoro_tts/integrations/ffmpeg.py, scripts/smoke_full.ps1, tests/integrations/test_integrations_helpers.py` |
| `PHONEMIZER_ESPEAK_LIBRARY` | `<process-env>` | `tools\\espeak\\runtime\\eSpeak NG\\libespeak-ng.dll` | `scripts/doctor.ps1, scripts/smoke_full.ps1` |
| `POSTFX_CROSSFADE_MS` | `<none>` | `25` | `<not found>` |
| `POSTFX_ENABLED` | `<none>` | `0` | `<not found>` |
| `POSTFX_FADE_IN_MS` | `<none>` | `12` | `<not found>` |
| `POSTFX_FADE_OUT_MS` | `<none>` | `40` | `<not found>` |
| `POSTFX_LOUDNESS_ENABLED` | `<none>` | `1` | `<not found>` |
| `POSTFX_LOUDNESS_TARGET_LUFS` | `<none>` | `-16` | `<not found>` |
| `POSTFX_LOUDNESS_TRUE_PEAK_DB` | `<none>` | `-1.0` | `<not found>` |
| `POSTFX_TRIM_ENABLED` | `<none>` | `1` | `<not found>` |
| `POSTFX_TRIM_KEEP_MS` | `<none>` | `25` | `<not found>` |
| `POSTFX_TRIM_THRESHOLD_DB` | `<none>` | `-42` | `<not found>` |
| `PRONUNCIATION_RULES_PATH` | `"data/pronunciation_rules.json"` | `data\\pronunciation_rules.json` | `app.py` |
| `SPACE_ID` | `""` | `<not documented>` | `kokoro_tts/config.py` |
| `SPACY_EN_AUTO_DOWNLOAD_MODELS` | `""` | `<not documented>` | `kokoro_tts/domain/expressions.py` |
| `SPACY_EN_MODEL_AUTO_DOWNLOAD` | `<none>` | `1` | `<not found>` |
| `TORCH_NUM_INTEROP_THREADS` | `<none>` | `0` | `<not found>` |
| `TORCH_NUM_THREADS` | `<none>` | `0` | `<not found>` |
| `TTS_ONLY_MODE` | `<none>` | `0` | `<not found>` |
| `TTS_PREWARM_ASYNC` | `<none>` | `0` | `<not found>` |
| `TTS_PREWARM_ENABLED` | `<none>` | `1` | `<not found>` |
| `TTS_PREWARM_STYLE` | `"neutral"` | `neutral` | `kokoro_tts/config.py` |
| `TTS_PREWARM_VOICE` | `"af_heart"` | `af_heart` | `kokoro_tts/config.py` |
| `WORDNET_AUTO_DOWNLOAD` | `<none>` | `1` | `<not found>` |
| `WORDNET_DATA_DIR` | `str(Path(__file__` | `data\\nltk_data` | `kokoro_tts/domain/expressions.py` |
