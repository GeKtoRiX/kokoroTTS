# Kokoro TTS (Windows, project-local `.venv`)

Local Gradio app for `hexgrad/Kokoro-82M` with:
- generate + stream modes
- voice mix and inline dialogue tags
- persistent pronunciation dictionary (JSON rules by language)
- multilingual voices (9 languages)
- history panel + file cleanup
- text normalization (time/number)
- style presets (`neutral`, `narrator`, `energetic`)
- output formats `wav/mp3/ogg` (ffmpeg fallback handling)

## Known issues

- `Stream`: in some sessions, voice can stick to the first selected voice even after changing it in UI. This bug is currently unresolved.

## Isolation rule

This project is configured for **project-local runtime only**:
- Python runtime is downloaded into `tools/python312/runtime`
- Python packages are installed only into `.venv`
- ffmpeg and espeak-ng are kept in `tools/`
- no global `pip install` is required

## Quick start

```powershell
cd D:\pythonProjects\kokoroTTS
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_local.ps1
.\run.bat
```

First run may download model weights and voices from Hugging Face.

## What bootstrap does

`scripts/bootstrap_local.ps1`:
1. Downloads/extracts Python 3.12.10 to `tools/python312/runtime`
2. Creates `.venv` from that local runtime
3. Installs dependencies from `requirements.txt` into `.venv`
4. Downloads/extracts ffmpeg to `tools/ffmpeg`
5. Downloads/extracts espeak-ng to `tools/espeak/runtime`
6. Creates `.env` from `.env.example` (if missing)
7. Writes discovered local binary paths into `.env`

## Diagnostics and smoke tests

```powershell
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
powershell -ExecutionPolicy Bypass -File scripts\smoke_full.ps1
```

- `doctor.ps1` verifies `.venv`, package imports, tool paths, and `KPipeline` init for `a,b,e,f,h,i,j,p,z`.
- `smoke_full.ps1` runs runtime checks for generate/stream/dialogue/mix/split/output formats/multilingual voices.

## Environment variables

Create/edit `.env` (or start from `.env.example`):

- `FFMPEG_BINARY`, `FFPROBE_BINARY`
- `ESPEAK_DATA_PATH`, `PHONEMIZER_ESPEAK_LIBRARY`
- `KOKORO_REPO_ID` (default `hexgrad/Kokoro-82M`)
- `OUTPUT_DIR` (default `outputs`)
- `MAX_CHUNK_CHARS` (default `500`)
- `HISTORY_LIMIT` (default `5`)
- `NORMALIZE_TIMES`, `NORMALIZE_NUMBERS`
- `DEFAULT_OUTPUT_FORMAT` (`wav|mp3|ogg`)
- `DEFAULT_CONCURRENCY_LIMIT`
- `LOG_LEVEL`, `FILE_LOG_LEVEL`, `LOG_DIR`
- `LOG_EVERY_N_SEGMENTS`
- `UI_PRIMARY_HUE`
- `MORPH_DB_ENABLED`, `MORPH_DB_PATH`, `MORPH_DB_TABLE_PREFIX`
- `PRONUNCIATION_RULES_PATH` (default `data/pronunciation_rules.json`)
- `WORDNET_DATA_DIR`, `WORDNET_AUTO_DOWNLOAD`, `SPACY_EN_MODEL_AUTO_DOWNLOAD`

When `MORPH_DB_ENABLED=1`, each `generate` run writes English token analysis into SQLite:
- `<prefix>lexemes` (deduplicated by key, insert-ignore only)
- `<prefix>token_occurrences` (token occurrences, insert-ignore only)
- `<prefix>expressions` (phrasal verbs and idioms)

Generated files are grouped by date inside `OUTPUT_DIR`:
- `OUTPUT_DIR/YYYY-MM-DD/records` for audio files
- `OUTPUT_DIR/YYYY-MM-DD/vocabulary` for morphology exports (`.ods`/`.csv`)

UI includes a `Morphology DB Export` accordion with `Download ODS` for `lexemes`, `token_occurrences`, `expressions`, or `POS table` (LibreOffice Calc native format).
`POS table` export uses columns by parts of speech (Noun, Verb, Adjective, etc.) and rows with words.

UI also includes a `Pronunciation dictionary` accordion:
- load/apply persistent JSON rules without restart
- import rules from `.json`
- export current rules to `OUTPUT_DIR/YYYY-MM-DD/vocabulary`

Expression detection uses:
- spaCy `DependencyMatcher` for verb + particle phrasal verbs (lemma-based)
- WordNet-powered phrase matching for idioms (WordNet is downloaded locally to `WORDNET_DATA_DIR`)

`run.bat` auto-loads `.env` and auto-detects local tools under `tools/` when variables are not set.

## Voice and language coverage

The app exposes the full Kokoro v1.0 voice set from `VOICES.md`:
- `a` American English
- `b` British English
- `e` Spanish
- `f` French
- `h` Hindi
- `i` Italian
- `j` Japanese
- `p` Brazilian Portuguese
- `z` Mandarin Chinese

UI includes a language selector that filters available voices. Inline `[voice=...]` tags still work per segment.

## Style presets

UI and API expose a `style_preset` parameter with values:
- `neutral`
- `narrator`
- `energetic`

Kokoro itself does not provide native emotion controls. In this app, presets are implemented as runtime tuning:
- speed/pause multipliers
- optional pass-through of a style argument to `KPipeline` only when that argument exists in the installed Kokoro version

## Fallback path (if local Python bootstrap fails)

If `tools/python312/runtime` cannot be used on your machine:

```powershell
winget uninstall --id Python.Python.3.14 --silent --accept-source-agreements
winget install --id Python.Python.3.12 --silent --accept-source-agreements --accept-package-agreements
```

Then recreate `.venv` with Python 3.12 and continue using project-local package installs only.

## Source references

- `https://github.com/hexgrad/kokoro`
- `https://huggingface.co/spaces/hexgrad/Kokoro-TTS/tree/main`
- `https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md`
