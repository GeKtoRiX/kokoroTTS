# Kokoro TTS (Windows, project-local `.venv`)

Local Tkinter desktop app for `hexgrad/Kokoro-82M` with:
- generate + stream modes
- voice mix and inline dialogue tags
- persistent pronunciation dictionary (JSON rules by language)
- multilingual voices (9 languages)
- history panel + file cleanup
- text normalization (time/number)
- style presets (`neutral`, `narrator`, `energetic`)
- output formats `wav/mp3/ogg` (ffmpeg fallback handling)
- morphology SQLite read-only preview tab

## Known issues

- `Stream`: in some sessions, voice can stick to the first selected voice even after changing it in UI. This bug is currently unresolved.

## Isolation rule

This project is configured for **project-local runtime only**:
- Python runtime is downloaded into `tools/python312/runtime`
- Python packages are installed only into `.venv`
- ffmpeg and espeak-ng are kept in `tools/`
- no global `pip install` is required
- desktop Audio player uses `python-vlc` and requires VLC runtime installed on the machine

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
.\.venv\Scripts\python.exe scripts\check_english_lexemes.py --text "Look up the words, then run the tests." --json
```

- `doctor.ps1` verifies `.venv`, package imports, tool paths, and `KPipeline` init for `a,b,e,f,h,i,j,p,z`.
- `smoke_full.ps1` runs runtime checks for generate/stream/dialogue/mix/split/output formats/multilingual voices.
- `check_english_lexemes.py` performs a real lexeme split smoke check for English text.

## Maintenance scripts (.bat)

Technical `.bat` scripts are grouped under `scripts\bat\` (project root keeps only `run.bat`):

```powershell
.\scripts\bat\cleanup_project.bat
.\scripts\bat\clear_morphology_db.bat
```

- `cleanup_project.bat` clears `LOG_DIR`, `OUTPUT_DIR`, and safe project-local caches.
- `clear_morphology_db.bat` removes all rows from morphology tables while keeping the SQLite file.

## Developer quality checks

Install dev tooling into `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest --cov=kokoro_tts --cov=app --cov-report=term-missing -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
```

Synthetic ingest benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_morph_ingest.py --parts 5 --segments-per-part 120 --tokens-per-segment 200
```

Real TTS inference profiling (first vs warm latency, CPU/GPU):

```powershell
.\.venv\Scripts\python.exe scripts\profile_tts_inference.py --mode both --warm-runs 3 --tts-only 1 --save-outputs 0
```

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
- `TTS_ONLY_MODE` (`1` disables Morphology DB writes)
- `LOG_LEVEL`, `FILE_LOG_LEVEL`, `LOG_DIR`
- `LOG_EVERY_N_SEGMENTS`
- `TORCH_NUM_THREADS`, `TORCH_NUM_INTEROP_THREADS` (`0` keeps torch defaults)
- `TTS_PREWARM_ENABLED`, `TTS_PREWARM_ASYNC`, `TTS_PREWARM_VOICE`, `TTS_PREWARM_STYLE`
- `MORPH_DB_ENABLED`, `MORPH_DB_PATH`, `MORPH_DB_TABLE_PREFIX`
- `MORPH_LOCAL_EXPRESSIONS_ENABLED` (`0` by default; set `1` to re-enable local phrasal/idiom extractor)
- `MORPH_ASYNC_INGEST` (`0` by default; optional background morphology DB writes)
- `MORPH_ASYNC_MAX_PENDING` (max pending async morphology tasks before sync fallback)
- `PRONUNCIATION_RULES_PATH` (default `data/pronunciation_rules.json`)
- `WORDNET_DATA_DIR`, `WORDNET_AUTO_DOWNLOAD`, `SPACY_EN_MODEL_AUTO_DOWNLOAD`
- `MORPH_SPACY_MODELS` (comma-separated spaCy model priority, default tries `trf,lg,md,sm`)
- `MORPH_STANZA_PACKAGE`, `MORPH_STANZA_USE_GPU`
- `MORPH_FLAIR_ENABLED` (`1` by default), `MORPH_FLAIR_MODELS` (comma-separated Flair model priority)
- `MORPH_TEXTACY_ENABLED` (`1` by default), `MORPH_PHRASEMACHINE_ENABLED` (`1` by default)
- `MORPH_PYWSD_ENABLED` (`0` by default; optional idiom noise filter)
- `MORPH_PYWSD_AUTO_DOWNLOAD` (`1` by default; downloads required NLTK resources for PyWSD)
- `SPACY_EN_AUTO_DOWNLOAD_MODELS` (comma-separated, default `en_core_web_sm`)

When `MORPH_DB_ENABLED=1`, each `generate` run writes English token analysis into SQLite:
- `<prefix>lexemes` (deduplicated by key, insert-ignore only)
- `<prefix>token_occurrences` (token occurrences, insert-ignore only)
- `<prefix>expressions` (phrasal verbs and idioms)

Generated files are grouped by date inside `OUTPUT_DIR`:
- `OUTPUT_DIR/YYYY-MM-DD/records` for audio files
- `OUTPUT_DIR/YYYY-MM-DD/vocabulary` for morphology exports (`.ods`/`.csv`/`.txt`/`.xlsx`)

UI includes a `Morphology DB Export` accordion with selectable export format (`.ods`, `.csv`, `.txt`, `.xlsx`)
for `lexemes`, `token_occurrences`, `expressions`, or `General table`.
`General table` export uses columns by parts of speech (Noun, Verb, Adjective, etc.) and rows with words.

UI also includes a read-only `Morphology DB` tab for browsing datasets
(`lexemes`, `occurrences`, `expressions`).
Manual row edits are intentionally removed from the app; use an external DB tool.

UI also includes a `Pronunciation dictionary` accordion:
- load/apply persistent JSON rules without restart
- import rules from `.json`
- export current rules to `OUTPUT_DIR/YYYY-MM-DD/vocabulary`

UI includes a `Runtime mode` accordion with a single mode selector:
- `Default`
- `TTS + Morphology`

Mode behavior:
- `Default` is plain TTS: morphology ingest/DB writes are disabled.
- `TTS + Morphology` keeps morphology ingest/DB writes enabled.
- Tab visibility follows mode:
- `Default`: hides `Morphology DB`.
- `TTS + Morphology`: shows `Morphology DB`.

Expression detection uses:
- spaCy `DependencyMatcher` for verb + particle phrasal verbs (lemma-based)
- WordNet-window matcher for separable phrasal verbs when dependency parse misses
- optional `textacy` token-pattern matcher for verb-particle windows
- optional `phrasemachine` phrase spans as additional phrasal candidates
- WordNet-powered phrase matching for idioms (WordNet is downloaded locally to `WORDNET_DATA_DIR`)
- optional `pywsd` context filter for noisy idioms

Lexeme tagging cascade is:
- Stanza (primary)
- spaCy (merge/backfill)
- Flair POS tagger (optional fallback for unresolved tags)
- deterministic heuristics (`NUM`, `SYM`, `X`) as final fallback

High-accuracy setup (recommended):
- install `spacy-transformers` (already in `requirements.txt` for Python < 3.14)
- download transformer model with `python -m spacy download en_core_web_trf`
- set `MORPH_SPACY_MODELS=en_core_web_trf,en_core_web_lg,en_core_web_md,en_core_web_sm`

Important: do not install `en_core_web_trf` from PyPI package name `en-core-web-trf`; this
name is a placeholder and points to `spacy package` download flow.

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
