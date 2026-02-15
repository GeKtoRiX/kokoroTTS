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
- lesson text builder via LM Studio OpenAI-compatible API
- morphology SQLite CRUD tab (view/add/edit/delete)

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
.\.venv\Scripts\python.exe scripts\check_english_lexemes.py --text "Look up the words, then run the tests." --json
```

- `doctor.ps1` verifies `.venv`, package imports, tool paths, and `KPipeline` init for `a,b,e,f,h,i,j,p,z`.
- `smoke_full.ps1` runs runtime checks for generate/stream/dialogue/mix/split/output formats/multilingual voices.
- `check_english_lexemes.py` performs a real lexeme split smoke check for English text; add `--verify-llm` to run an extra LM Studio POS verification pass (`LM_VERIFY_*` env vars).

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
.\.venv\Scripts\python.exe scripts\benchmark_morph_ingest.py --parts 5 --segments-per-part 120 --tokens-per-segment 200 --with-verify
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
- `TTS_ONLY_MODE` (`1` disables Morphology DB writes and all LLM requests)
- `LLM_ONLY_MODE` (`1` enables `TTS + Morphology` mode: disables LLM requests, keeps Morphology DB writes enabled)
- `LOG_LEVEL`, `FILE_LOG_LEVEL`, `LOG_DIR`
- `LOG_EVERY_N_SEGMENTS`
- `UI_PRIMARY_HUE`
- `MORPH_DB_ENABLED`, `MORPH_DB_PATH`, `MORPH_DB_TABLE_PREFIX`
- `MORPH_LOCAL_EXPRESSIONS_ENABLED` (`0` by default; set `1` to re-enable local phrasal/idiom extractor)
- `PRONUNCIATION_RULES_PATH` (default `data/pronunciation_rules.json`)
- `WORDNET_DATA_DIR`, `WORDNET_AUTO_DOWNLOAD`, `SPACY_EN_MODEL_AUTO_DOWNLOAD`
- `LM_STUDIO_BASE_URL`, `LM_STUDIO_API_KEY`, `LM_STUDIO_MODEL`
- `LM_STUDIO_TIMEOUT_SECONDS`, `LM_STUDIO_TEMPERATURE`, `LM_STUDIO_MAX_TOKENS`
- `LM_VERIFY_ENABLED`, `LM_VERIFY_BASE_URL`, `LM_VERIFY_API_KEY`, `LM_VERIFY_MODEL`
- `LM_VERIFY_TIMEOUT_SECONDS`, `LM_VERIFY_TEMPERATURE`, `LM_VERIFY_MAX_TOKENS`
- `LM_VERIFY_MAX_RETRIES`, `LM_VERIFY_WORKERS`

For LLM timeouts, set `LM_STUDIO_TIMEOUT_SECONDS=0` or `LM_VERIFY_TIMEOUT_SECONDS=0` to wait without a timeout.

When `MORPH_DB_ENABLED=1`, each `generate` run writes English token analysis into SQLite:
- `<prefix>lexemes` (deduplicated by key, insert-ignore only)
- `<prefix>token_occurrences` (token occurrences, insert-ignore only)
- `<prefix>expressions` (phrasal verbs and idioms)
- `<prefix>reviews` (LM verification results, local-vs-LM comparison)

When `LM_VERIFY_ENABLED=1` and `LM_VERIFY_MODEL` is set, morphology ingest runs a background
LM Studio verification pass for English text sentence-by-sentence (one request per sentence).
Local tags remain the source of truth.
By default (`MORPH_LOCAL_EXPRESSIONS_ENABLED=0`), phrasal verbs and idioms are taken from
LM verify `new_expressions` and merged into the same DB ingest flow.
If LM verify is disabled and local expressions are also disabled, `expressions` table may remain empty.

Generated files are grouped by date inside `OUTPUT_DIR`:
- `OUTPUT_DIR/YYYY-MM-DD/records` for audio files
- `OUTPUT_DIR/YYYY-MM-DD/vocabulary` for morphology exports (`.ods`/`.csv`/`.txt`/`.xlsx`)

UI includes a `Morphology DB Export` accordion with selectable export format (`.ods`, `.csv`, `.txt`, `.xlsx`)
for `lexemes`, `token_occurrences`, `expressions`, `reviews`, or `General table`.
`General table` export uses columns by parts of speech (Noun, Verb, Adjective, etc.) and rows with words.

UI also includes a `Morphology DB` tab with basic CRUD operations for `morphology.sqlite3`:
- browse rows by dataset (`lexemes`, `occurrences`, `expressions`, `reviews`)
- add row from JSON
- update row by `id` (or `dedup_key` for `lexemes`)
- delete row by `id` (or `dedup_key` for `lexemes`)

CRUD JSON examples:

```json
{"source":"manual","token_text":"Cats","lemma":"cat","upos":"NOUN"}
```

Add row example for `occurrences`.

```json
{"token_text":"Dogs","lemma":"dog"}
```

Update row example for `occurrences` with `row id = 15`.

Delete example: set `row id = 15` and click `Delete row`.
For `lexemes`, use `dedup_key` as row id, for example: `run|verb`.

UI also includes a `Pronunciation dictionary` accordion:
- load/apply persistent JSON rules without restart
- import rules from `.json`
- export current rules to `OUTPUT_DIR/YYYY-MM-DD/vocabulary`

UI also includes a `Lesson Builder (LLM)` tab:
- sends raw text to LM Studio over OpenAI-compatible Chat Completions API
- rewrites it into an English lesson script with detailed exercise explanations
- allows inserting generated lesson text back into the main TTS input

UI includes a `Runtime mode` accordion with a single mode selector:
- `Default`
- `TTS + Morphology`
- `Full`

Mode behavior:
- `Default` is plain TTS: morphology ingest/DB writes and all LLM requests are disabled.
- `TTS + Morphology` disables LLM requests but keeps morphology ingest/DB writes enabled.
- `Full` enables TTS + morphology + LLM.
- In `TTS + Morphology`, local phrasal verbs/idioms extraction is forced on even when `MORPH_LOCAL_EXPRESSIONS_ENABLED=0`.
- Tab visibility follows mode:
- `Default`: hides `Lesson Builder` and `Morphology DB` tabs.
- `TTS + Morphology`: hides `Lesson Builder`, shows `Morphology DB`.
- `Full`: shows both `Lesson Builder` and `Morphology DB`.

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
