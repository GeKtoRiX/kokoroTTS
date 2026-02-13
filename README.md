# Kokoro TTS (Windows, project-local `.venv`)

Local Gradio app for `hexgrad/Kokoro-82M` with:
- generate + stream modes
- voice mix and inline dialogue tags
- multilingual voices (9 languages)
- history panel + file cleanup
- text normalization (time/number)
- output formats `wav/mp3/ogg` (ffmpeg fallback handling)

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
