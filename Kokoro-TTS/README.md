---
title: Kokoro TTS
emoji: ❤️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.24.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Upgraded to v1.0!
disable_embedding: true
---

# Kokoro TTS

Kokoro TTS is a Gradio app for the Kokoro-82M text-to-speech model. It generates audio files or streams audio while it is synthesized. Outputs are saved to `outputs/` and recent results are listed in the History panel.

## Capabilities
- Generate speech from text with a curated set of voices.
- Stream audio chunks in real time or render a full file in one pass.
- Switch speakers inline with `[voice=...]` tags and mix voices by averaging multiple IDs.
- Split one input into multiple output files with `|`.
- Control speed and optional pauses between sentences.
- Toggle text normalization for times and numbers.
- Preview tokenizer output for the current text (up to model context limits).
- Choose output format (`wav`, `mp3`, `ogg`); `mp3` and `ogg` require ffmpeg and fall back to `wav` if missing.
- Select GPU or CPU (automatic CPU fallback on GPU errors).
- Keep and clear a history of saved files.
- Pronunciation tweaks via Markdown links and `/slashes/`, plus intonation tweaks with punctuation or stress markers (see UI note).
- Gradio API endpoints for generate, tokenize, stream, and predict when API is enabled.
- Configure model, logging, output paths, and concurrency with environment variables.

## Minimal examples
Generate a single clip:
```text
Hello, world.
```

Split into multiple files:
```text
Hello there.|This is the second file.
```

Switch voices inside one input:
```text
[voice=af_heart]Hello.[voice=am_michael]Hi there.
```

Mix voices:
```text
[voice=af_heart,am_michael]This is a mixed voice.
```

Pronunciation hint using Markdown link syntax:
```text
[Kokoro](/ko-ko-ro/)
```

Time and number normalization (if enabled):
```text
Meet me at 12:30. I got 42% on the 3rd try.
```

## Configuration
Environment variables (all optional):
- `KOKORO_REPO_ID`: model repo id (default `hexgrad/Kokoro-82M`).
- `OUTPUT_DIR`: where generated files are stored (default `outputs`).
- `MAX_CHUNK_CHARS`: soft chunk size for generation (default `500`).
- `HISTORY_LIMIT`: number of history slots in the UI (default `5`).
- `NORMALIZE_TIMES`: enable time normalization (`1` or `0`).
- `NORMALIZE_NUMBERS`: enable number normalization (`1` or `0`).
- `DEFAULT_OUTPUT_FORMAT`: default format (`wav`, `mp3`, `ogg`).
- `FFMPEG_BINARY`/`FFPROBE_BINARY`: explicit ffmpeg paths if not on PATH.
- `LOG_LEVEL`/`FILE_LOG_LEVEL` and `LOG_DIR`: logging controls and log location.
- `DEFAULT_CONCURRENCY_LIMIT`: Gradio queue limit (0 disables).
- `SPACE_ID`: used to determine streaming character limits on Spaces.
- `UI_PRIMARY_HUE`: primary theme hue in the UI.
- `HF_TOKEN`: optional token for higher Hugging Face Hub rate limits.

## Run locally
```powershell
pip install -r requirements.txt
python app.py
```
