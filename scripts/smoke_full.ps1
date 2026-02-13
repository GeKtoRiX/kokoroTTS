[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line.TrimStart().StartsWith("#")) {
            continue
        }
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) {
            continue
        }
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

Import-DotEnv -Path ".env"

$venvPython = Join-Path $ProjectRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw ".venv is missing. Run scripts\\bootstrap_local.ps1 first."
}

if ($env:FFMPEG_BINARY) {
    $env:PATH = (Split-Path -Parent $env:FFMPEG_BINARY) + ";" + $env:PATH
}
if ($env:PHONEMIZER_ESPEAK_LIBRARY) {
    $env:PATH = (Split-Path -Parent $env:PHONEMIZER_ESPEAK_LIBRARY) + ";" + $env:PATH
}

Write-Step "Running runtime smoke tests (this may download model/voices on first run)"
@'
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("OUTPUT_DIR", str(Path("outputs") / "smoke"))
os.environ.setdefault("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")

import app


def assert_generate(label: str, **kwargs):
    result, tokens = app.generate_first(**kwargs)
    assert result is not None, f"{label}: no audio generated"
    assert tokens, f"{label}: empty token output"
    return result, tokens


assert_generate(
    "basic",
    text="Hello from Kokoro.",
    voice="af_heart",
    use_gpu=False,
)

stream_gen = app.generate_all(
    text="Streaming hello from Kokoro.",
    voice="af_heart",
    use_gpu=False,
)
first_chunk = next(stream_gen)
assert first_chunk is not None and first_chunk[0] == 24000, "stream: invalid first chunk"

assert_generate(
    "dialogue",
    text="[voice=af_heart]Hello.[voice=am_michael]Hi there.",
    voice="af_heart",
    use_gpu=False,
)

assert_generate(
    "voice_mix",
    text="This is a mixed voice example.",
    voice="af_heart",
    mix_enabled=True,
    voice_mix=["af_heart", "af_bella"],
    use_gpu=False,
)

assert_generate(
    "split",
    text="Part one.|Part two.",
    voice="af_heart",
    use_gpu=False,
)
assert len(app.APP_STATE.last_saved_paths) >= 2, "split: expected at least 2 output files"

assert_generate(
    "format_mp3",
    text="MP3 output format check.",
    voice="af_heart",
    output_format="mp3",
    use_gpu=False,
)
mp3_path = Path(app.APP_STATE.last_saved_paths[-1])
assert mp3_path.suffix.lower() in {".mp3", ".wav"}, "mp3 fallback path is invalid"

assert_generate(
    "format_ogg",
    text="OGG output format check.",
    voice="af_heart",
    output_format="ogg",
    use_gpu=False,
)
ogg_path = Path(app.APP_STATE.last_saved_paths[-1])
assert ogg_path.suffix.lower() in {".ogg", ".wav"}, "ogg fallback path is invalid"

multi_samples = {
    "a": ("af_heart", "Hello from Kokoro."),
    "b": ("bf_emma", "Good morning from Britain."),
    "e": ("ef_dora", "Hola desde Kokoro."),
    "f": ("ff_siwis", "Bonjour depuis Kokoro."),
    "h": ("hf_alpha", "Namaste doston."),
    "i": ("if_sara", "Ciao da Kokoro."),
    "j": ("jf_alpha", "こんにちは。"),
    "p": ("pf_dora", "Ola do Brasil."),
    "z": ("zf_xiaobei", "你好，Kokoro。"),
}

for lang, (voice, text) in multi_samples.items():
    assert_generate(
        f"lang_{lang}",
        text=text,
        voice=voice,
        use_gpu=False,
    )

print("SMOKE_OK")
'@ | & $venvPython -

Write-Step "Smoke tests finished"
