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

Write-Step "Python and pip"
& $venvPython --version
& $venvPython -m pip --version

Write-Step "Binary tool discovery"
if (-not $env:FFMPEG_BINARY) {
    $env:FFMPEG_BINARY = Get-ChildItem -LiteralPath "tools\\ffmpeg" -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $env:FFPROBE_BINARY) {
    $env:FFPROBE_BINARY = Get-ChildItem -LiteralPath "tools\\ffmpeg" -Recurse -Filter "ffprobe.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $env:ESPEAK_DATA_PATH) {
    $env:ESPEAK_DATA_PATH = Get-ChildItem -LiteralPath "tools\\espeak" -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ieq "espeak-ng-data" } |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $env:PHONEMIZER_ESPEAK_LIBRARY) {
    $env:PHONEMIZER_ESPEAK_LIBRARY = Get-ChildItem -LiteralPath "tools\\espeak" -Recurse -Filter "libespeak-ng.dll" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

foreach ($key in @("FFMPEG_BINARY", "FFPROBE_BINARY", "ESPEAK_DATA_PATH", "PHONEMIZER_ESPEAK_LIBRARY")) {
    $value = [Environment]::GetEnvironmentVariable($key, "Process")
    if ($value) {
        Write-Host "$key = $value"
    }
    else {
        Write-Warning "$key is not set"
    }
}

if ($env:FFMPEG_BINARY -and (Test-Path -LiteralPath $env:FFMPEG_BINARY)) {
    & $env:FFMPEG_BINARY -version | Select-Object -First 1
}

Write-Step "Python package and language checks"
$env:KOKORO_SKIP_APP_INIT = "1"
@'
import os
import sys

print("Python executable:", sys.executable)
print("KOKORO_SKIP_APP_INIT:", os.getenv("KOKORO_SKIP_APP_INIT"))

import kokoro
import misaki
print("kokoro version:", kokoro.__version__)
print("misaki version:", misaki.__version__)

from kokoro import KPipeline

repo_id = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
languages = ["a", "b", "e", "f", "h", "i", "j", "p", "z"]
failed = []
for lang in languages:
    try:
        KPipeline(lang_code=lang, model=False, repo_id=repo_id)
        print(f"[OK] lang={lang}")
    except Exception as exc:
        failed.append((lang, str(exc)))
        print(f"[FAIL] lang={lang}: {exc}")

import app
print("app import: OK")

if failed:
    raise SystemExit("Language initialization failures detected")
'@ | & $venvPython -

Write-Step "Doctor checks passed"
