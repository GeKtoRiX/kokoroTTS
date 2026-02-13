@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if not exist ".venv\Scripts\python.exe" (
  echo Error: .venv not found inside "%PROJECT_DIR%".
  echo Run: powershell -ExecutionPolicy Bypass -File scripts\bootstrap_local.ps1
  pause
  exit /b 1
)

if exist ".env" (
  for /f "usebackq tokens=* delims=" %%L in (".env") do (
    set "LINE=%%L"
    if not "!LINE!"=="" if "!LINE:~0,1!" NEQ "#" (
      for /f "tokens=1,* delims==" %%A in ("!LINE!") do (
        if not "%%A"=="" set "%%A=%%B"
      )
    )
  )
)

if not defined FFMPEG_BINARY (
  for /f "delims=" %%I in ('dir /s /b "tools\ffmpeg\ffmpeg.exe" 2^>nul') do (
    if not defined FFMPEG_BINARY set "FFMPEG_BINARY=%%~fI"
  )
)
if not defined FFPROBE_BINARY (
  for /f "delims=" %%I in ('dir /s /b "tools\ffmpeg\ffprobe.exe" 2^>nul') do (
    if not defined FFPROBE_BINARY set "FFPROBE_BINARY=%%~fI"
  )
)
if not defined ESPEAK_DATA_PATH (
  for /f "delims=" %%I in ('dir /s /b /ad "tools\espeak\*espeak-ng-data" 2^>nul') do (
    if not defined ESPEAK_DATA_PATH set "ESPEAK_DATA_PATH=%%~fI"
  )
)
if not defined PHONEMIZER_ESPEAK_LIBRARY (
  for /f "delims=" %%I in ('dir /s /b "tools\espeak\libespeak-ng.dll" 2^>nul') do (
    if not defined PHONEMIZER_ESPEAK_LIBRARY set "PHONEMIZER_ESPEAK_LIBRARY=%%~fI"
  )
)
if defined FFMPEG_BINARY for %%I in ("%FFMPEG_BINARY%") do set "PATH=%%~dpI;%PATH%"
if defined PHONEMIZER_ESPEAK_LIBRARY for %%I in ("%PHONEMIZER_ESPEAK_LIBRARY%") do set "PATH=%%~dpI;%PATH%"

if not defined GRADIO_SERVER_NAME set "GRADIO_SERVER_NAME=127.0.0.1"
if not defined GRADIO_SERVER_PORT set "GRADIO_SERVER_PORT=7860"
set "APP_HOST=%GRADIO_SERVER_NAME%"
if /I "%APP_HOST%"=="0.0.0.0" set "APP_HOST=127.0.0.1"
if /I "%APP_HOST%"=="[::]" set "APP_HOST=127.0.0.1"
set "APP_URL=http://%APP_HOST%:%GRADIO_SERVER_PORT%"

".venv\Scripts\python.exe" app.py
pause
