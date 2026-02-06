@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

if not exist ".venv\Scripts\python.exe" (
  echo Error: .venv not found inside "%PROJECT_DIR%".
  pause
  exit /b 1
)

if not defined GRADIO_SERVER_NAME set "GRADIO_SERVER_NAME=127.0.0.1"
if not defined GRADIO_SERVER_PORT set "GRADIO_SERVER_PORT=7860"
set "APP_HOST=%GRADIO_SERVER_NAME%"
if /I "%APP_HOST%"=="0.0.0.0" set "APP_HOST=127.0.0.1"
if /I "%APP_HOST%"=="[::]" set "APP_HOST=127.0.0.1"
set "APP_URL=http://%APP_HOST%:%GRADIO_SERVER_PORT%"

".venv\Scripts\python.exe" app.py
pause
