@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\..\.."
set "PROJECT_DIR=%CD%"

set "LOG_DIR="
set "OUTPUT_DIR="
if exist ".env" (
  for /f "usebackq tokens=* delims=" %%L in (".env") do (
    set "LINE=%%L"
    if not "!LINE!"=="" if "!LINE:~0,1!" NEQ "#" (
      for /f "tokens=1,* delims==" %%A in ("!LINE!") do (
        if /I "%%A"=="LOG_DIR" set "LOG_DIR=%%B"
        if /I "%%A"=="OUTPUT_DIR" set "OUTPUT_DIR=%%B"
      )
    )
  )
)

if not defined LOG_DIR set "LOG_DIR=logs"
if not defined OUTPUT_DIR set "OUTPUT_DIR=outputs"
set "LOG_DIR=!LOG_DIR:"=!"
set "OUTPUT_DIR=!OUTPUT_DIR:"=!"

set "LOG_DIR_ABS=!LOG_DIR!"
if not "!LOG_DIR_ABS:~1,1!"==":" if not "!LOG_DIR_ABS:~0,2!"=="\\" set "LOG_DIR_ABS=%PROJECT_DIR%\!LOG_DIR_ABS!"
for %%I in ("!LOG_DIR_ABS!") do set "LOG_DIR_ABS=%%~fI"

set "OUTPUT_DIR_ABS=!OUTPUT_DIR!"
if not "!OUTPUT_DIR_ABS:~1,1!"==":" if not "!OUTPUT_DIR_ABS:~0,2!"=="\\" set "OUTPUT_DIR_ABS=%PROJECT_DIR%\!OUTPUT_DIR_ABS!"
for %%I in ("!OUTPUT_DIR_ABS!") do set "OUTPUT_DIR_ABS=%%~fI"

echo.
echo ===============================================
echo  PROJECT CLEANUP
echo ===============================================
echo This script will delete:
echo  - contents of LOG_DIR:     !LOG_DIR_ABS!
echo  - contents of OUTPUT_DIR:  !OUTPUT_DIR_ABS!
echo  - project caches:
echo      .pytest_cache, .mypy_cache, .ruff_cache, .coverage, htmlcov
echo      __pycache__ and *.pyc/*.pyo under kokoro_tts/, tests/, scripts/
echo.
set /p "CONFIRM=Type CLEAN to continue: "
if /I not "!CONFIRM!"=="CLEAN" (
  echo.
  echo Cancelled. No changes were made.
  exit /b 1
)

if /I "!LOG_DIR_ABS!"=="%PROJECT_DIR%" (
  echo Error: LOG_DIR points to the project root. Refusing to continue.
  exit /b 1
)
if "!LOG_DIR_ABS:~1,1!"==":" if "!LOG_DIR_ABS:~3!"=="" (
  echo Error: LOG_DIR points to a drive root. Refusing to continue.
  exit /b 1
)

if /I "!OUTPUT_DIR_ABS!"=="%PROJECT_DIR%" (
  echo Error: OUTPUT_DIR points to the project root. Refusing to continue.
  exit /b 1
)
if "!OUTPUT_DIR_ABS:~1,1!"==":" if "!OUTPUT_DIR_ABS:~3!"=="" (
  echo Error: OUTPUT_DIR points to a drive root. Refusing to continue.
  exit /b 1
)

if not exist "!LOG_DIR_ABS!" (
  mkdir "!LOG_DIR_ABS!" >nul 2>&1
) else (
  for /f "delims=" %%I in ('dir /a /b "!LOG_DIR_ABS!" 2^>nul') do (
    if exist "!LOG_DIR_ABS!\%%I\*" (
      rd /s /q "!LOG_DIR_ABS!\%%I" >nul 2>&1
    ) else (
      del /f /q "!LOG_DIR_ABS!\%%I" >nul 2>&1
    )
  )
)
echo Cleared LOG_DIR contents:
echo   !LOG_DIR_ABS!

if not exist "!OUTPUT_DIR_ABS!" (
  mkdir "!OUTPUT_DIR_ABS!" >nul 2>&1
) else (
  for /f "delims=" %%I in ('dir /a /b "!OUTPUT_DIR_ABS!" 2^>nul') do (
    if exist "!OUTPUT_DIR_ABS!\%%I\*" (
      rd /s /q "!OUTPUT_DIR_ABS!\%%I" >nul 2>&1
    ) else (
      del /f /q "!OUTPUT_DIR_ABS!\%%I" >nul 2>&1
    )
  )
)
echo Cleared OUTPUT_DIR contents:
echo   !OUTPUT_DIR_ABS!

for %%D in (".pytest_cache" ".mypy_cache" ".ruff_cache" "htmlcov" "__pycache__") do (
  if exist "%PROJECT_DIR%\%%~D" (
    rd /s /q "%PROJECT_DIR%\%%~D" >nul 2>&1
    echo Removed cache directory:
    echo   %PROJECT_DIR%\%%~D
  )
)
if exist "%PROJECT_DIR%\.coverage" (
  del /f /q "%PROJECT_DIR%\.coverage" >nul 2>&1
  echo Removed cache file:
  echo   %PROJECT_DIR%\.coverage
)

for %%B in ("%PROJECT_DIR%\kokoro_tts" "%PROJECT_DIR%\tests" "%PROJECT_DIR%\scripts") do (
  if exist "%%~fB" (
    for /d /r "%%~fB" %%D in (__pycache__) do rd /s /q "%%~fD" >nul 2>&1
    for /r "%%~fB" %%F in (*.pyc *.pyo) do del /f /q "%%~fF" >nul 2>&1
    echo Cleared Python bytecode cache under:
    echo   %%~fB
  )
)

echo.
echo Done. Logs, outputs, and safe project caches are cleared.
exit /b 0
