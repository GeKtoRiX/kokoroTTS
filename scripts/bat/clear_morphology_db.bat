@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\..\.."
set "PROJECT_DIR=%CD%"

set "MORPH_DB_PATH="
set "MORPH_DB_TABLE_PREFIX="
if exist ".env" (
  for /f "usebackq tokens=* delims=" %%L in (".env") do (
    set "LINE=%%L"
    if not "!LINE!"=="" if "!LINE:~0,1!" NEQ "#" (
      for /f "tokens=1,* delims==" %%A in ("!LINE!") do (
        if /I "%%A"=="MORPH_DB_PATH" set "MORPH_DB_PATH=%%B"
        if /I "%%A"=="MORPH_DB_TABLE_PREFIX" set "MORPH_DB_TABLE_PREFIX=%%B"
      )
    )
  )
)

if not defined MORPH_DB_PATH set "MORPH_DB_PATH=data/morphology.sqlite3"
if not defined MORPH_DB_TABLE_PREFIX set "MORPH_DB_TABLE_PREFIX=morph_"
set "MORPH_DB_PATH=!MORPH_DB_PATH:"=!"
set "MORPH_DB_TABLE_PREFIX=!MORPH_DB_TABLE_PREFIX:"=!"
for %%I in ("!MORPH_DB_PATH!") do set "DB_PATH=%%~fI"

echo.
echo ===============================================
echo  WARNING: FULL MORPHOLOGY DB DATA RESET
echo ===============================================
echo This operation permanently deletes ALL rows in morphology tables:
echo   !DB_PATH!
echo.
echo The DB file itself will NOT be deleted.
echo.
set /p "CONFIRM=Type DELETE to continue: "
if /I not "!CONFIRM!"=="DELETE" (
  echo.
  echo Cancelled. No changes were made.
  exit /b 1
)

if not exist "!DB_PATH!" (
  echo.
  echo Error: DB file not found:
  echo   !DB_PATH!
  exit /b 1
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "!PYTHON_EXE!" set "PYTHON_EXE=python"

set "TMP_SCRIPT=%TEMP%\clear_morphology_db_%RANDOM%_%RANDOM%.py"
> "!TMP_SCRIPT!" echo import re
>> "!TMP_SCRIPT!" echo import sqlite3
>> "!TMP_SCRIPT!" echo import sys
>> "!TMP_SCRIPT!" echo.
>> "!TMP_SCRIPT!" echo db_path = sys.argv[1]
>> "!TMP_SCRIPT!" echo table_prefix = sys.argv[2] if len(sys.argv) ^> 2 else "morph_"
>> "!TMP_SCRIPT!" echo table_prefix = re.sub(r"[^A-Za-z0-9_]", "", table_prefix or "morph_") or "morph_"
>> "!TMP_SCRIPT!" echo.
>> "!TMP_SCRIPT!" echo tables = [
>> "!TMP_SCRIPT!" echo     f"{table_prefix}lexemes",
>> "!TMP_SCRIPT!" echo     f"{table_prefix}token_occurrences",
>> "!TMP_SCRIPT!" echo     f"{table_prefix}expressions",
>> "!TMP_SCRIPT!" echo ]
>> "!TMP_SCRIPT!" echo.
>> "!TMP_SCRIPT!" echo con = sqlite3.connect(db_path)
>> "!TMP_SCRIPT!" echo try:
>> "!TMP_SCRIPT!" echo     cur = con.cursor()
>> "!TMP_SCRIPT!" echo     existing = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
>> "!TMP_SCRIPT!" echo     with con:
>> "!TMP_SCRIPT!" echo         for table_name in tables:
>> "!TMP_SCRIPT!" echo             if table_name in existing:
>> "!TMP_SCRIPT!" echo                 cur.execute(f'DELETE FROM "{table_name}"')
>> "!TMP_SCRIPT!" echo         if "sqlite_sequence" in existing:
>> "!TMP_SCRIPT!" echo             for table_name in tables:
>> "!TMP_SCRIPT!" echo                 cur.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table_name,))
>> "!TMP_SCRIPT!" echo finally:
>> "!TMP_SCRIPT!" echo     con.close()

"!PYTHON_EXE!" "!TMP_SCRIPT!" "!DB_PATH!" "!MORPH_DB_TABLE_PREFIX!"
set "EXIT_CODE=!ERRORLEVEL!"
del /f /q "!TMP_SCRIPT!" >nul 2>&1

if not "!EXIT_CODE!"=="0" (
  echo.
  echo Error: failed to clear DB content. Make sure app is closed and try again.
  exit /b !EXIT_CODE!
)

echo.
echo Done. Morphology DB tables are cleared. DB file is kept.
exit /b 0
