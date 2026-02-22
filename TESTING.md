# Testing

## Quick run

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Coverage

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=kokoro_tts --cov=app --cov-report=term-missing -q
```

## Security checks

```powershell
.\.venv\Scripts\bandit.exe -q -r app.py kokoro_tts
.\.venv\Scripts\pip-audit.exe -r requirements.txt
```

## Project map regeneration

```powershell
.\.venv\Scripts\python.exe scripts\generate_project_map.py
```

## Run only one area

```powershell
.\.venv\Scripts\python.exe -m pytest tests\app -q
.\.venv\Scripts\python.exe -m pytest tests\ui -q
.\.venv\Scripts\python.exe -m pytest tests\domain -q
.\.venv\Scripts\python.exe -m pytest tests\storage -q
```

## Test layout

- `tests/app` - public facade and entrypoints in `app.py`
- `tests/application` - application services/state/history logic
- `tests/config` - environment/config loading
- `tests/contracts` - public API/CLI contracts
- `tests/domain` - normalization, splitting, voice, morphology domain logic
- `tests/integrations` - ffmpeg/model integration helpers
- `tests/scripts` - smoke tests for project scripts
- `tests/storage` - repositories and writers
- `tests/support` - secondary support module coverage
- `tests/ui` - Tkinter UI behaviors

Notes:
- `tests/conftest.py` provides lightweight stubs for heavy optional dependencies.
- Most tests set `KOKORO_SKIP_APP_INIT=1` to avoid full runtime boot during unit runs.
