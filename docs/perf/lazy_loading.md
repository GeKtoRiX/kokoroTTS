# Lazy Loading

## Changes

1. `MorphologyRepository` default analyzer/extractor are now lazy wrappers:
   - `_default_analyzer()` imports `analyze_english_text` on first use.
   - `_default_expression_extractor()` imports `extract_english_expressions` on first use.
2. `initialize_app_services()` now imports expression extractor only when
   `morph_local_expressions_enabled=True`.

Files:

- `kokoro_tts/storage/morphology_repository.py`
- `kokoro_tts/application/bootstrap.py`

## Why

- Avoid eager imports of morphology/expression stacks on startup paths where feature usage is optional.
- Keep startup dependency loading aligned to actual runtime feature use.

## Startup benchmark

Commands:

```powershell
.\.venv\Scripts\python.exe bench\startup\benchmark_startup.py --phase before --repeats 5
.\.venv\Scripts\python.exe bench\startup\benchmark_startup.py --phase after --repeats 5
```

Artifacts:

- `profiles/startup/20260222_142819_before_startup.json`
- `profiles/startup/20260222_163142_after_startup.json`

Measured impact:

- `app` import mean: `3.492876s -> 3.502503s` (within noise)
- `morph_repo` import mean: `1.076527s -> 1.089166s` (within noise)
- RSS means unchanged within measurement noise.

Conclusion:

- Lazy-loading behavior is now explicit and correctness-safe, but this environment showed no material startup-time gain for these two cold-import probes.
