# Performance Summary

## Static Footprint

- Python files scanned: `82`
- Total Python LOC: `20822`
- Largest files by LOC:
  - `kokoro_tts/ui/tkinter_app.py`: `2541`
  - `tests/ui/test_tkinter_ui_interactions.py`: `1842`
  - `kokoro_tts/ui/features/audio_player_feature.py`: `1324`
  - `kokoro_tts/storage/morphology_repository.py`: `955`
  - `kokoro_tts/application/state.py`: `851`
  - `scripts/generate_project_map.py`: `830`
  - `kokoro_tts/domain/expressions.py`: `793`
  - `app.py`: `782`
  - `tests/storage/test_morphology_repository.py`: `659`
  - `kokoro_tts/domain/morphology.py`: `622`

## Synthetic Morphology Ingest Benchmark

- Input: `5 parts x 120 segments x 200 tokens`
- Token rows produced: `120000`
- Expression rows produced: `0`
- Elapsed seconds: `0.316688`
- Approx token rows/sec: `378,922`

## Operational Notes

- Main runtime hotspots are the UI layer and NLP-heavy domain modules.
- Use `scripts/profile_tts_inference.py` for device-level first/warm generation latency.
- Use `scripts/benchmark_morph_ingest.py` for regression checks on ingest throughput.
