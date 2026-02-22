# Caching Changes

## Cache targets

1. Segment normalization outputs in morphology ingest hot path.
2. Existing analyzer/extractor result caches retained for compatibility.

File:

- `kokoro_tts/storage/morphology_repository.py`

## Implementation

- Added `_segment_templates_cache` (bounded `OrderedDict` LRU-style via `_cache_store`).
- Cache key: raw `segment_text`.
- Cache value:
  - `segment_hash`
  - token template tuple
  - expression template tuple

This caches expensive, pure normalization work and avoids repeated JSON serialization.

## Invalidation and bounds

- Max entries controlled by existing `segment_cache_size` constructor argument.
- LRU eviction via `OrderedDict.popitem(last=False)` in `_cache_store`.
- Setting cache size to `0` disables caching.

## Correctness

- Output row schema is unchanged.
- Existing cache-related tests still pass.
- Error behavior unchanged (ingest remains best-effort with logging).

## Microbenchmark signal

From `profiles/micro/20260222_163226_after_morph_collect_micro.json`:

- cache enabled/hit mean: `39.760 ms`
- cache disabled mean: `121.011 ms`
- speedup: `3.044x`
