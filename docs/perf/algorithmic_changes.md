# Algorithmic Changes

## 1) Segment template normalization cache

File: `kokoro_tts/storage/morphology_repository.py`

Change:

- Added `_segment_templates_cache` with pre-normalized token/expression templates.
- Added `_collect_segment_token_templates()` and `_collect_segment_expression_templates()`.
- `_collect_ingest_rows()` now reuses cached normalized templates per segment text.

Why:

- Baseline showed repeated `json.dumps` and token-field normalization for identical segments.

Complexity impact:

- Before: `O(S * T * normalize)` for `S` segments and `T` tokens per segment.
- After: `O(U * T * normalize) + O(S * T * row_instantiate)` where `U` is unique segment texts.
- For repeated segments (`U << S`), normalization cost is significantly reduced.

## 2) Ingest preparation made single-pass

File: `kokoro_tts/storage/morphology_repository.py`

Change:

- In `ingest_dialogue_parts()`, lexeme dedup and occurrence tuple construction now happen in one loop over rows.

Why:

- Removed duplicate passes and temporary object churn in hot path.

## 3) Lower-overhead row objects

File: `kokoro_tts/storage/morphology_repository.py`

Change:

- `MorphRow` and `ExpressionRow` changed to `@dataclass(..., slots=True)`.

Why:

- Reduces per-row Python object footprint and attribute dictionary overhead.

## Microbenchmark

Command:

```powershell
.\.venv\Scripts\python.exe bench\micro\benchmark_morph_collect.py --phase after --include-cache-disabled 1
```

Artifact:

- `profiles/micro/20260222_163226_after_morph_collect_micro.json`

Result:

- `cache_hit` mean: `39.760 ms`
- `cache_miss` mean: `42.339 ms`
- `cache_disabled` mean: `121.011 ms`
- cache speedup (`cache_disabled / cache_hit`): `3.044x`
