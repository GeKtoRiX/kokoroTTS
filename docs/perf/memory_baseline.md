# Memory Baseline

Date: 2026-02-22  
Command:

```powershell
.\.venv\Scripts\python.exe bench\profile_memory.py --phase before
```

Artifact:

- `profiles/memory/20260222_140930_before_morph_collect.json`
- `profiles/memory/20260222_140930_before_morph_collect_rss_trend.csv`

Workload:

- parts: `10`
- segments/part: `160`
- tokens/segment: `160`
- unique segments: `24`

Result:

- elapsed: `23.532990s`
- peak RSS (sampled): `366.348 MB`
- tracemalloc peak: `69.524 MB`
- token rows: `256000`
- expression rows: `1600`

Top allocators (`tracemalloc`):

1. `kokoro_tts/storage/morphology_repository.py:453` (`45,056,144 bytes`, `512,001` allocs)
2. `json/encoder.py:258` (`13,568,112 bytes`, `256,002` allocs)
3. `kokoro_tts/storage/morphology_repository.py:448` (`11,520,000 bytes`, `256,000` allocs)

Observation:

- Object churn in row construction and JSON encoding dominated Python allocations.

