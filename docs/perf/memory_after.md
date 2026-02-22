# Memory After

Date: 2026-02-22  
Command:

```powershell
.\.venv\Scripts\python.exe bench\profile_memory.py --phase after
```

Artifact:

- `profiles/memory/20260222_162103_after_morph_collect.json`
- `profiles/memory/20260222_162103_after_morph_collect_rss_trend.csv`

Workload: identical to baseline.

Result:

- elapsed: `6.384234s`
- peak RSS (sampled): `257.883 MB`
- tracemalloc peak: `35.445 MB`
- token rows: `256000`
- expression rows: `1600`

Delta vs baseline:

- memory profile elapsed: `-72.87%`
- peak RSS: `366.348 MB -> 257.883 MB` (`-29.61%`)
- traced peak allocation: `69.524 MB -> 35.445 MB` (`-49.02%`)

Top allocators (`tracemalloc`):

1. `kokoro_tts/storage/morphology_repository.py:689` (`34,816,000 bytes`, `256,000` allocs)
2. `kokoro_tts/storage/morphology_repository.py:688` (`2,055,456 bytes`, `1` alloc)
3. `kokoro_tts/storage/morphology_repository.py:720` (`230,552 bytes`, `1,601` allocs)

Observation:

- High-volume `json.dumps` allocation pressure from baseline is removed from the top allocation set.
