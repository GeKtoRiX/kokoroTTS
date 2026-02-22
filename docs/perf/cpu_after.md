# CPU After

Date: 2026-02-22  
Command:

```powershell
.\.venv\Scripts\python.exe bench\profile_cpu.py --phase after
```

Artifact:

- `profiles/cpu/20260222_162050_after_morph_collect.prof`
- `profiles/cpu/20260222_162050_after_morph_collect.json`

Workload: identical to baseline.

Result:

- elapsed: `0.443594s`
- token rows: `134400`
- expression rows: `960`

Delta vs baseline:

- CPU elapsed: `1.658760s -> 0.443594s` (`-73.26%`)

Top cumulative CPU hotspots:

1. `kokoro_tts/storage/morphology_repository.py::_collect_ingest_rows` (`0.443516s`)
2. `kokoro_tts/storage/morphology_repository.py::_append_token_rows` (`0.437859s`)
3. `MorphRow.__init__` dataclass construction (`0.334966s`)
4. `kokoro_tts/storage/morphology_repository.py::_append_expression_rows` (`0.003434s`)
5. `kokoro_tts/storage/morphology_repository.py::_segment_templates` (`0.001244s`)

Observation:

- JSON serialization and repeated token normalization dropped out of the top hotspots due segment-template caching.
