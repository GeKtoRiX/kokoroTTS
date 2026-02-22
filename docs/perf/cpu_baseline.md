# CPU Baseline

Date: 2026-02-22  
Command:

```powershell
.\.venv\Scripts\python.exe bench\profile_cpu.py --phase before
```

Artifact:

- `profiles/cpu/20260222_140900_before_morph_collect.prof`
- `profiles/cpu/20260222_140900_before_morph_collect.json`

Workload:

- parts: `8`
- segments/part: `120`
- tokens/segment: `140`
- unique segments: `24`
- segment cache size: `1024`

Result:

- elapsed: `1.658760s`
- token rows: `134400`
- expression rows: `960`

Top cumulative CPU hotspots:

1. `kokoro_tts/storage/morphology_repository.py::_collect_ingest_rows` (`1.658709s`)
2. `kokoro_tts/storage/morphology_repository.py::_collect_segment_tokens` (`1.638579s`)
3. `kokoro_tts/storage/morphology_repository.py::_serialize_feats_json` (`0.596582s`)
4. `json.dumps` (`0.528076s`)
5. `json.encoder.encode` (`0.384522s`)

Observation:

- Dominant time was spent repeatedly normalizing and serializing token metadata for each segment, especially `feats_json` JSON encoding.

