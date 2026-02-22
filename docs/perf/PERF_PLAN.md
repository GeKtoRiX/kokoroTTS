# Performance Plan

Date: 2026-02-22  
Scope: deterministic performance pass focused on morphology ingest + startup path.
Status: completed (see `docs/perf/REPORT.md`)

## Why this path

- `MorphologyRepository._collect_ingest_rows` is CPU/memory intensive under large ingest payloads.
- `KokoroState._persist_morphology` can block generation path when async queue is saturated.
- Import-time path currently loads morphology/expression modules at startup even when optional features are off.

## Baseline measurements to capture

1. CPU profile (`bench/profile_cpu.py`)
   - workload: deterministic synthetic parts/segments/tokens
   - output: top cumulative CPU functions and raw `.prof`
2. Memory profile (`bench/profile_memory.py`)
   - workload: same deterministic ingest workload
   - output: sampled peak RSS + tracemalloc peak + top allocators
3. Load benchmark (`bench/load/run_morph_load.py`)
   - profiles:
     - steady-state (`concurrency=4`)
     - stress (`concurrency=16`)
   - output: throughput, error rate, p50/p95/p99, CPU/RSS trend samples
4. Startup benchmark (`bench/startup/benchmark_startup.py`)
   - targets: `app`, `morph_repo`
   - output: cold-process import latency + RSS

## Hypotheses

1. `MorphologyRepository` spends significant time allocating dataclass rows and repeatedly normalizing token/expression fields.
2. Memory peaks are amplified by transient object copies in ingest row assembly and conversion.
3. Under concurrent load, queue contention and repeated per-segment normalization inflate p95/p99 latency.
4. Startup can improve by lazy-loading heavy default analyzer/extractor modules.

## Planned changes

1. Algorithmic optimization
   - reduce duplicate work in row collection
   - reduce copies/conversions on hot path
2. Caching
   - cache normalized segment templates for repeated segment text
3. Async heavy path
   - tighten async enqueue behavior in `KokoroState` to reduce lock-held overhead and blocking
4. Lazy loading
   - defer default morphology/expression analyzer imports until first use

## Verification

- Run perf harnesses `before` and `after` with >=3 repeats for load and startup.
- Run full test suite and smoke checks after changes.
- Publish artifacts and reproducible commands under `docs/perf` and `profiles/`.
