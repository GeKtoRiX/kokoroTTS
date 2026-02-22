# Performance Report

Date: 2026-02-22

## Executive Summary

Critical flow: morphology ingest row collection and concurrent load around that path.

- CPU elapsed: `1.658760s -> 0.443594s` (`-73.26%`)
- Memory-profile elapsed: `23.532990s -> 6.384234s` (`-72.87%`)
- Peak RSS: `366.348 MB -> 257.883 MB` (`-29.61%`)
- Tracemalloc peak: `69.524 MB -> 35.445 MB` (`-49.02%`)
- Steady load throughput: `8.505 -> 17.416 rps` (`+104.76%`)
- Steady p50: `449.550 -> 222.407 ms` (`-50.53%`)
- Steady p95: `679.878 -> 340.024 ms` (`-49.99%`)
- Steady p99: `873.678 -> 430.025 ms` (`-50.78%`)
- Stress load throughput: `2.812 -> 4.942 rps` (`+75.75%`)
- Stress p50: `5618.282 -> 3281.684 ms` (`-41.59%`)
- Stress p95: `7253.849 -> 3571.003 ms` (`-50.77%`)
- Stress p99: `8242.538 -> 3828.343 ms` (`-53.55%`)
- Error rate remained `0.0%` in all benchmark profiles.
- Startup lazy-load probes were neutral (within noise) in this environment.

## Artifacts

- Environment snapshot: `docs/perf/ENVIRONMENT.md`
- Dependency lock: `requirements-lock.txt`
- CPU before/after:
  - `profiles/cpu/20260222_140900_before_morph_collect.json`
  - `profiles/cpu/20260222_162050_after_morph_collect.json`
- Memory before/after:
  - `profiles/memory/20260222_140930_before_morph_collect.json`
  - `profiles/memory/20260222_162103_after_morph_collect.json`
- Load before/after:
  - `profiles/load/20260222_140936_before_steady.json`
  - `profiles/load/20260222_140936_before_stress.json`
  - `profiles/load/20260222_162112_after_steady.json`
  - `profiles/load/20260222_162112_after_stress.json`
- Startup probes:
  - `profiles/startup/20260222_142819_before_startup.json`
  - `profiles/startup/20260222_163142_after_startup.json`
- Microbenchmark:
  - `profiles/micro/20260222_163226_after_morph_collect_micro.json`

## Validation

- Full tests: `188 passed`
- Smoke: `scripts/smoke_full.ps1` returned `SMOKE_OK`

## Reproduction

See `docs/perf/PERF_README.md` for exact commands.

## Risk / Follow-up

1. Add optional guardrail workflow execution in CI via manual dispatch (`.github/workflows/perf-guard.yml`).
2. Extend current synthetic perf suite with a second real-model scenario (`scripts/profile_tts_inference.py`) for broader end-to-end inference coverage.
