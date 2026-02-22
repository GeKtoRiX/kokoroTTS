# Load After

Date: 2026-02-22  
Command:

```powershell
.\.venv\Scripts\python.exe bench\load\run_morph_load.py --phase after --repeats 3
```

Artifacts:

- `profiles/load/20260222_162112_after_steady.json`
- `profiles/load/20260222_162112_after_stress.json`

## Steady-State Profile

- concurrency: `4`
- requests per run: `240`
- repeats: `3`
- throughput: `17.416 rps` (`stddev 0.462`)
- latency p50: `222.407 ms`
- latency p95: `340.024 ms` (`stddev 92.012`)
- latency p99: `430.025 ms` (`stddev 145.208`)
- error rate: `0.0%`
- RSS peak mean: `210.283 MB`

## Stress Profile

- concurrency: `16`
- requests per run: `960`
- repeats: `3`
- throughput: `4.942 rps` (`stddev 0.052`)
- latency p50: `3281.684 ms`
- latency p95: `3571.003 ms` (`stddev 65.389`)
- latency p99: `3828.343 ms` (`stddev 277.510`)
- error rate: `0.0%`
- RSS peak mean: `314.389 MB`

## Delta vs Baseline

- steady throughput: `+104.76%`
- steady p50: `-50.53%`
- steady p95: `-49.99%`
- steady p99: `-50.78%`
- stress throughput: `+75.75%`
- stress p50: `-41.59%`
- stress p95: `-50.77%`
- stress p99: `-53.55%`
- stress RSS peak: `441.508 MB -> 314.389 MB` (`-28.79%`)
