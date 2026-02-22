# Load Baseline

Date: 2026-02-22  
Command:

```powershell
.\.venv\Scripts\python.exe bench\load\run_morph_load.py --phase before --repeats 3
```

Artifacts:

- `profiles/load/20260222_140936_before_steady.json`
- `profiles/load/20260222_140936_before_stress.json`

## Steady-State Profile

- concurrency: `4`
- requests per run: `240`
- repeats: `3`
- throughput: `8.505 rps` (`stddev 0.273`)
- latency p50: `449.550 ms`
- latency p95: `679.878 ms` (`stddev 64.623`)
- latency p99: `873.678 ms` (`stddev 41.092`)
- error rate: `0.0%`
- RSS peak mean: `223.526 MB`

## Stress Profile

- concurrency: `16`
- requests per run: `960`
- repeats: `3`
- throughput: `2.812 rps` (`stddev 0.034`)
- latency p50: `5618.282 ms`
- latency p95: `7253.849 ms` (`stddev 589.218`)
- latency p99: `8242.538 ms` (`stddev 1159.464`)
- error rate: `0.0%`
- RSS peak mean: `441.508 MB`
