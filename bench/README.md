# Performance Bench Harness

Deterministic harnesses for CPU, memory, load, and startup measurements.

## CPU profile

```powershell
.\.venv\Scripts\python.exe bench\profile_cpu.py --phase before
.\.venv\Scripts\python.exe bench\profile_cpu.py --phase after
```

## Memory profile

```powershell
.\.venv\Scripts\python.exe bench\profile_memory.py --phase before
.\.venv\Scripts\python.exe bench\profile_memory.py --phase after
```

## Load benchmark

```powershell
.\.venv\Scripts\python.exe bench\load\run_morph_load.py --phase before --repeats 3
.\.venv\Scripts\python.exe bench\load\run_morph_load.py --phase after --repeats 3
```

## Startup benchmark

```powershell
.\.venv\Scripts\python.exe bench\startup\benchmark_startup.py --phase before --repeats 5
.\.venv\Scripts\python.exe bench\startup\benchmark_startup.py --phase after --repeats 5
```

## Microbenchmark + guard

```powershell
.\.venv\Scripts\python.exe bench\micro\benchmark_morph_collect.py --phase after --include-cache-disabled 1
.\.venv\Scripts\python.exe bench\perf_guard.py --result-json profiles\micro\<artifact>.json
```
