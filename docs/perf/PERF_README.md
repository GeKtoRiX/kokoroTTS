# Perf Reproduction Guide

Run from repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe bench\profile_cpu.py --phase before
.\.venv\Scripts\python.exe bench\profile_memory.py --phase before
.\.venv\Scripts\python.exe bench\load\run_morph_load.py --phase before --repeats 3
.\.venv\Scripts\python.exe bench\startup\benchmark_startup.py --phase before --repeats 5
.\.venv\Scripts\python.exe bench\micro\benchmark_morph_collect.py --phase before --include-cache-disabled 1
```

After optimizations:

```powershell
.\.venv\Scripts\python.exe bench\profile_cpu.py --phase after
.\.venv\Scripts\python.exe bench\profile_memory.py --phase after
.\.venv\Scripts\python.exe bench\load\run_morph_load.py --phase after --repeats 3
.\.venv\Scripts\python.exe bench\startup\benchmark_startup.py --phase after --repeats 5
.\.venv\Scripts\python.exe bench\micro\benchmark_morph_collect.py --phase after --include-cache-disabled 1
```

Artifacts:

- CPU: `profiles/cpu/`
- Memory: `profiles/memory/`
- Load: `profiles/load/`
- Startup: `profiles/startup/`
- Microbench: `profiles/micro/`

Optional perf guard check:

```powershell
.\.venv\Scripts\python.exe bench\perf_guard.py --result-json profiles\micro\<artifact>.json
```
