from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap


TARGETS: dict[str, str] = {
    "app": "import app",
    "bootstrap": "from kokoro_tts.application import bootstrap as _bootstrap",
    "morph_repo": "from kokoro_tts.storage import morphology_repository as _morphology_repository",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cold-process startup benchmark for import-time cost.",
    )
    parser.add_argument("--phase", choices=("before", "after"), default="before")
    parser.add_argument("--targets", default="app,morph_repo")
    parser.add_argument("--repeats", type=int, default=5)
    return parser


def _run_once(target_code: str) -> dict[str, float]:
    code = textwrap.dedent(
        f"""
        import os
        import time
        import psutil

        os.environ["KOKORO_SKIP_APP_INIT"] = "1"
        start = time.perf_counter()
        {target_code}
        elapsed = time.perf_counter() - start
        rss = psutil.Process().memory_info().rss
        print(f"elapsed={{elapsed:.9f}}")
        print(f"rss={{rss}}")
        """
    ).strip()
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    values: dict[str, float] = {"elapsed_s": 0.0, "rss_bytes": 0.0}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "elapsed":
            values["elapsed_s"] = float(value)
        elif key == "rss":
            values["rss_bytes"] = float(value)
    return values


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    selected_targets = [
        item.strip().lower() for item in str(args.targets or "").split(",") if item.strip()
    ]
    if not selected_targets:
        selected_targets = ["app", "morph_repo"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("profiles") / "startup"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    for target_name in selected_targets:
        if target_name not in TARGETS:
            raise ValueError(f"Unsupported startup target '{target_name}'.")
        runs = [_run_once(TARGETS[target_name]) for _ in range(max(1, int(args.repeats)))]
        elapsed_values = [float(run["elapsed_s"]) for run in runs]
        rss_values = [float(run["rss_bytes"]) for run in runs]
        target_payload = {
            "target": target_name,
            "runs": runs,
            "elapsed_s": {
                "mean": statistics.mean(elapsed_values),
                "stddev": _stddev(elapsed_values),
                "min": min(elapsed_values),
                "max": max(elapsed_values),
            },
            "rss_mb": {
                "mean": statistics.mean(rss_values) / (1024.0 * 1024.0),
                "stddev": _stddev(rss_values) / (1024.0 * 1024.0),
                "min": min(rss_values) / (1024.0 * 1024.0),
                "max": max(rss_values) / (1024.0 * 1024.0),
            },
        }
        results[target_name] = target_payload
        print(f"target={target_name}")
        print(f"elapsed_mean_s={target_payload['elapsed_s']['mean']:.6f}")
        print(f"rss_mean_mb={target_payload['rss_mb']['mean']:.3f}")

    payload = {
        "phase": args.phase,
        "timestamp": timestamp,
        "python_executable": sys.executable,
        "repeats": int(args.repeats),
        "results": results,
    }
    output_path = output_dir / f"{timestamp}_{args.phase}_startup.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

