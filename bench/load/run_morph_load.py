from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time

import psutil


BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))


from workloads.morph_workload import MorphWorkloadConfig, build_parts, build_repository


@dataclass(frozen=True)
class LoadProfileConfig:
    name: str
    concurrency: int
    requests: int
    workload: MorphWorkloadConfig


LOAD_PROFILES: dict[str, LoadProfileConfig] = {
    "steady": LoadProfileConfig(
        name="steady",
        concurrency=4,
        requests=240,
        workload=MorphWorkloadConfig(
            parts=4,
            segments_per_part=60,
            tokens_per_segment=100,
            unique_segments=20,
        ),
    ),
    "stress": LoadProfileConfig(
        name="stress",
        concurrency=16,
        requests=960,
        workload=MorphWorkloadConfig(
            parts=6,
            segments_per_part=90,
            tokens_per_segment=120,
            unique_segments=30,
        ),
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concurrent deterministic load benchmark for morphology ingest row collection.",
    )
    parser.add_argument("--phase", choices=("before", "after"), default="before")
    parser.add_argument("--profiles", default="steady,stress")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--segment-cache-size", type=int, default=1024)
    parser.add_argument("--sample-ms", type=float, default=100.0)
    parser.add_argument("--payload-count", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=3)
    return parser


class _ResourceSampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.01, float(interval_seconds))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="load-resource-sampler", daemon=True)
        self._process = psutil.Process()
        self.samples: list[dict[str, float]] = []
        self._started_at = 0.0

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._process.cpu_percent(None)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            rss_bytes = self._process.memory_info().rss
            cpu_percent = self._process.cpu_percent(None)
            self.samples.append(
                {
                    "elapsed_s": now - self._started_at,
                    "rss_bytes": float(rss_bytes),
                    "rss_mb": float(rss_bytes / (1024.0 * 1024.0)),
                    "cpu_percent": float(cpu_percent),
                }
            )
            time.sleep(self.interval_seconds)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = int(math.ceil((percentile / 100.0) * len(sorted_values))) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return float(sorted_values[rank])


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _build_payloads(config: MorphWorkloadConfig, payload_count: int) -> list[list[list[tuple[str, str]]]]:
    payload_count = max(1, int(payload_count))
    return [build_parts(config, offset=index * config.segments_per_part) for index in range(payload_count)]


def _run_single_request(repo, payload, source: str) -> tuple[float, int, int, str | None]:
    started_at = time.perf_counter()
    try:
        token_rows, expression_rows = repo._collect_ingest_rows(payload, source=source)
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        return elapsed, 0, 0, str(exc)
    elapsed = time.perf_counter() - started_at
    return elapsed, len(token_rows), len(expression_rows), None


def _run_profile_once(
    profile: LoadProfileConfig,
    *,
    segment_cache_size: int,
    sample_seconds: float,
    payload_count: int,
    warmups: int,
) -> dict[str, object]:
    repo = build_repository(segment_cache_size=segment_cache_size)
    payloads = _build_payloads(profile.workload, payload_count)
    for warmup_index in range(max(0, int(warmups))):
        repo._collect_ingest_rows(payloads[warmup_index % len(payloads)], source="load_warmup")

    latencies_ms: list[float] = []
    token_total = 0
    expression_total = 0
    errors = 0
    error_messages: list[str] = []

    sampler = _ResourceSampler(sample_seconds)
    sampler.start()
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=profile.concurrency, thread_name_prefix="perf-load") as pool:
        futures = [
            pool.submit(
                _run_single_request,
                repo,
                payloads[index % len(payloads)],
                "load_benchmark",
            )
            for index in range(profile.requests)
        ]
        for future in as_completed(futures):
            elapsed_s, token_rows, expression_rows, error = future.result()
            latencies_ms.append(elapsed_s * 1000.0)
            token_total += int(token_rows)
            expression_total += int(expression_rows)
            if error is not None:
                errors += 1
                if len(error_messages) < 5:
                    error_messages.append(error)
    elapsed_total_s = time.perf_counter() - started_at
    sampler.stop()

    success_count = max(0, profile.requests - errors)
    throughput_rps = 0.0
    if elapsed_total_s > 0.0:
        throughput_rps = float(success_count / elapsed_total_s)
    sample_cpu_peak = max((sample["cpu_percent"] for sample in sampler.samples), default=0.0)
    sample_rss_peak_mb = max((sample["rss_mb"] for sample in sampler.samples), default=0.0)

    return {
        "requests": profile.requests,
        "concurrency": profile.concurrency,
        "successes": success_count,
        "errors": errors,
        "error_rate_percent": (float(errors) / float(profile.requests) * 100.0)
        if profile.requests
        else 0.0,
        "elapsed_total_s": elapsed_total_s,
        "throughput_rps": throughput_rps,
        "latency_ms": {
            "p50": _percentile(latencies_ms, 50.0),
            "p95": _percentile(latencies_ms, 95.0),
            "p99": _percentile(latencies_ms, 99.0),
            "max": max(latencies_ms) if latencies_ms else 0.0,
            "mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        },
        "token_rows_total": token_total,
        "expression_rows_total": expression_total,
        "resource": {
            "cpu_peak_percent": sample_cpu_peak,
            "rss_peak_mb": sample_rss_peak_mb,
            "samples": sampler.samples,
        },
        "errors_sample": error_messages,
    }


def _aggregate_runs(profile: LoadProfileConfig, runs: list[dict[str, object]]) -> dict[str, object]:
    throughput = [float(run["throughput_rps"]) for run in runs]
    p50 = [float(run["latency_ms"]["p50"]) for run in runs]
    p95 = [float(run["latency_ms"]["p95"]) for run in runs]
    p99 = [float(run["latency_ms"]["p99"]) for run in runs]
    error_rates = [float(run["error_rate_percent"]) for run in runs]
    cpu_peaks = [float(run["resource"]["cpu_peak_percent"]) for run in runs]
    rss_peaks = [float(run["resource"]["rss_peak_mb"]) for run in runs]
    return {
        "profile": profile.name,
        "requests": profile.requests,
        "concurrency": profile.concurrency,
        "runs": len(runs),
        "workload": {
            "parts": profile.workload.parts,
            "segments_per_part": profile.workload.segments_per_part,
            "tokens_per_segment": profile.workload.tokens_per_segment,
            "unique_segments": profile.workload.unique_segments,
        },
        "throughput_rps": {
            "mean": statistics.mean(throughput),
            "stddev": _stddev(throughput),
        },
        "latency_ms": {
            "p50_mean": statistics.mean(p50),
            "p50_stddev": _stddev(p50),
            "p95_mean": statistics.mean(p95),
            "p95_stddev": _stddev(p95),
            "p99_mean": statistics.mean(p99),
            "p99_stddev": _stddev(p99),
        },
        "error_rate_percent": {
            "mean": statistics.mean(error_rates),
            "stddev": _stddev(error_rates),
        },
        "resource": {
            "cpu_peak_percent_mean": statistics.mean(cpu_peaks),
            "cpu_peak_percent_stddev": _stddev(cpu_peaks),
            "rss_peak_mb_mean": statistics.mean(rss_peaks),
            "rss_peak_mb_stddev": _stddev(rss_peaks),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    selected_profiles = [
        profile.strip().lower() for profile in str(args.profiles or "").split(",") if profile.strip()
    ]
    if not selected_profiles:
        selected_profiles = ["steady", "stress"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("profiles") / "load"
    output_dir.mkdir(parents=True, exist_ok=True)

    for profile_name in selected_profiles:
        if profile_name not in LOAD_PROFILES:
            raise ValueError(f"Unsupported profile '{profile_name}'.")
        profile = LOAD_PROFILES[profile_name]
        runs: list[dict[str, object]] = []
        for _ in range(max(1, int(args.repeats))):
            runs.append(
                _run_profile_once(
                    profile,
                    segment_cache_size=args.segment_cache_size,
                    sample_seconds=float(args.sample_ms) / 1000.0,
                    payload_count=args.payload_count,
                    warmups=args.warmups,
                )
            )
        aggregate = _aggregate_runs(profile, runs)
        payload = {
            "phase": args.phase,
            "timestamp": timestamp,
            "profile": profile_name,
            "aggregate": aggregate,
            "runs": runs,
        }
        output_path = output_dir / f"{timestamp}_{args.phase}_{profile_name}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"profile={profile_name}")
        print(f"summary={output_path.as_posix()}")
        print(f"throughput_rps_mean={aggregate['throughput_rps']['mean']:.3f}")
        print(f"latency_p95_mean_ms={aggregate['latency_ms']['p95_mean']:.3f}")
        print(f"latency_p99_mean_ms={aggregate['latency_ms']['p99_mean']:.3f}")
        print(f"error_rate_mean_percent={aggregate['error_rate_percent']['mean']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

