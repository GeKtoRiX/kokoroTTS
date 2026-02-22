from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
import tracemalloc

import psutil


BENCH_ROOT = Path(__file__).resolve().parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))


from workloads.morph_workload import MorphWorkloadConfig, build_parts, build_repository


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic memory profile for morphology ingest row collection.",
    )
    parser.add_argument("--phase", choices=("before", "after"), default="before")
    parser.add_argument("--parts", type=int, default=10)
    parser.add_argument("--segments-per-part", type=int, default=160)
    parser.add_argument("--tokens-per-segment", type=int, default=160)
    parser.add_argument("--unique-segments", type=int, default=24)
    parser.add_argument("--segment-cache-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--sample-ms", type=float, default=5.0)
    parser.add_argument("--tracemalloc-frames", type=int, default=25)
    return parser


class _RssSampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.001, float(interval_seconds))
        self._stop = threading.Event()
        self.samples: list[tuple[float, int]] = []
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._process = psutil.Process()
        self._started_at = 0.0

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            rss = self._process.memory_info().rss
            self.samples.append((now - self._started_at, rss))
            time.sleep(self.interval_seconds)


def _snapshot_top_allocations(
    snapshot: tracemalloc.Snapshot,
    limit: int,
) -> list[dict[str, object]]:
    top_stats = snapshot.statistics("lineno")
    out: list[dict[str, object]] = []
    for stat in top_stats[: max(1, int(limit))]:
        frame = stat.traceback[0]
        out.append(
            {
                "file": frame.filename,
                "line": int(frame.lineno),
                "size_bytes": int(stat.size),
                "count": int(stat.count),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = MorphWorkloadConfig(
        parts=args.parts,
        segments_per_part=args.segments_per_part,
        tokens_per_segment=args.tokens_per_segment,
        unique_segments=args.unique_segments,
    )
    repo = build_repository(segment_cache_size=args.segment_cache_size)
    parts = build_parts(config)

    for _ in range(max(0, int(args.warmups))):
        repo._collect_ingest_rows(parts, source="memory_profile_warmup")

    process = psutil.Process()
    rss_before = process.memory_info().rss
    sampler = _RssSampler(interval_seconds=float(args.sample_ms) / 1000.0)
    tracemalloc.start(max(1, int(args.tracemalloc_frames)))
    sampler.start()

    started_at = time.perf_counter()
    token_rows, expression_rows = repo._collect_ingest_rows(parts, source="memory_profile")
    elapsed_s = time.perf_counter() - started_at

    sampler.stop()
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    rss_after = process.memory_info().rss

    max_sampled_rss = max((sample[1] for sample in sampler.samples), default=rss_after)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("profiles") / "memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"{timestamp}_{args.phase}_morph_collect"
    json_path = output_dir / f"{file_stem}.json"
    trend_path = output_dir / f"{file_stem}_rss_trend.csv"

    trend_lines = ["elapsed_s,rss_bytes"]
    trend_lines.extend(f"{elapsed:.6f},{rss}" for elapsed, rss in sampler.samples)
    trend_path.write_text("\n".join(trend_lines) + "\n", encoding="utf-8")

    payload = {
        "phase": args.phase,
        "timestamp": timestamp,
        "workload": {
            "parts": config.parts,
            "segments_per_part": config.segments_per_part,
            "tokens_per_segment": config.tokens_per_segment,
            "unique_segments": config.unique_segments,
            "segment_cache_size": args.segment_cache_size,
        },
        "result": {
            "elapsed_seconds": elapsed_s,
            "token_rows": len(token_rows),
            "expression_rows": len(expression_rows),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_peak_sampled_bytes": max_sampled_rss,
            "rss_peak_sampled_mb": max_sampled_rss / (1024.0 * 1024.0),
            "tracemalloc_current_bytes": traced_current,
            "tracemalloc_peak_bytes": traced_peak,
            "tracemalloc_peak_mb": traced_peak / (1024.0 * 1024.0),
        },
        "top_allocations": _snapshot_top_allocations(snapshot, args.top),
        "rss_trend_path": trend_path.as_posix(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"phase={args.phase}")
    print(f"summary={json_path.as_posix()}")
    print(f"elapsed_seconds={elapsed_s:.6f}")
    print(f"rss_peak_sampled_mb={payload['result']['rss_peak_sampled_mb']:.3f}")
    print(f"tracemalloc_peak_mb={payload['result']['tracemalloc_peak_mb']:.3f}")
    print(f"token_rows={len(token_rows)}")
    print(f"expression_rows={len(expression_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

