from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time


BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))


from workloads.morph_workload import MorphWorkloadConfig, build_parts, build_repository


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Microbenchmark morphology row collection (cache-hit vs cache-miss scenarios).",
    )
    parser.add_argument("--phase", choices=("before", "after", "ci"), default="after")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--parts", type=int, default=4)
    parser.add_argument("--segments-per-part", type=int, default=40)
    parser.add_argument("--tokens-per-segment", type=int, default=120)
    parser.add_argument("--segment-cache-size", type=int, default=1024)
    parser.add_argument("--include-cache-disabled", type=int, default=1)
    return parser


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * q))
    return float(sorted_values[index])


def _run_case(
    *,
    name: str,
    config: MorphWorkloadConfig,
    runs: int,
    warmups: int,
    segment_cache_size: int,
) -> dict[str, object]:
    repo = build_repository(segment_cache_size=segment_cache_size)
    parts = build_parts(config)

    for _ in range(max(0, int(warmups))):
        repo._collect_ingest_rows(parts, source=f"micro_{name}_warmup")

    elapsed_ms: list[float] = []
    token_rows_count = 0
    expression_rows_count = 0
    for _ in range(max(1, int(runs))):
        started_at = time.perf_counter()
        token_rows, expression_rows = repo._collect_ingest_rows(parts, source=f"micro_{name}")
        elapsed_ms.append((time.perf_counter() - started_at) * 1000.0)
        token_rows_count = len(token_rows)
        expression_rows_count = len(expression_rows)

    return {
        "case": name,
        "workload": {
            "parts": config.parts,
            "segments_per_part": config.segments_per_part,
            "tokens_per_segment": config.tokens_per_segment,
            "unique_segments": config.unique_segments,
        },
        "runs": len(elapsed_ms),
        "latency_ms": {
            "mean": statistics.mean(elapsed_ms),
            "stddev": statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
            "p50": _quantile(elapsed_ms, 0.50),
            "p95": _quantile(elapsed_ms, 0.95),
            "min": min(elapsed_ms),
            "max": max(elapsed_ms),
        },
        "token_rows": token_rows_count,
        "expression_rows": expression_rows_count,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repeated_case = MorphWorkloadConfig(
        parts=args.parts,
        segments_per_part=args.segments_per_part,
        tokens_per_segment=args.tokens_per_segment,
        unique_segments=1,
    )
    unique_case = MorphWorkloadConfig(
        parts=args.parts,
        segments_per_part=args.segments_per_part,
        tokens_per_segment=args.tokens_per_segment,
        unique_segments=max(1, args.parts * args.segments_per_part),
    )
    repeated = _run_case(
        name="cache_hit",
        config=repeated_case,
        runs=args.runs,
        warmups=args.warmups,
        segment_cache_size=args.segment_cache_size,
    )
    unique = _run_case(
        name="cache_miss",
        config=unique_case,
        runs=args.runs,
        warmups=args.warmups,
        segment_cache_size=args.segment_cache_size,
    )
    disabled = None
    if bool(args.include_cache_disabled):
        disabled = _run_case(
            name="cache_disabled",
            config=repeated_case,
            runs=args.runs,
            warmups=args.warmups,
            segment_cache_size=0,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("profiles") / "micro"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_{args.phase}_morph_collect_micro.json"
    payload = {
        "phase": args.phase,
        "timestamp": timestamp,
        "segment_cache_size": args.segment_cache_size,
        "cases": [case for case in (repeated, unique, disabled) if case is not None],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"summary={output_path.as_posix()}")
    print(f"cache_hit_mean_ms={repeated['latency_ms']['mean']:.3f}")
    print(f"cache_miss_mean_ms={unique['latency_ms']['mean']:.3f}")
    if disabled is not None:
        print(f"cache_disabled_mean_ms={disabled['latency_ms']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
