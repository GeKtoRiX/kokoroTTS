from __future__ import annotations

import argparse
import cProfile
from datetime import datetime
import io
import json
from pathlib import Path
import pstats
import sys
import time


BENCH_ROOT = Path(__file__).resolve().parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))


from workloads.morph_workload import MorphWorkloadConfig, build_parts, build_repository


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic CPU profile for morphology ingest row collection.",
    )
    parser.add_argument("--phase", choices=("before", "after"), default="before")
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--segments-per-part", type=int, default=120)
    parser.add_argument("--tokens-per-segment", type=int, default=140)
    parser.add_argument("--unique-segments", type=int, default=24)
    parser.add_argument("--segment-cache-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--top", type=int, default=25)
    return parser


def _top_functions(stats: pstats.Stats, top: int) -> list[dict[str, object]]:
    ranked = sorted(
        (
            {
                "file": file_name,
                "line": line_no,
                "function": function_name,
                "calls": int(call_count),
                "primitive_calls": int(primitive_calls),
                "total_time_s": float(total_time),
                "cumulative_time_s": float(cumulative_time),
            }
            for (file_name, line_no, function_name), (
                call_count,
                primitive_calls,
                total_time,
                cumulative_time,
                _,
            ) in stats.stats.items()
        ),
        key=lambda entry: entry["cumulative_time_s"],
        reverse=True,
    )
    return ranked[: max(1, int(top))]


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
        repo._collect_ingest_rows(parts, source="cpu_profile_warmup")

    profiler = cProfile.Profile()
    started_at = time.perf_counter()
    profiler.enable()
    token_rows, expression_rows = repo._collect_ingest_rows(parts, source="cpu_profile")
    profiler.disable()
    elapsed_s = time.perf_counter() - started_at

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("profiles") / "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"{timestamp}_{args.phase}_morph_collect"

    profile_path = output_dir / f"{file_stem}.prof"
    text_summary_path = output_dir / f"{file_stem}.txt"
    json_summary_path = output_dir / f"{file_stem}.json"

    profiler.dump_stats(profile_path.as_posix())
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    output_buffer = io.StringIO()
    stats.stream = output_buffer
    stats.print_stats(max(1, int(args.top)))
    top_functions = _top_functions(stats, args.top)

    summary_payload = {
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
        },
        "top_functions": top_functions,
        "profile_path": profile_path.as_posix(),
    }

    text_summary_path.write_text(output_buffer.getvalue(), encoding="utf-8")
    json_summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"phase={args.phase}")
    print(f"profile={profile_path.as_posix()}")
    print(f"summary={json_summary_path.as_posix()}")
    print(f"elapsed_seconds={elapsed_s:.6f}")
    print(f"token_rows={len(token_rows)}")
    print(f"expression_rows={len(expression_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

