from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate microbenchmark thresholds for perf-regression guardrails.",
    )
    parser.add_argument("--result-json", required=True, help="Path to microbenchmark JSON output.")
    parser.add_argument("--max-cache-hit-ms", type=float, default=120.0)
    parser.add_argument("--min-cache-speedup", type=float, default=1.5)
    return parser


def _case_lookup(payload: dict[str, object], name: str) -> dict[str, object] | None:
    for item in payload.get("cases", []):
        if isinstance(item, dict) and str(item.get("case")) == name:
            return item
    return None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    cache_hit = _case_lookup(payload, "cache_hit")
    cache_disabled = _case_lookup(payload, "cache_disabled")
    if cache_hit is None:
        raise ValueError("Missing 'cache_hit' case in result payload.")

    cache_hit_mean = float(cache_hit["latency_ms"]["mean"])
    if cache_hit_mean > float(args.max_cache_hit_ms):
        raise RuntimeError(
            f"cache_hit mean latency {cache_hit_mean:.3f}ms exceeds max {args.max_cache_hit_ms:.3f}ms"
        )

    if cache_disabled is not None:
        cache_disabled_mean = float(cache_disabled["latency_ms"]["mean"])
        if cache_hit_mean <= 0:
            raise RuntimeError("cache_hit mean latency is invalid (<=0).")
        speedup = cache_disabled_mean / cache_hit_mean
        if speedup < float(args.min_cache_speedup):
            raise RuntimeError(
                "cache speedup is below threshold: "
                f"{speedup:.3f}x < {args.min_cache_speedup:.3f}x"
            )
        print(f"cache_hit_mean_ms={cache_hit_mean:.3f}")
        print(f"cache_disabled_mean_ms={cache_disabled_mean:.3f}")
        print(f"cache_speedup_x={speedup:.3f}")
    else:
        print(f"cache_hit_mean_ms={cache_hit_mean:.3f}")

    print("PERF_GUARD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

