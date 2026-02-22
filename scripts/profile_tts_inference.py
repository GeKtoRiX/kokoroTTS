from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TEXT = (
    "Hello! This is a realistic production latency test for Kokoro TTS. "
    "We measure first generation and subsequent generations with mixed punctuation, "
    "numbers like 1,234 and time 12:30 p.m. to include preprocessing costs. "
    "Please keep speaking naturally and clearly for the benchmark."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile real Kokoro TTS inference latency in production-like settings.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Benchmark text.")
    parser.add_argument("--voice", default="af_heart", help="Voice id.")
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu", "both"),
        default="both",
        help="Device mode to profile.",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=3,
        help="Number of warm runs after first run.",
    )
    parser.add_argument(
        "--tts-only",
        type=int,
        default=1,
        help="Set runtime mode to TTS-only (1) or TTS+Morphology (0).",
    )
    parser.add_argument(
        "--save-outputs",
        type=int,
        default=0,
        help="Save output files during benchmark (1=yes, 0=no).",
    )
    parser.add_argument(
        "--profile-python",
        type=int,
        default=0,
        help="Print cProfile top functions for a single warm run (1=yes, 0=no).",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=25,
        help="Top function count for cProfile output.",
    )
    return parser


def _mode_flags(mode: str, cuda_available: bool) -> list[tuple[str, bool]]:
    if mode == "cpu":
        return [("cpu", False)]
    if mode == "gpu":
        return [("gpu", True)]
    flags = [("cpu", False)]
    if cuda_available:
        flags.append(("gpu", True))
    return flags


def _run_once(
    state, *, text: str, voice: str, use_gpu: bool, save_outputs: bool
) -> tuple[float, int]:
    started = time.perf_counter()
    result, _ = state.generate_first(
        text=text,
        voice=voice,
        use_gpu=use_gpu,
        save_outputs=save_outputs,
        pause_seconds=0.0,
        output_format="wav",
        style_preset="neutral",
    )
    elapsed = time.perf_counter() - started
    samples = 0 if result is None else int(result[1].shape[0])
    return elapsed, samples


def _print_profile(
    state, *, text: str, voice: str, use_gpu: bool, save_outputs: bool, top: int
) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    _run_once(
        state,
        text=text,
        voice=voice,
        use_gpu=use_gpu,
        save_outputs=save_outputs,
    )
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(top)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    os.environ["KOKORO_SKIP_APP_INIT"] = "0"

    import app

    app.set_tts_only_mode(bool(args.tts_only))
    state = app._current_app_state()
    if state is None:
        print("error=app_state_unavailable")
        return 1

    flags = _mode_flags(args.mode, app.CUDA_AVAILABLE)
    if args.mode == "gpu" and not app.CUDA_AVAILABLE:
        print("mode=gpu status=unavailable")
        return 0

    for label, use_gpu in flags:
        first_elapsed, samples = _run_once(
            state,
            text=args.text,
            voice=args.voice,
            use_gpu=use_gpu,
            save_outputs=bool(args.save_outputs),
        )
        warm_elapsed: list[float] = []
        for _ in range(max(0, int(args.warm_runs))):
            elapsed, _ = _run_once(
                state,
                text=args.text,
                voice=args.voice,
                use_gpu=use_gpu,
                save_outputs=bool(args.save_outputs),
            )
            warm_elapsed.append(elapsed)
        warm_avg = (sum(warm_elapsed) / len(warm_elapsed)) if warm_elapsed else 0.0
        print(
            f"mode={label} first={first_elapsed:.6f} warm_avg={warm_avg:.6f} "
            f"warm={';'.join(f'{value:.6f}' for value in warm_elapsed)} samples={samples}"
        )
        if args.profile_python:
            _print_profile(
                state,
                text=args.text,
                voice=args.voice,
                use_gpu=use_gpu,
                save_outputs=bool(args.save_outputs),
                top=max(1, int(args.profile_top)),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
