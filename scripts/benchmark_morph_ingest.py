from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kokoro_tts.storage.morphology_repository import MorphologyRepository


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic benchmark for MorphologyRepository._collect_ingest_rows.",
    )
    parser.add_argument("--parts", type=int, default=5, help="Number of top-level parts.")
    parser.add_argument(
        "--segments-per-part",
        type=int,
        default=120,
        help="Segments in each part.",
    )
    parser.add_argument(
        "--tokens-per-segment",
        type=int,
        default=200,
        help="Approximate token count in each segment.",
    )
    return parser


def _build_text(token_count: int) -> str:
    return " ".join("Token" for _ in range(max(1, token_count)))


def _analyzer(text: str) -> dict[str, object]:
    items: list[dict[str, object]] = []
    offset = 0
    for word in text.split():
        start = text.find(word, offset)
        end = start + len(word)
        offset = end
        items.append(
            {
                "token": word,
                "lemma": word.lower(),
                "upos": "NOUN",
                "feats": {},
                "start": start,
                "end": end,
                "key": f"{word.lower()}|noun",
            }
        )
    return {"items": items}


def _expression_extractor(_text: str) -> list[dict[str, object]]:
    return []


def _build_parts(part_count: int, segments_per_part: int, tokens_per_segment: int):
    segment_text = _build_text(tokens_per_segment)
    return [
        [("af_heart", segment_text) for _ in range(segments_per_part)]
        for _ in range(part_count)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    parts = _build_parts(args.parts, args.segments_per_part, args.tokens_per_segment)
    repository = MorphologyRepository(
        enabled=False,
        db_path=":memory:",
        analyzer=_analyzer,
        expression_extractor=_expression_extractor,
    )

    started_at = time.perf_counter()
    token_rows, expression_rows = repository._collect_ingest_rows(
        parts,
        source="benchmark",
    )
    elapsed = time.perf_counter() - started_at

    segment_count = args.parts * args.segments_per_part
    print(f"segments={segment_count}")
    print(f"tokens={len(token_rows)}")
    print(f"expressions={len(expression_rows)}")
    print(f"elapsed_seconds={elapsed:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
