from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kokoro_tts.domain.lexeme_checks import (
    analyze_and_validate_english_lexemes,
    load_lm_verify_settings_from_env,
    verify_english_lexemes_with_lm,
)
from kokoro_tts.integrations.lm_studio import LmStudioError

DEFAULT_TEXT = "Look up the words, then run the tests again."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-check English text splitting into lexemes with optional LM verification.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="English text to analyze.",
    )
    parser.add_argument(
        "--verify-llm",
        action="store_true",
        help="Run additional LM Studio verification (requires LM_VERIFY_* env vars).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary at the end.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        analysis = analyze_and_validate_english_lexemes(args.text)
        items = analysis.get("items", [])
        token_count = len(items) if isinstance(items, list) else 0

        summary: dict[str, object] = {
            "status": "ok",
            "token_count": token_count,
            "items": items,
        }
        print(f"LEXEME_CHECK_OK tokens={token_count}")

        if args.verify_llm:
            settings = load_lm_verify_settings_from_env()
            verify_summary = verify_english_lexemes_with_lm(
                args.text,
                analysis,
                settings=settings,
            )
            summary["llm"] = verify_summary
            print(f"LLM_VERIFY_OK mismatches={verify_summary['mismatch_count']}")

        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, LmStudioError, RuntimeError) as exc:
        print(f"LEXEME_CHECK_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
