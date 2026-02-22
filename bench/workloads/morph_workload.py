from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from kokoro_tts.storage.morphology_repository import MorphologyRepository


DEFAULT_VOICES: tuple[str, ...] = ("af_heart", "bf_emma", "af_sarah")


@dataclass(frozen=True)
class MorphWorkloadConfig:
    parts: int = 8
    segments_per_part: int = 120
    tokens_per_segment: int = 140
    unique_segments: int = 24
    voices: Sequence[str] = DEFAULT_VOICES


def build_segment_catalog(*, unique_segments: int, tokens_per_segment: int) -> list[str]:
    unique_segments = max(1, int(unique_segments))
    tokens_per_segment = max(1, int(tokens_per_segment))
    catalog: list[str] = []
    for segment_id in range(unique_segments):
        prefix = f"s{segment_id:03d}"
        words = [f"{prefix}_t{token_id:03d}" for token_id in range(tokens_per_segment)]
        catalog.append(" ".join(words))
    return catalog


def build_parts(
    config: MorphWorkloadConfig,
    *,
    offset: int = 0,
) -> list[list[tuple[str, str]]]:
    parts = max(1, int(config.parts))
    segments_per_part = max(1, int(config.segments_per_part))
    voices = tuple(config.voices) or DEFAULT_VOICES
    catalog = build_segment_catalog(
        unique_segments=config.unique_segments,
        tokens_per_segment=config.tokens_per_segment,
    )
    built: list[list[tuple[str, str]]] = []
    cursor = max(0, int(offset))
    for part_index in range(parts):
        row: list[tuple[str, str]] = []
        for segment_index in range(segments_per_part):
            catalog_index = (cursor + segment_index) % len(catalog)
            voice = voices[(part_index + segment_index) % len(voices)]
            row.append((voice, catalog[catalog_index]))
        built.append(row)
        cursor += segments_per_part
    return built


def synthetic_analyzer(text: str) -> dict[str, object]:
    token_rows: list[dict[str, object]] = []
    offset = 0
    for index, token in enumerate(str(text or "").split()):
        start = offset
        end = start + len(token)
        offset = end + 1
        lemma = token.lower()
        upos = "NOUN" if index % 7 else "VERB"
        token_rows.append(
            {
                "token": token,
                "lemma": lemma,
                "upos": upos,
                "feats": {"Len": str(len(token))},
                "start": start,
                "end": end,
                "key": f"{lemma}|{upos.lower()}",
            }
        )
    return {"language": "en", "items": token_rows}


def synthetic_expression_extractor(text: str) -> list[dict[str, object]]:
    tokens = [item for item in str(text or "").split() if item]
    if len(tokens) < 2:
        return []
    first = tokens[0]
    second = tokens[1]
    combined = f"{first} {second}"
    return [
        {
            "text": combined,
            "lemma": combined.lower(),
            "kind": "phrasal_verb",
            "start": 0,
            "end": len(combined),
            "key": f"{combined.lower()}|phrasal_verb",
            "source": "synthetic",
            "wordnet": False,
        }
    ]


def build_repository(
    *,
    enabled: bool = False,
    db_path: str = ":memory:",
    segment_cache_size: int = 1024,
) -> MorphologyRepository:
    return MorphologyRepository(
        enabled=bool(enabled),
        db_path=db_path,
        analyzer=synthetic_analyzer,
        expression_extractor=synthetic_expression_extractor,
        segment_cache_size=segment_cache_size,
    )

