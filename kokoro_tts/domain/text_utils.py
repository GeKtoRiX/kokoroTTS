"""Shared utilities for handling text spans that should be skipped."""
from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

MD_LINK_RE = re.compile(r"\[[^\]]*\]\(/[^)\n]*\)")
SLASHED_RE = re.compile(r"(?<![A-Za-z0-9:])/[^\n/]+/")


def _find_skip_spans(text: str) -> list[tuple[int, int]]:
    if "[" not in text and "/" not in text:
        return []
    spans: list[tuple[int, int]] = []
    for match in MD_LINK_RE.finditer(text):
        spans.append((match.start(), match.end()))
    for match in SLASHED_RE.finditer(text):
        spans.append((match.start(), match.end()))
    if not spans:
        return []
    spans.sort(key=lambda item: item[0])
    return _merge_sorted_spans(spans)


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = list(spans)
    if not spans:
        return []
    spans.sort(key=lambda item: item[0])
    return _merge_sorted_spans(spans)


def _apply_outside_spans(
    text: str,
    spans: Sequence[tuple[int, int]],
    func: Callable[[str], str],
) -> str:
    if not spans:
        return func(text)
    parts: list[str] = []
    last = 0
    for start, end in spans:
        if start > last:
            parts.append(func(text[last:start]))
        parts.append(text[start:end])
        last = end
    if last < len(text):
        parts.append(func(text[last:]))
    return "".join(parts)


def _is_within_spans(index: int, spans: Sequence[tuple[int, int]]) -> bool:
    if not spans:
        return False
    if any(spans[i][0] > spans[i + 1][0] for i in range(len(spans) - 1)):
        return any(start <= index < end for start, end in spans)
    left = 0
    right = len(spans) - 1
    while left <= right:
        middle = (left + right) // 2
        start, end = spans[middle]
        if index < start:
            right = middle - 1
        elif index >= end:
            left = middle + 1
        else:
            return True
    return False


def _merge_sorted_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]
