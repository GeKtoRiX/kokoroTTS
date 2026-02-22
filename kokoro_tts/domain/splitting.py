"""Sentence and chunk splitting helpers."""

from __future__ import annotations

import logging
import re

from .text_utils import _find_skip_spans

logger = logging.getLogger("kokoro_app")

SENTENCE_BREAK_RE = re.compile(r'([.!?]+)(["\')\]]?)(\s+|$)')
SOFT_BREAK_RE = re.compile(r"[,;:\-]\s+")

ABBREV_TITLES = {"mr", "ms", "mrs", "dr", "prof", "sr", "jr", "st", "vs", "etc"}
ABBREV_DOTTED = {"e.g", "i.e"}


def _is_abbrev(text: str, punct_index: int) -> bool:
    if punct_index >= 3:
        dotted = text[punct_index - 3 : punct_index].lower()
        if dotted in ABBREV_DOTTED:
            if punct_index == 3 or not text[punct_index - 4].isalnum():
                return True
    i = punct_index - 1
    while i >= 0 and text[i].isalpha():
        i -= 1
    word = text[i + 1 : punct_index].lower()
    if not word:
        return False
    if word in ABBREV_TITLES:
        if i < 0 or not text[i].isalnum():
            return True
    return False


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    if not any(ch in text for ch in ".!?"):
        return [text.strip()]
    sentences: list[str] = []
    replacements = 0
    start = 0
    skip_spans = _find_skip_spans(text)
    span_index = 0
    span_count = len(skip_spans)
    for match in SENTENCE_BREAK_RE.finditer(text):
        match_start = match.start(1)
        while span_index < span_count and skip_spans[span_index][1] <= match_start:
            span_index += 1
        if (
            span_index < span_count
            and skip_spans[span_index][0] <= match_start < skip_spans[span_index][1]
        ):
            continue
        if _is_abbrev(text, match_start):
            continue
        end = match.end(1)
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
        replacements += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    logger.debug(
        "Sentence split: inserted_breaks=%s count=%s",
        replacements,
        len(sentences),
    )
    return sentences


def _split_long_piece(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]
    parts = [p.strip() for p in SOFT_BREAK_RE.split(piece) if p.strip()]
    if len(parts) <= 1:
        words = piece.split()
        out: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in words:
            add_len = len(word) + (1 if current else 0)
            if current_len + add_len > max_chars and current:
                out.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += add_len
        if current:
            out.append(" ".join(current))
        return out

    out: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + 2 + len(part) <= max_chars:
            current = f"{current}, {part}"
        else:
            out.append(current)
            current = part
    if current:
        out.append(current)
    return out


def smart_split(text: str, max_chars: int, keep_sentences: bool = False) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = split_sentences(text)
    if keep_sentences:
        chunks: list[str] = []
        for sentence in sentences:
            if len(sentence) > max_chars:
                chunks.extend(_split_long_piece(sentence, max_chars))
            else:
                chunks.append(sentence)
        logger.debug("Smart split: sentences=%s chunks=%s", len(sentences), len(chunks))
        return chunks

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(_split_long_piece(sentence, max_chars))
            continue
        add_len = len(sentence) + (1 if current else 0)
        if current_len + add_len <= max_chars:
            current.append(sentence)
            current_len += add_len
        else:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
    if current:
        chunks.append(" ".join(current))
    logger.debug("Smart split: sentences=%s chunks=%s", len(sentences), len(chunks))
    return chunks


def split_parts(text: str) -> list[str]:
    if "|" not in text:
        return [text]
    parts = [part.strip() for part in text.split("|")]
    parts = [part for part in parts if part]
    if not parts:
        return [""]
    logger.debug("Manual split parts=%s", len(parts))
    return parts
