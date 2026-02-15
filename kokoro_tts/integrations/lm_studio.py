"""LM Studio integration via OpenAI-compatible Chat Completions API."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

LESSON_SYSTEM_PROMPT = """
You are an expert English teacher and curriculum writer.

Your task is to transform raw educational text into a spoken lesson script suitable for text-to-speech narration.

Rules:
1. Output only in English.
2. Keep key facts and instructions from the source.
3. Produce a clear structure:
   - Lesson title
   - Short lesson overview
   - Detailed exercise-by-exercise breakdown
4. For each exercise include:
   - Exercise name or number
   - Goal
   - Step-by-step reasoning
   - Worked solution/explanation
   - Common mistakes
   - Final answer or answer-check guidance
5. If exercises are not clearly separated, infer sensible exercise boundaries and label them Exercise 1, Exercise 2, etc.
6. Make the text natural for listening (short paragraphs, smooth transitions, no tables).
7. Return plain text only. Do not use markdown syntax like **bold**, *italic*, # headings, or markdown bullets.
8. Do not mention these rules.
""".strip()

POS_VERIFY_SYSTEM_PROMPT = """
You verify English token-level morphology annotations.

You are given:
- raw segment text
- local token annotations with offsets
- locked expressions (phrasal verbs/idioms) that MUST NOT be changed

Rules:
1. Return JSON only.
2. Do not add markdown, code fences, comments, or prose.
3. Keep locked expressions unchanged (boundaries and type).
4. token_checks must contain one entry per token index from input.
5. Use UPOS tags from: NOUN, VERB, ADJ, ADV, PRON, PROPN, NUM, DET, ADP, CCONJ, SCONJ, PART, INTJ, PUNCT, SYM, X.
6. Return expression candidates in "expression_table" as a two-column markdown table:
   - first column header: phrasal_verbs
   - second column header: idioms
7. Put one expression per cell. If a cell has no value, leave it empty.
8. If no expressions are found, return an empty table with only header and separator rows.
9. Output must be a single valid JSON object.

Output schema:
{
  "token_checks": [{"token_index": 0, "lemma": "run", "upos": "VERB", "feats": {"VerbForm":"Inf"}}],
  "expression_table": "| phrasal_verbs | idioms |\\n|---|---|\\n| stand up for himself | rock the boat |",
  "new_expressions": []
}
""".strip()

VERIFY_ALLOWED_UPOS = {
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "PROPN",
    "NUM",
    "DET",
    "ADP",
    "CCONJ",
    "SCONJ",
    "PART",
    "INTJ",
    "PUNCT",
    "SYM",
    "X",
}
VERIFY_ALLOWED_EXPRESSION_KINDS = {"phrasal_verb", "idiom"}


@dataclass(frozen=True)
class LessonRequest:
    raw_text: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    temperature: float
    max_tokens: int
    extra_instructions: str = ""


@dataclass(frozen=True)
class PosVerifyRequest:
    segment_text: str
    tokens: list[dict[str, object]]
    locked_expressions: list[dict[str, object]]
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    temperature: float
    max_tokens: int


class LmStudioError(RuntimeError):
    """Raised when LM Studio request fails or returns invalid payload."""


MARKDOWN_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
MARKDOWN_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
MARKDOWN_UNDERSCORE_RE = re.compile(r"__(.+?)__", re.DOTALL)
MARKDOWN_LIST_STAR_RE = re.compile(r"(?m)^(\s*)\*\s+")


def _normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise LmStudioError("LM Studio base URL is empty.")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _model_prefers_no_think(model: str) -> bool:
    model_id = str(model or "").strip().lower()
    return "qwen3" in model_id


def _maybe_prefix_no_think(model: str, content: str) -> str:
    text = str(content or "")
    if not _model_prefers_no_think(model):
        return text
    stripped = text.lstrip()
    if stripped.startswith("/no_think"):
        return text
    return f"/no_think\n{text}"


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", ""))
                if item_type == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunk for chunk in chunks if chunk)
    return ""


def _parse_chat_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LmStudioError("LM Studio response has no choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LmStudioError("LM Studio response has invalid choice format.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LmStudioError("LM Studio response has no message in the first choice.")
    content = _extract_text_from_content(message.get("content"))
    result = content.strip()
    if not result:
        raise LmStudioError("LM Studio returned an empty message.")
    return result


def _sanitize_markdown_for_tts(text: str) -> str:
    """Remove common markdown emphasis markers to keep output TTS-friendly plain text."""
    cleaned = text.replace("\r\n", "\n")
    for _ in range(3):
        cleaned = MARKDOWN_BOLD_RE.sub(r"\1", cleaned)
        cleaned = MARKDOWN_UNDERSCORE_RE.sub(r"\1", cleaned)
        cleaned = MARKDOWN_ITALIC_RE.sub(r"\1", cleaned)
    cleaned = MARKDOWN_LIST_STAR_RE.sub(r"\1- ", cleaned)
    cleaned = cleaned.replace("**", "")
    return cleaned.strip()


def _parse_lesson_response(data: dict[str, Any]) -> str:
    return _sanitize_markdown_for_tts(_parse_chat_response(data))


def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    payload: dict[str, object],
) -> dict[str, Any]:
    endpoint = f"{_normalize_base_url(base_url)}/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {(api_key or 'lm-studio').strip()}",
    }
    http_request = urllib.request.Request(
        endpoint,
        data=body,
        headers=headers,
        method="POST",
    )
    request_timeout: float | None
    try:
        request_timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        request_timeout = None
    if request_timeout is not None and request_timeout <= 0:
        request_timeout = None
    try:
        if request_timeout is None:
            response_ctx = urllib.request.urlopen(http_request)
        else:
            response_ctx = urllib.request.urlopen(http_request, timeout=request_timeout)
        with response_ctx as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_payload = exc.read().decode("utf-8", errors="replace").strip()
        snippet = error_payload[:500] if error_payload else "No body"
        raise LmStudioError(f"LM Studio HTTP {exc.code}: {snippet}") from exc
    except urllib.error.URLError as exc:
        raise LmStudioError(f"Failed to reach LM Studio endpoint: {endpoint}") from exc
    except TimeoutError as exc:
        raise LmStudioError("LM Studio request timed out.") from exc
    except OSError as exc:
        raise LmStudioError(f"LM Studio connection error: {exc}") from exc

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise LmStudioError("LM Studio returned invalid JSON.") from exc


def _extract_first_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise LmStudioError("LM Studio returned an empty message.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise LmStudioError("LM Studio verify response is not valid JSON.")
    if not isinstance(payload, dict):
        raise LmStudioError("LM Studio verify response must be a JSON object.")
    return payload


def _normalize_verify_feats(raw_feats: object) -> dict[str, str]:
    if not isinstance(raw_feats, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw_feats.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            normalized[key_text] = value_text
    return normalized


def _normalize_expression_table_cell(value: object) -> str:
    text = str(value or "").strip().strip("`").strip("\"'")
    text = " ".join(text.split())
    if text.lower() in {"", "-", "—", "none", "n/a", "na", "null", "[]"}:
        return ""
    return text


def _split_expression_cell_values(value: object) -> list[str]:
    text = _normalize_expression_table_cell(value)
    if not text:
        return []
    parts = re.split(r"[;,]+", text)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        candidate = _normalize_expression_table_cell(part)
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _parse_expression_table(table_text: object) -> list[dict[str, object]]:
    raw = str(table_text or "").replace("\r", "\n").strip()
    if not raw:
        return []
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    rows = [line for line in lines if "|" in line]
    if len(rows) < 2:
        return []

    def _parse_row(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    header_cells = _parse_row(rows[0])
    if len(header_cells) < 2:
        return []
    header_lower = [cell.lower().strip() for cell in header_cells]

    phrasal_index = -1
    idiom_index = -1
    for index, header in enumerate(header_lower):
        if phrasal_index < 0 and ("phrasal" in header or "verb" in header):
            phrasal_index = index
        if idiom_index < 0 and "idiom" in header:
            idiom_index = index
    if phrasal_index < 0:
        phrasal_index = 0
    if idiom_index < 0:
        idiom_index = 1 if len(header_cells) > 1 else -1

    out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row_line in rows[1:]:
        cells = _parse_row(row_line)
        if all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in cells if cell.strip()):
            continue
        if phrasal_index >= 0 and phrasal_index < len(cells):
            for phrase in _split_expression_cell_values(cells[phrasal_index]):
                dedupe_key = (phrase.lower(), "phrasal_verb")
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                out.append({"text": phrase, "kind": "phrasal_verb"})
        if idiom_index >= 0 and idiom_index < len(cells):
            for phrase in _split_expression_cell_values(cells[idiom_index]):
                dedupe_key = (phrase.lower(), "idiom")
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                out.append({"text": phrase, "kind": "idiom"})
    return out


def _validate_verify_payload(
    *,
    data: dict[str, Any],
    token_count: int,
) -> dict[str, object]:
    raw_checks = data.get("token_checks")
    if not isinstance(raw_checks, list):
        raise LmStudioError("LM Studio verify response must include token_checks list.")

    by_index: dict[int, dict[str, object]] = {}
    for item in raw_checks:
        if not isinstance(item, dict):
            raise LmStudioError("LM Studio verify token_checks items must be objects.")
        try:
            index_value = int(item.get("token_index"))
        except (TypeError, ValueError) as exc:
            raise LmStudioError("LM Studio verify token_index must be an integer.") from exc
        if index_value < 0 or index_value >= token_count:
            raise LmStudioError("LM Studio verify token_index is out of range.")
        lemma = str(item.get("lemma", "")).strip()
        if not lemma:
            raise LmStudioError("LM Studio verify lemma must be a non-empty string.")
        upos = str(item.get("upos", "")).strip().upper()
        if upos not in VERIFY_ALLOWED_UPOS:
            raise LmStudioError("LM Studio verify upos is invalid.")
        by_index[index_value] = {
            "token_index": index_value,
            "lemma": lemma,
            "upos": upos,
            "feats": _normalize_verify_feats(item.get("feats")),
        }

    if len(by_index) != token_count:
        raise LmStudioError("LM Studio verify token_checks must cover all token indices.")
    token_checks = [by_index[index] for index in range(token_count)]

    raw_expressions = data.get("new_expressions", [])
    if raw_expressions is None:
        raw_expressions = []
    if isinstance(raw_expressions, str):
        raw_expressions = [raw_expressions]
    if not isinstance(raw_expressions, list):
        raise LmStudioError("LM Studio verify new_expressions must be a list or string.")

    new_expressions: list[dict[str, object]] = []
    for item in raw_expressions:
        if isinstance(item, str):
            table_items = _parse_expression_table(item)
            if table_items:
                new_expressions.extend(table_items)
                continue
            text = item.strip()
            if text:
                new_expressions.append({"text": text})
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        normalized: dict[str, object] = {"text": text}
        lemma = str(item.get("lemma", "")).strip()
        if lemma:
            normalized["lemma"] = lemma
        kind = str(item.get("kind", "")).strip().lower()
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
            has_valid_span = start >= 0 and end > start
        except (TypeError, ValueError):
            has_valid_span = False
            start = -1
            end = -1
        if kind in VERIFY_ALLOWED_EXPRESSION_KINDS and has_valid_span:
            normalized["kind"] = kind
            normalized["start"] = start
            normalized["end"] = end
        new_expressions.append(normalized)

    table_items = _parse_expression_table(data.get("expression_table", ""))
    if table_items:
        new_expressions.extend(table_items)

    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for item in new_expressions:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        kind = str(item.get("kind", "")).strip().lower()
        start_raw = item.get("start")
        end_raw = item.get("end")
        try:
            start = int(start_raw) if start_raw is not None and str(start_raw).strip() else -1
        except (TypeError, ValueError):
            start = -1
        try:
            end = int(end_raw) if end_raw is not None and str(end_raw).strip() else -1
        except (TypeError, ValueError):
            end = -1
        key = (text.lower(), kind, start, end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return {"token_checks": token_checks, "new_expressions": deduped}


def verify_pos_with_context(request: PosVerifyRequest) -> dict[str, object]:
    segment_text = (request.segment_text or "").strip()
    if not segment_text:
        raise LmStudioError("Segment text is empty.")
    model = (request.model or "").strip()
    if not model:
        raise LmStudioError("Verify model name is empty. Set LM_VERIFY_MODEL.")
    tokens = request.tokens if isinstance(request.tokens, list) else []
    if not tokens:
        raise LmStudioError("Verify request tokens are empty.")
    locked = request.locked_expressions if isinstance(request.locked_expressions, list) else []

    user_payload = {
        "segment_text": segment_text,
        "tokens": tokens,
        "locked_expressions": locked,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": POS_VERIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _maybe_prefix_no_think(
                    model,
                    json.dumps(user_payload, ensure_ascii=False),
                ),
            },
        ],
        "temperature": float(request.temperature),
        "max_tokens": int(request.max_tokens),
    }
    response = _post_chat_completion(
        base_url=request.base_url,
        api_key=request.api_key,
        timeout_seconds=request.timeout_seconds,
        payload=payload,
    )
    content = _parse_chat_response(response)
    parsed_object = _extract_first_json_object(content)
    return _validate_verify_payload(data=parsed_object, token_count=len(tokens))


def generate_lesson_text(request: LessonRequest) -> str:
    raw_text = (request.raw_text or "").strip()
    if not raw_text:
        raise LmStudioError("Raw text is empty.")
    model = (request.model or "").strip()
    if not model:
        raise LmStudioError(
            "Model name is empty. Set LM_STUDIO_MODEL or enter model in the UI."
        )

    user_prompt = (
        "Convert the following raw text into a detailed lesson script.\n\n"
        "Raw text:\n"
        f"{raw_text}"
    )
    extra = (request.extra_instructions or "").strip()
    if extra:
        user_prompt += f"\n\nAdditional instructions:\n{extra}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LESSON_SYSTEM_PROMPT},
            {"role": "user", "content": _maybe_prefix_no_think(model, user_prompt)},
        ],
        "temperature": float(request.temperature),
        "max_tokens": int(request.max_tokens),
    }
    parsed = _post_chat_completion(
        base_url=request.base_url,
        api_key=request.api_key,
        timeout_seconds=request.timeout_seconds,
        payload=payload,
    )
    return _parse_lesson_response(parsed)
