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


def _parse_response(data: dict[str, Any]) -> str:
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
    return _sanitize_markdown_for_tts(result)


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
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(request.temperature),
        "max_tokens": int(request.max_tokens),
    }

    endpoint = f"{_normalize_base_url(request.base_url)}/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {(request.api_key or 'lm-studio').strip()}",
    }
    http_request = urllib.request.Request(
        endpoint,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=int(request.timeout_seconds)) as response:
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
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise LmStudioError("LM Studio returned invalid JSON.") from exc
    return _parse_response(parsed)
