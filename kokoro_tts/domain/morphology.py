"""English token analysis with deterministic JSON output for DB ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import re
import unicodedata
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

UPOS_VALUES = {
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

_SPACY_POS_MAP = {
    "AUX": "VERB",
}

_SYMBOL_EXTRAS = {"$", "%", "&", "@", "+", "=", "*", "^", "~"}
_REGEX_TOKEN_RE = re.compile(
    r"\s+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]",
    re.UNICODE,
)
_NUM_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")


@dataclass(frozen=True)
class RawToken:
    text: str
    start: int
    end: int
    is_space: bool
    is_punct: bool
    like_num: bool


@dataclass(frozen=True)
class TokenAnnotation:
    lemma: str
    upos: str
    feats: dict[str, str]


Tokenizer = Callable[[str], Sequence[RawToken]]
Annotator = Callable[[str, Sequence[RawToken]], Sequence[TokenAnnotation] | None]


def analyze_english_text(
    text: str,
    *,
    tokenizer: Tokenizer | None = None,
    annotator: Annotator | None = None,
) -> dict[str, object]:
    """Return token analysis in stable JSON-ready schema."""
    if text == "":
        return {"language": "en", "items": []}

    tokenize = tokenizer or _default_tokenize
    annotate = annotator or _default_annotate

    raw_tokens = [token for token in tokenize(text) if not token.is_space]
    if not raw_tokens:
        return {"language": "en", "items": []}

    annotations = annotate(text, raw_tokens)
    if annotations is None or len(annotations) != len(raw_tokens):
        annotations = [_heuristic_annotation(token) for token in raw_tokens]

    items: list[dict[str, object]] = []
    for token, annotation in zip(raw_tokens, annotations):
        upos = _normalize_upos(annotation.upos, token)
        if upos == "PUNCT":
            continue
        lemma = _normalize_lemma(annotation.lemma, token.text)
        feats = _normalize_feats(annotation.feats)
        items.append(
            {
                "token": token.text,
                "lemma": lemma,
                "upos": upos,
                "feats": feats,
                "start": token.start,
                "end": token.end,
                "key": _build_key(lemma, upos),
            }
        )

    return {"language": "en", "items": items}


def _default_tokenize(text: str) -> list[RawToken]:
    nlp = _load_spacy_model()
    if nlp is None:
        return _regex_tokenize(text)

    doc = nlp.make_doc(text)
    return [
        RawToken(
            text=token.text,
            start=token.idx,
            end=token.idx + len(token.text),
            is_space=bool(token.is_space),
            is_punct=bool(token.is_punct),
            like_num=bool(token.like_num),
        )
        for token in doc
    ]


def _default_annotate(
    text: str,
    tokens: Sequence[RawToken],
) -> list[TokenAnnotation]:
    stanza_annotations = _annotate_with_stanza(tokens)
    if stanza_annotations is not None and len(stanza_annotations) == len(tokens):
        return stanza_annotations

    spacy_annotations = _annotate_with_spacy(text, tokens)
    if spacy_annotations is not None and len(spacy_annotations) == len(tokens):
        return spacy_annotations

    return [_heuristic_annotation(token) for token in tokens]


def _annotate_with_stanza(
    tokens: Sequence[RawToken],
) -> list[TokenAnnotation] | None:
    pipeline = _load_stanza_pipeline()
    if pipeline is None:
        return None

    try:
        doc = pipeline([[token.text for token in tokens]])
    except Exception:
        logger.exception("Stanza annotation failed, falling back")
        return None

    words = [word for sentence in doc.sentences for word in sentence.words]
    if len(words) != len(tokens):
        logger.warning(
            "Stanza token/word mismatch: tokens=%s words=%s",
            len(tokens),
            len(words),
        )
        return None

    return [
        TokenAnnotation(
            lemma=word.lemma or token.text,
            upos=word.upos or "X",
            feats=_parse_ud_feats(word.feats),
        )
        for token, word in zip(tokens, words)
    ]


def _annotate_with_spacy(
    text: str,
    tokens: Sequence[RawToken],
) -> list[TokenAnnotation] | None:
    nlp = _load_spacy_model()
    if nlp is None:
        return None

    try:
        doc = nlp(text)
    except Exception:
        logger.exception("spaCy annotation failed, falling back")
        return None

    non_space = [token for token in doc if not token.is_space]
    if len(non_space) != len(tokens):
        token_map = {(token.idx, token.idx + len(token.text)): token for token in non_space}
        mapped: list[TokenAnnotation] = []
        for raw in tokens:
            match = token_map.get((raw.start, raw.end))
            if match is None:
                return None
            mapped.append(
                TokenAnnotation(
                    lemma=match.lemma_ or raw.text,
                    upos=match.pos_ or "X",
                    feats={key: str(value) for key, value in match.morph.to_dict().items()},
                )
            )
        return mapped

    return [
        TokenAnnotation(
            lemma=token.lemma_ or raw.text,
            upos=token.pos_ or "X",
            feats={key: str(value) for key, value in token.morph.to_dict().items()},
        )
        for raw, token in zip(tokens, non_space)
    ]


def _heuristic_annotation(token: RawToken) -> TokenAnnotation:
    return TokenAnnotation(
        lemma=token.text,
        upos=_fallback_upos(token),
        feats={},
    )


def _fallback_upos(token: RawToken) -> str:
    if token.is_punct:
        return "PUNCT"
    if token.like_num or _NUM_RE.match(token.text):
        return "NUM"
    if _is_symbol_token(token.text):
        return "SYM"
    return "X"


def _normalize_upos(raw_upos: str, token: RawToken) -> str:
    candidate = (raw_upos or "").strip().upper()
    candidate = _SPACY_POS_MAP.get(candidate, candidate)
    if candidate in UPOS_VALUES:
        return candidate
    return _fallback_upos(token)


def _normalize_lemma(raw_lemma: str, token_text: str) -> str:
    lemma = (raw_lemma or "").strip()
    return lemma if lemma else token_text


def _normalize_feats(raw_feats: dict[str, object] | None) -> dict[str, str]:
    if not raw_feats:
        return {}
    return {
        str(key): str(value)
        for key, value in raw_feats.items()
        if str(key).strip() and str(value).strip()
    }


def _build_key(lemma: str, upos: str) -> str:
    lemma_lower = lemma.strip().lower()
    return f"{lemma_lower}|{upos.lower()}"


def _parse_ud_feats(feats: str | None) -> dict[str, str]:
    if not feats:
        return {}
    parsed: dict[str, str] = {}
    for chunk in feats.split("|"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            parsed[key] = value
    return parsed


def _is_symbol_token(token_text: str) -> bool:
    stripped = token_text.strip()
    if not stripped:
        return False
    if stripped in _SYMBOL_EXTRAS:
        return True
    if any(ch.isalnum() for ch in stripped):
        return False
    return all(unicodedata.category(ch).startswith("S") for ch in stripped)


def _regex_tokenize(text: str) -> list[RawToken]:
    tokens: list[RawToken] = []
    for match in _REGEX_TOKEN_RE.finditer(text):
        chunk = match.group(0)
        is_space = chunk.isspace()
        is_punct = _is_punct_text(chunk)
        like_num = bool(_NUM_RE.match(chunk))
        tokens.append(
            RawToken(
                text=chunk,
                start=match.start(),
                end=match.end(),
                is_space=is_space,
                is_punct=is_punct,
                like_num=like_num,
            )
        )
    return tokens


def _is_punct_text(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if _is_symbol_token(stripped):
        return False
    return all(unicodedata.category(ch).startswith("P") for ch in stripped)


@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        import spacy
    except Exception:
        return None

    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
    except Exception:
        logger.info("spaCy model en_core_web_sm not available, using blank('en')")
        try:
            return spacy.blank("en")
        except Exception:
            return None


@lru_cache(maxsize=1)
def _load_stanza_pipeline():
    try:
        import stanza
    except Exception:
        return None

    try:
        return stanza.Pipeline(
            lang="en",
            processors="tokenize,pos,lemma",
            tokenize_pretokenized=True,
            use_gpu=False,
            verbose=False,
        )
    except Exception:
        logger.info("Stanza pipeline unavailable; continuing without stanza")
        return None
