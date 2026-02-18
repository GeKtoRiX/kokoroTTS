"""English token analysis with deterministic JSON output for DB ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import os
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
_UNINFORMATIVE_UPOS = {"X", "SYM"}
_SCONJ_HINT_WORDS = {
    "after",
    "although",
    "as",
    "because",
    "before",
    "if",
    "once",
    "since",
    "though",
    "unless",
    "until",
    "when",
    "whenever",
    "whereas",
    "while",
}
_PENN_POS_TO_UPOS = {
    "CC": "CCONJ",
    "CD": "NUM",
    "DT": "DET",
    "EX": "PRON",
    "FW": "X",
    "IN": "ADP",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "LS": "X",
    "MD": "VERB",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "PROPN",
    "NNPS": "PROPN",
    "PDT": "DET",
    "POS": "PART",
    "PRP": "PRON",
    "PRP$": "PRON",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "RP": "PART",
    "SYM": "SYM",
    "TO": "PART",
    "UH": "INTJ",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "WDT": "DET",
    "WP": "PRON",
    "WP$": "PRON",
    "WRB": "ADV",
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
    spacy_annotations = _annotate_with_spacy(text, tokens)
    combined = _merge_annotation_sources(tokens, stanza_annotations, spacy_annotations)
    if combined is None:
        combined = [_heuristic_annotation(token) for token in tokens]

    if _needs_flair_backfill(tokens, combined):
        flair_annotations = _annotate_with_flair(tokens)
        combined = _merge_annotation_sources(tokens, combined, flair_annotations) or combined
    return combined


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


def _merge_annotation_sources(
    tokens: Sequence[RawToken],
    primary: Sequence[TokenAnnotation] | None,
    secondary: Sequence[TokenAnnotation] | None,
) -> list[TokenAnnotation] | None:
    primary_ok = _annotations_match_tokens(primary, len(tokens))
    secondary_ok = _annotations_match_tokens(secondary, len(tokens))
    if not primary_ok and not secondary_ok:
        return None
    if primary_ok and not secondary_ok:
        return list(primary or [])
    if secondary_ok and not primary_ok:
        return list(secondary or [])

    assert primary is not None
    assert secondary is not None
    merged: list[TokenAnnotation] = []
    for token, first, second in zip(tokens, primary, secondary):
        merged.append(_merge_annotation_pair(token, first, second))
    return merged


def _annotations_match_tokens(
    annotations: Sequence[TokenAnnotation] | None,
    token_count: int,
) -> bool:
    return annotations is not None and len(annotations) == token_count


def _merge_annotation_pair(
    token: RawToken,
    first: TokenAnnotation,
    second: TokenAnnotation,
) -> TokenAnnotation:
    first_upos = _normalize_upos(first.upos, token)
    second_upos = _normalize_upos(second.upos, token)
    merged_upos = first_upos
    if first_upos in _UNINFORMATIVE_UPOS and second_upos not in _UNINFORMATIVE_UPOS:
        merged_upos = second_upos
    elif (
        first_upos == "ADP"
        and second_upos == "SCONJ"
        and token.text.strip().lower() in _SCONJ_HINT_WORDS
    ):
        merged_upos = "SCONJ"

    first_lemma = _normalize_lemma(first.lemma, token.text)
    second_lemma = _normalize_lemma(second.lemma, token.text)
    merged_lemma = first_lemma
    if _lemma_is_identity(first_lemma, token.text) and not _lemma_is_identity(second_lemma, token.text):
        merged_lemma = second_lemma

    merged_feats = _normalize_feats(second.feats)
    merged_feats.update(_normalize_feats(first.feats))
    return TokenAnnotation(lemma=merged_lemma, upos=merged_upos, feats=merged_feats)


def _lemma_is_identity(lemma: str, token_text: str) -> bool:
    return lemma.strip().lower() == token_text.strip().lower()


def _needs_flair_backfill(
    tokens: Sequence[RawToken],
    annotations: Sequence[TokenAnnotation],
) -> bool:
    for token, annotation in zip(tokens, annotations):
        upos = _normalize_upos(annotation.upos, token)
        if upos in _UNINFORMATIVE_UPOS and not token.is_punct and not token.like_num:
            return True
    return False


def _annotate_with_flair(
    tokens: Sequence[RawToken],
) -> list[TokenAnnotation] | None:
    tagger = _load_flair_tagger()
    if tagger is None:
        return None
    try:
        from flair.data import Sentence
    except Exception:
        return None

    token_texts = [token.text for token in tokens]
    if not token_texts:
        return []
    try:
        sentence = Sentence(token_texts, use_tokenizer=False)
    except TypeError:
        sentence = Sentence(" ".join(token_texts))
    except Exception:
        logger.exception("Flair sentence creation failed")
        return None

    try:
        tagger.predict(sentence)
    except Exception:
        logger.exception("Flair annotation failed, falling back")
        return None

    flair_tokens = list(getattr(sentence, "tokens", []))
    if len(flair_tokens) != len(tokens):
        logger.warning(
            "Flair token mismatch: tokens=%s flair_tokens=%s",
            len(tokens),
            len(flair_tokens),
        )
        return None

    tag_type = str(getattr(tagger, "tag_type", "") or "upos")
    annotations: list[TokenAnnotation] = []
    for raw_token, flair_token in zip(tokens, flair_tokens):
        label = _read_flair_label(flair_token, tag_type)
        if not label:
            label = _read_flair_label(flair_token, "upos")
        if not label:
            label = _read_flair_label(flair_token, "pos")
        annotations.append(
            TokenAnnotation(
                lemma=raw_token.text,
                upos=_normalize_flair_upos(label, raw_token),
                feats={},
            )
        )
    return annotations


def _read_flair_label(token, tag_type: str) -> str:
    try:
        label = token.get_label(tag_type)
    except Exception:
        return ""
    value = str(getattr(label, "value", "")).strip()
    return value


def _normalize_flair_upos(label: str, token: RawToken) -> str:
    candidate = str(label or "").strip().upper()
    if candidate in UPOS_VALUES:
        return candidate
    candidate = _PENN_POS_TO_UPOS.get(candidate, candidate)
    if candidate == "ADP" and token.text.strip().lower() in _SCONJ_HINT_WORDS:
        return "SCONJ"
    if candidate in UPOS_VALUES:
        return candidate
    return _fallback_upos(token)


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

    for model_name in _spacy_model_candidates():
        try:
            return spacy.load(model_name, disable=["ner", "parser", "textcat"])
        except Exception:
            continue

    logger.info("spaCy model is unavailable, using blank('en')")
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

    pipeline_kwargs = {
        "lang": "en",
        "processors": "tokenize,pos,lemma",
        "tokenize_pretokenized": True,
        "tokenize_no_ssplit": True,
        "use_gpu": _env_flag("MORPH_STANZA_USE_GPU"),
        "verbose": False,
    }
    package_name = os.getenv("MORPH_STANZA_PACKAGE", "").strip()
    if package_name:
        pipeline_kwargs["package"] = package_name

    try:
        return stanza.Pipeline(**pipeline_kwargs)
    except Exception:
        logger.info("Stanza pipeline unavailable; continuing without stanza")
        return None


@lru_cache(maxsize=1)
def _load_flair_tagger():
    if not _env_flag("MORPH_FLAIR_ENABLED", default=True):
        return None
    try:
        from flair.models import SequenceTagger
    except Exception:
        return None

    for model_name in _flair_model_candidates():
        try:
            tagger = SequenceTagger.load(model_name)
            logger.info("Flair tagger loaded: %s", model_name)
            return tagger
        except Exception:
            continue
    logger.info("Flair tagger is unavailable; continuing without flair")
    return None


def _spacy_model_candidates() -> list[str]:
    raw = str(os.getenv("MORPH_SPACY_MODELS", "") or "").strip()
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if values:
            return values
    return [
        "en_core_web_trf",
        "en_core_web_lg",
        "en_core_web_md",
        "en_core_web_sm",
    ]


def _flair_model_candidates() -> list[str]:
    raw = str(os.getenv("MORPH_FLAIR_MODELS", "") or "").strip()
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if values:
            return values
    return [
        "flair/upos-english-fast",
        "flair/upos-english",
        "pos-fast",
        "pos",
    ]


def _env_flag(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    value = os.getenv(name, fallback).strip().lower()
    return value in {"1", "true", "yes", "on"}
