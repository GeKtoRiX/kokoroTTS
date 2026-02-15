"""Detection of phrasal verbs and idioms in English text."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import logging
import os
from pathlib import Path
import re
from typing import Iterable

logger = logging.getLogger(__name__)

_PHRASAL_PARTICLES = {
    "about",
    "across",
    "after",
    "along",
    "around",
    "aside",
    "away",
    "back",
    "by",
    "down",
    "for",
    "forth",
    "forward",
    "in",
    "into",
    "off",
    "on",
    "out",
    "over",
    "through",
    "together",
    "under",
    "up",
}
_PHRASE_TOKEN_RE = re.compile(r"^[a-z][a-z'-]*$", re.IGNORECASE)
_PHRASAL_DEP_STRICT = "prt"
_PHRASAL_RELAXED_DEPS = {"prep", "advmod"}
_IDIOM_FALLBACK_PHRASES = {
    "break the ice",
    "when pigs fly",
}


@dataclass(frozen=True)
class ExpressionItem:
    text: str
    lemma: str
    kind: str
    start: int
    end: int
    key: str
    source: str
    wordnet: bool


def extract_english_expressions(text: str) -> list[dict[str, object]]:
    """Return detected expressions as JSON-serializable dictionaries."""
    if not text.strip():
        return []
    nlp = _load_expression_spacy_model()
    if nlp is None:
        return []

    doc = nlp(text)
    wordnet_phrasal, wordnet_idioms = _load_wordnet_phrase_inventory()
    items: dict[tuple[str, int, int, str], ExpressionItem] = {}
    for expression in _extract_dependency_phrasals(doc, wordnet_phrasal):
        items[(expression.kind, expression.start, expression.end, expression.lemma)] = expression
    for expression in _extract_wordnet_idioms(doc, wordnet_idioms):
        items[(expression.kind, expression.start, expression.end, expression.lemma)] = expression

    ordered = sorted(items.values(), key=lambda item: (item.start, item.end, item.kind))
    return [
        {
            "text": item.text,
            "lemma": item.lemma,
            "kind": item.kind,
            "start": item.start,
            "end": item.end,
            "key": item.key,
            "source": item.source,
            "wordnet": item.wordnet,
        }
        for item in ordered
    ]


def _extract_dependency_phrasals(doc, wordnet_phrasal: set[str]) -> list[ExpressionItem]:
    matcher = _load_dependency_matcher()
    if matcher is None:
        return []

    items: list[ExpressionItem] = []
    for _, token_ids in matcher(doc):
        token_map = _map_match_tokens(doc, token_ids)
        verb_token = token_map.get("verb")
        particle_token = token_map.get("particle")
        if verb_token is None or particle_token is None:
            continue
        particle_lemma = _token_lemma_lower(particle_token)
        if particle_lemma not in _PHRASAL_PARTICLES:
            continue
        lemma = f"{_token_lemma_lower(verb_token)} {particle_lemma}"
        # For non-prt dependencies, accept only phrases present in WordNet inventory
        # to limit false positives from generic verb+adposition patterns.
        particle_dep = str(getattr(particle_token, "dep_", "")).strip().lower()
        if particle_dep != _PHRASAL_DEP_STRICT and lemma not in wordnet_phrasal:
            continue
        text_value = _span_like_text([verb_token, particle_token])
        start = min(verb_token.idx, particle_token.idx)
        end = max(
            verb_token.idx + len(verb_token.text),
            particle_token.idx + len(particle_token.text),
        )
        items.append(
            ExpressionItem(
                text=text_value,
                lemma=lemma,
                kind="phrasal_verb",
                start=start,
                end=end,
                key=f"{lemma}|phrasal_verb",
                source="dependency_matcher",
                wordnet=lemma in wordnet_phrasal,
            )
        )
    return items


def _extract_wordnet_idioms(doc, wordnet_idioms: set[str]) -> list[ExpressionItem]:
    matcher = _load_idiom_lemma_matcher()
    if matcher is None:
        return []
    idiom_lookup = _build_idiom_lookup(wordnet_idioms)
    items: list[ExpressionItem] = []
    for _, start, end in matcher(doc):
        span = doc[start:end]
        lemma = " ".join(_token_lemma_lower(token) for token in span)
        if not lemma:
            continue
        normalized = lemma.lower()
        canonical = idiom_lookup.get(normalized)
        if not canonical:
            continue
        items.append(
            ExpressionItem(
                text=span.text,
                lemma=canonical,
                kind="idiom",
                start=span.start_char,
                end=span.end_char,
                key=f"{canonical}|idiom",
                source="wordnet_phrase_matcher",
                wordnet=True,
            )
        )
    return items


def _map_match_tokens(doc, token_ids: Iterable[int]) -> dict[str, object]:
    tokens = [doc[token_id] for token_id in token_ids]
    token_map = {}
    for token in tokens:
        if token.dep_ == "prt" and "particle" not in token_map:
            token_map["particle"] = token
            continue
        if token.pos_ in ("VERB", "AUX") and "verb" not in token_map:
            token_map["verb"] = token
    if "verb" not in token_map and tokens:
        token_map["verb"] = tokens[0]
    if "particle" not in token_map and len(tokens) > 1:
        token_map["particle"] = tokens[-1]
    return token_map


def _token_lemma_lower(token) -> str:
    lemma = (token.lemma_ or token.text or "").strip().lower()
    return lemma or token.text.lower()


def _span_like_text(tokens) -> str:
    ordered = sorted(tokens, key=lambda token: token.i)
    return " ".join(token.text for token in ordered)


def _build_idiom_lookup(idioms: set[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for idiom in idioms:
        canonical = str(idiom or "").strip().lower()
        if not canonical:
            continue
        lookup.setdefault(canonical, canonical)
        for variant in _phrase_inflection_variants(canonical):
            lookup.setdefault(variant, canonical)
    return lookup


def _phrase_inflection_variants(phrase: str) -> set[str]:
    tokens = [token for token in str(phrase or "").strip().lower().split() if token]
    if not tokens:
        return set()
    option_groups = [_token_inflection_options(token) for token in tokens]
    return {" ".join(combo) for combo in product(*option_groups)}


def _token_inflection_options(token: str) -> set[str]:
    options = {token}
    if not token.isalpha() or len(token) <= 3:
        return options
    singular = _naive_singular(token)
    options.add(singular)
    return {option for option in options if option}


def _naive_singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return f"{token[:-3]}y"
    if token.endswith("es") and len(token) > 3 and token[-3] in ("s", "x", "z"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


@lru_cache(maxsize=1)
def _load_expression_spacy_model():
    try:
        import spacy
    except Exception:
        return None

    try:
        return spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    except Exception:
        auto_download = os.getenv("SPACY_EN_MODEL_AUTO_DOWNLOAD", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if auto_download:
            try:
                from spacy.cli import download as spacy_download

                logger.info("Downloading spaCy model en_core_web_sm for expression matching")
                spacy_download("en_core_web_sm")
                return spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            except Exception:
                logger.exception("spaCy model auto-download failed")
        logger.info("Expression matcher disabled: spaCy model en_core_web_sm not available")
        return None


@lru_cache(maxsize=1)
def _load_dependency_matcher():
    nlp = _load_expression_spacy_model()
    if nlp is None:
        return None
    try:
        from spacy.matcher import DependencyMatcher
    except Exception:
        return None

    matcher = DependencyMatcher(nlp.vocab)
    pattern = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}},
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "particle",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["ADP", "ADV", "PART"]},
                "DEP": {"IN": [_PHRASAL_DEP_STRICT, *_PHRASAL_RELAXED_DEPS]},
                "LEMMA": {"IN": sorted(_PHRASAL_PARTICLES)},
            },
        },
    ]
    matcher.add("PHRASAL_VERB", [pattern])
    return matcher


@lru_cache(maxsize=1)
def _load_idiom_lemma_matcher():
    nlp = _load_expression_spacy_model()
    if nlp is None:
        return None
    _, idioms = _load_wordnet_phrase_inventory()
    if not idioms:
        return None
    try:
        from spacy.matcher import Matcher
    except Exception:
        return None

    matcher = Matcher(nlp.vocab, validate=False)
    patterns = [
        [
            _idiom_match_token_pattern(token)
            for token in phrase.split()
        ]
        for phrase in sorted(idioms)
    ]
    if not patterns:
        return None
    matcher.add("WORDNET_IDIOM", patterns)
    return matcher


def _idiom_match_token_pattern(token: str) -> dict[str, object]:
    options = sorted(_token_inflection_options(token))
    if len(options) == 1:
        return {"LEMMA": options[0]}
    return {"LEMMA": {"IN": options}}


@lru_cache(maxsize=1)
def _load_wordnet_phrase_inventory() -> tuple[set[str], set[str]]:
    wn = _load_wordnet_corpus()
    if wn is None:
        return set(), set()
    phrasal: set[str] = set()
    idioms: set[str] = set()
    try:
        synsets = wn.all_synsets()
    except Exception:
        logger.exception("WordNet synset iteration failed")
        return set(), set()

    for synset in synsets:
        for lemma_name in synset.lemma_names():
            if "_" not in lemma_name:
                continue
            phrase = lemma_name.replace("_", " ").lower().strip()
            tokens = phrase.split()
            if len(tokens) < 2 or len(tokens) > 4:
                continue
            if not all(_PHRASE_TOKEN_RE.match(token) for token in tokens):
                continue
            if synset.pos() == "v" and len(tokens) in (2, 3) and tokens[-1] in _PHRASAL_PARTICLES:
                phrasal.add(phrase)
            else:
                # Two-token WordNet phrases are often compositional collocations
                # ("forest fire", "and then") and create many false-positive idioms.
                if len(tokens) >= 3:
                    idioms.add(phrase)
    idioms.update(_IDIOM_FALLBACK_PHRASES)
    return phrasal, idioms


@lru_cache(maxsize=1)
def _load_wordnet_corpus():
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except Exception:
        return None

    data_dir = os.getenv(
        "WORDNET_DATA_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "nltk_data"),
    )
    os.makedirs(data_dir, exist_ok=True)
    if data_dir not in nltk.data.path:
        nltk.data.path.insert(0, data_dir)

    try:
        wn.ensure_loaded()
        return wn
    except LookupError:
        auto_download = os.getenv("WORDNET_AUTO_DOWNLOAD", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not auto_download:
            logger.info("WordNet data missing and WORDNET_AUTO_DOWNLOAD is disabled")
            return None
        try:
            logger.info("Downloading WordNet locally to %s", data_dir)
            nltk.download("wordnet", download_dir=data_dir, quiet=True)
            nltk.download("omw-1.4", download_dir=data_dir, quiet=True)
            wn.ensure_loaded()
            return wn
        except Exception:
            logger.exception("WordNet download failed")
            return None
