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
    for expression in _extract_inventory_phrasals(doc, wordnet_phrasal):
        items[(expression.kind, expression.start, expression.end, expression.lemma)] = expression
    for expression in _extract_textacy_phrasals(doc, wordnet_phrasal):
        items[(expression.kind, expression.start, expression.end, expression.lemma)] = expression
    for expression in _extract_phrasemachine_phrasals(doc, wordnet_phrasal):
        items[(expression.kind, expression.start, expression.end, expression.lemma)] = expression
    idioms = _extract_wordnet_idioms(doc, wordnet_idioms)
    idioms = _filter_idioms_with_pywsd(doc, idioms)
    for expression in idioms:
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
        start = min(verb_token.idx, particle_token.idx)
        end = max(
            verb_token.idx + len(verb_token.text),
            particle_token.idx + len(particle_token.text),
        )
        text_value = _surface_span_text(doc, start, end, [verb_token, particle_token])
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


def _extract_inventory_phrasals(doc, wordnet_phrasal: set[str]) -> list[ExpressionItem]:
    if not wordnet_phrasal:
        return []
    tokens = [token for token in doc]
    if not tokens:
        return []

    seen: set[tuple[int, int, str]] = set()
    out: list[ExpressionItem] = []
    for verb_index, verb_token in enumerate(tokens):
        if str(getattr(verb_token, "pos_", "")).upper() not in {"VERB", "AUX"}:
            continue
        verb_lemma = _token_lemma_lower(verb_token)
        if not verb_lemma:
            continue

        max_index = min(len(tokens), verb_index + 5)
        for particle_index in range(verb_index + 1, max_index):
            particle_token = tokens[particle_index]
            particle_lemma = _token_lemma_lower(particle_token)
            if particle_lemma not in _PHRASAL_PARTICLES:
                continue
            particle_pos = str(getattr(particle_token, "pos_", "")).upper()
            if particle_pos and particle_pos not in {"ADP", "ADV", "PART"}:
                continue
            lemma = f"{verb_lemma} {particle_lemma}"
            if lemma not in wordnet_phrasal:
                continue

            in_between = tokens[verb_index + 1 : particle_index]
            if any(_is_sentence_break_token(item) for item in in_between):
                break
            if any(str(getattr(item, "pos_", "")).upper() in {"VERB", "AUX"} for item in in_between):
                continue

            start = int(getattr(verb_token, "idx", 0))
            end = int(getattr(particle_token, "idx", 0)) + len(getattr(particle_token, "text", ""))
            marker = (start, end, lemma)
            if marker in seen:
                continue
            seen.add(marker)
            out.append(
                ExpressionItem(
                    text=_surface_span_text(doc, start, end, [verb_token, particle_token]),
                    lemma=lemma,
                    kind="phrasal_verb",
                    start=start,
                    end=end,
                    key=f"{lemma}|phrasal_verb",
                    source="wordnet_window",
                    wordnet=True,
                )
            )
    return out


def _extract_textacy_phrasals(doc, wordnet_phrasal: set[str]) -> list[ExpressionItem]:
    if not wordnet_phrasal or not _env_flag("MORPH_TEXTACY_ENABLED", default=True):
        return []
    try:
        from textacy.extract import matches as textacy_matches
    except Exception:
        return []

    pattern = [
        [
            {"POS": {"IN": ["VERB", "AUX"]}},
            {"OP": "*"},
            {
                "POS": {"IN": ["ADP", "ADV", "PART"]},
                "LEMMA": {"IN": sorted(_PHRASAL_PARTICLES)},
            },
        ]
    ]

    out: list[ExpressionItem] = []
    seen: set[tuple[int, int, str]] = set()
    try:
        spans = textacy_matches.token_matches(doc, pattern)
    except Exception:
        logger.exception("textacy token matching failed")
        return []
    for span in spans:
        item = _phrasal_item_from_token_window(
            doc,
            int(getattr(span, "start", 0)),
            int(getattr(span, "end", 0)),
            wordnet_phrasal,
            source="textacy_matcher",
        )
        if item is None:
            continue
        marker = (item.start, item.end, item.lemma)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


def _extract_phrasemachine_phrasals(doc, wordnet_phrasal: set[str]) -> list[ExpressionItem]:
    if not wordnet_phrasal or not _env_flag("MORPH_PHRASEMACHINE_ENABLED", default=True):
        return []
    try:
        import phrasemachine
    except Exception:
        return []

    tokens = [str(getattr(token, "text", "")) for token in doc]
    pos_tags = [
        str(getattr(token, "tag_", "") or getattr(token, "pos_", ""))
        for token in doc
    ]
    if not tokens:
        return []

    try:
        payload = phrasemachine.get_phrases(
            tokens=tokens,
            postags=pos_tags,
            output="token_spans",
        )
    except TypeError:
        try:
            payload = phrasemachine.get_phrases(tokens=tokens, postags=pos_tags)
        except Exception:
            logger.exception("phrasemachine extraction failed")
            return []
    except Exception:
        logger.exception("phrasemachine extraction failed")
        return []

    raw_spans = payload.get("token_spans", []) if isinstance(payload, dict) else []
    out: list[ExpressionItem] = []
    seen: set[tuple[int, int, str]] = set()
    for span in raw_spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start_index = int(span[0])
            end_index = int(span[1])
        except (TypeError, ValueError):
            continue
        item = _phrasal_item_from_token_window(
            doc,
            start_index,
            end_index,
            wordnet_phrasal,
            source="phrasemachine",
        )
        if item is None:
            continue
        marker = (item.start, item.end, item.lemma)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


def _phrasal_item_from_token_window(
    doc,
    start_index: int,
    end_index: int,
    wordnet_phrasal: set[str],
    *,
    source: str,
) -> ExpressionItem | None:
    try:
        doc_len = len(doc)
    except Exception:
        doc_len = len([token for token in doc])
    if end_index - start_index < 2 or end_index - start_index > 5:
        return None
    if start_index < 0 or end_index > doc_len or start_index >= end_index:
        return None
    tokens = [doc[index] for index in range(start_index, end_index)]
    verb_token = tokens[0]
    particle_token = tokens[-1]
    if str(getattr(verb_token, "pos_", "")).upper() not in {"VERB", "AUX"}:
        return None
    particle_pos = str(getattr(particle_token, "pos_", "")).upper()
    if particle_pos and particle_pos not in {"ADP", "ADV", "PART"}:
        return None
    particle_lemma = _token_lemma_lower(particle_token)
    if particle_lemma not in _PHRASAL_PARTICLES:
        return None
    middle_tokens = tokens[1:-1]
    if any(_is_sentence_break_token(token) for token in middle_tokens):
        return None
    if any(str(getattr(token, "pos_", "")).upper() in {"VERB", "AUX"} for token in middle_tokens):
        return None
    lemma = f"{_token_lemma_lower(verb_token)} {particle_lemma}"
    if lemma not in wordnet_phrasal:
        return None

    start = int(getattr(verb_token, "idx", 0))
    end = int(getattr(particle_token, "idx", 0)) + len(getattr(particle_token, "text", ""))
    return ExpressionItem(
        text=_surface_span_text(doc, start, end, [verb_token, particle_token]),
        lemma=lemma,
        kind="phrasal_verb",
        start=start,
        end=end,
        key=f"{lemma}|phrasal_verb",
        source=source,
        wordnet=True,
    )


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


def _filter_idioms_with_pywsd(doc, items: list[ExpressionItem]) -> list[ExpressionItem]:
    if not items or not _env_flag("MORPH_PYWSD_ENABLED", default=False):
        return items
    components = _load_pywsd_components()
    if components is None:
        return items
    simple_lesk, wn = components
    filtered: list[ExpressionItem] = []
    for item in items:
        if item.kind != "idiom":
            filtered.append(item)
            continue
        if _idiom_supported_by_context(doc, item, simple_lesk, wn):
            filtered.append(item)
    return filtered


def _idiom_supported_by_context(doc, item: ExpressionItem, simple_lesk, wn) -> bool:
    idiom_key = item.lemma.strip().lower().replace(" ", "_")
    if "_" not in idiom_key:
        return True
    try:
        idiom_synsets = wn.synsets(idiom_key)
    except Exception:
        return True
    if not idiom_synsets:
        return True
    idiom_synset_names = {synset.name() for synset in idiom_synsets}

    doc_text = str(getattr(doc, "text", "") or "")
    if not doc_text:
        return True
    if 0 <= item.start < item.end <= len(doc_text):
        phrase_text = doc_text[item.start:item.end]
    else:
        phrase_text = item.text
    normalized_phrase = phrase_text.strip() or item.text.strip()
    if not normalized_phrase:
        return True

    placeholder = idiom_key
    context_text = doc_text.replace(normalized_phrase, placeholder, 1)
    if placeholder not in context_text:
        return True

    for pos_value in _wordnet_pos_candidates(doc, item):
        try:
            if pos_value is None:
                predicted = simple_lesk(context_text, placeholder)
            else:
                predicted = simple_lesk(context_text, placeholder, pos=pos_value)
        except Exception:
            continue
        if predicted is None:
            continue
        predicted_name = str(getattr(predicted, "name", lambda: "")())
        if predicted_name in idiom_synset_names:
            return True
        try:
            lemma_names = {str(name).lower() for name in predicted.lemma_names()}
        except Exception:
            lemma_names = set()
        if idiom_key in lemma_names:
            return True
        return False
    return True


def _wordnet_pos_candidates(doc, item: ExpressionItem) -> list[str | None]:
    default = [None]
    pos_map = {
        "VERB": "v",
        "AUX": "v",
        "NOUN": "n",
        "PROPN": "n",
        "ADJ": "a",
        "ADV": "r",
    }
    candidates: list[str | None] = []
    for token in doc:
        token_start = int(getattr(token, "idx", -1))
        token_end = token_start + len(str(getattr(token, "text", "")))
        if token_start < item.start or token_end > item.end:
            continue
        pos_value = pos_map.get(str(getattr(token, "pos_", "")).upper())
        if pos_value and pos_value not in candidates:
            candidates.append(pos_value)
    return candidates + default


@lru_cache(maxsize=1)
def _load_pywsd_components():
    if not _ensure_pywsd_nltk_data():
        return None
    try:
        from pywsd.lesk import simple_lesk
    except Exception:
        return None
    wn = _load_wordnet_corpus()
    if wn is None:
        return None
    return simple_lesk, wn


def _ensure_pywsd_nltk_data() -> bool:
    try:
        import nltk
    except Exception:
        return False

    required = [
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
        "corpora/wordnet",
    ]
    missing: list[str] = []
    for resource_path in required:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(resource_path)
    if not missing:
        return True

    if not _env_flag("MORPH_PYWSD_AUTO_DOWNLOAD", default=True):
        logger.info(
            "PyWSD resources are missing (%s) and MORPH_PYWSD_AUTO_DOWNLOAD=0",
            ",".join(missing),
        )
        return False

    resource_names = [item.split("/", 1)[1] for item in missing]
    for name in resource_names:
        try:
            if not nltk.download(name, quiet=True):
                return False
        except Exception:
            logger.exception("Failed to download PyWSD NLTK resource: %s", name)
            return False
    return True


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


def _surface_span_text(doc, start: int, end: int, fallback_tokens) -> str:
    doc_text = str(getattr(doc, "text", "") or "")
    if doc_text and 0 <= start < end <= len(doc_text):
        return doc_text[start:end].strip()
    return _span_like_text(fallback_tokens)


def _is_sentence_break_token(token) -> bool:
    raw = str(getattr(token, "text", "")).strip()
    if not raw:
        return False
    if any(ch in ".!?;:" for ch in raw):
        return True
    return bool(getattr(token, "is_punct", False))


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

    for model_name in _spacy_model_candidates():
        try:
            return spacy.load(model_name, disable=["ner", "textcat"])
        except Exception:
            continue

    if _env_flag("SPACY_EN_MODEL_AUTO_DOWNLOAD", default=True):
        try:
            from spacy.cli import download as spacy_download

            for model_name in _spacy_auto_download_candidates():
                try:
                    logger.info("Downloading spaCy model %s for expression matching", model_name)
                    spacy_download(model_name)
                    return spacy.load(model_name, disable=["ner", "textcat"])
                except Exception:
                    continue
        except Exception:
            logger.exception("spaCy model auto-download failed")
    logger.info("Expression matcher disabled: spaCy model is unavailable")
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
        if not _env_flag("WORDNET_AUTO_DOWNLOAD", default=True):
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


def _spacy_auto_download_candidates() -> list[str]:
    raw = str(os.getenv("SPACY_EN_AUTO_DOWNLOAD_MODELS", "") or "").strip()
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if values:
            return values
    return ["en_core_web_sm"]


def _env_flag(name: str, *, default: bool = False) -> bool:
    default_value = "1" if default else "0"
    value = str(os.getenv(name, default_value) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}
