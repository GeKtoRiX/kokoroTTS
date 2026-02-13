import os
import sys
import types

from kokoro_tts.domain import expressions as e
from kokoro_tts.domain.expressions import ExpressionItem


class _FakeToken:
    def __init__(self, text, lemma, pos, dep, idx, i):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.dep_ = dep
        self.idx = idx
        self.i = i


class _FakeSpan:
    def __init__(self, tokens):
        self._tokens = tokens
        self.text = " ".join(token.text for token in tokens)
        self.start_char = tokens[0].idx
        self.end_char = tokens[-1].idx + len(tokens[-1].text)

    def __iter__(self):
        return iter(self._tokens)


class _FakeDoc:
    def __init__(self, tokens):
        self.tokens = tokens

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _FakeSpan(self.tokens[item.start : item.stop])
        return self.tokens[item]

    def __iter__(self):
        return iter(self.tokens)


def test_extract_english_expressions_returns_empty_without_model(monkeypatch):
    monkeypatch.setattr(e, "_load_expression_spacy_model", lambda: None)
    assert e.extract_english_expressions("look up") == []


def test_extract_english_expressions_merges_and_sorts(monkeypatch):
    doc = _FakeDoc([_FakeToken("look", "look", "VERB", "ROOT", 0, 0)])

    monkeypatch.setattr(e, "_load_expression_spacy_model", lambda: (lambda _text: doc))
    monkeypatch.setattr(e, "_load_wordnet_phrase_inventory", lambda: ({"look up"}, {"kick the bucket"}))
    monkeypatch.setattr(
        e,
        "_extract_dependency_phrasals",
        lambda _doc, _inv: [
            ExpressionItem(
                text="look up",
                lemma="look up",
                kind="phrasal_verb",
                start=0,
                end=7,
                key="look up|phrasal_verb",
                source="dep",
                wordnet=True,
            )
        ],
    )
    monkeypatch.setattr(
        e,
        "_extract_wordnet_idioms",
        lambda _doc, _inv: [
            ExpressionItem(
                text="kick the bucket",
                lemma="kick the bucket",
                kind="idiom",
                start=10,
                end=25,
                key="kick the bucket|idiom",
                source="wn",
                wordnet=True,
            )
        ],
    )

    items = e.extract_english_expressions("look up and then kick the bucket")
    assert [item["kind"] for item in items] == ["phrasal_verb", "idiom"]
    assert items[0]["wordnet"] is True


def test_extract_dependency_phrasals_filters_particles(monkeypatch):
    tokens = [
        _FakeToken("look", "look", "VERB", "ROOT", 0, 0),
        _FakeToken("up", "up", "ADP", "prt", 5, 1),
        _FakeToken("run", "run", "VERB", "ROOT", 9, 2),
        _FakeToken("quickly", "quickly", "ADV", "advmod", 13, 3),
    ]
    doc = _FakeDoc(tokens)
    monkeypatch.setattr(
        e,
        "_load_dependency_matcher",
        lambda: (lambda _doc: [("PHRASAL_VERB", [0, 1]), ("PHRASAL_VERB", [2, 3])]),
    )

    items = e._extract_dependency_phrasals(doc, {"look up"})
    assert len(items) == 1
    assert items[0].lemma == "look up"
    assert items[0].wordnet is True


def test_extract_wordnet_idioms_from_lemma_matcher(monkeypatch):
    tokens = [
        _FakeToken("kicked", "kick", "VERB", "ROOT", 0, 0),
        _FakeToken("the", "the", "DET", "det", 7, 1),
        _FakeToken("bucket", "bucket", "NOUN", "obj", 11, 2),
    ]
    doc = _FakeDoc(tokens)
    monkeypatch.setattr(e, "_load_idiom_lemma_matcher", lambda: (lambda _doc: [("WORDNET_IDIOM", 0, 3)]))

    items = e._extract_wordnet_idioms(doc, {"kick the bucket"})
    assert len(items) == 1
    assert items[0].kind == "idiom"
    assert items[0].lemma == "kick the bucket"


def test_map_match_tokens_uses_fallback_positions():
    tokens = [
        _FakeToken("alpha", "alpha", "X", "dep", 0, 0),
        _FakeToken("beta", "beta", "X", "dep", 6, 1),
    ]
    doc = _FakeDoc(tokens)
    mapped = e._map_match_tokens(doc, [0, 1])
    assert mapped["verb"] is tokens[0]
    assert mapped["particle"] is tokens[1]


def test_load_wordnet_phrase_inventory_filters_phrases(monkeypatch):
    class Synset:
        def __init__(self, pos, names):
            self._pos = pos
            self._names = names

        def pos(self):
            return self._pos

        def lemma_names(self):
            return self._names

    class WordNet:
        def all_synsets(self):
            return [
                Synset("v", ["look_up", "run"]),
                Synset("n", ["kick_the_bucket"]),
                Synset("n", ["123_invalid"]),
            ]

    monkeypatch.setattr(e, "_load_wordnet_corpus", lambda: WordNet())
    e._load_wordnet_phrase_inventory.cache_clear()
    phrasal, idioms = e._load_wordnet_phrase_inventory()
    assert "look up" in phrasal
    assert "kick the bucket" in idioms
    e._load_wordnet_phrase_inventory.cache_clear()


def test_load_expression_spacy_model_respects_auto_download_flag(monkeypatch):
    class FakeSpacy:
        def load(self, _name, disable=None):
            _ = disable
            raise RuntimeError("missing")

    monkeypatch.setitem(sys.modules, "spacy", FakeSpacy())
    monkeypatch.setenv("SPACY_EN_MODEL_AUTO_DOWNLOAD", "0")
    e._load_expression_spacy_model.cache_clear()
    assert e._load_expression_spacy_model() is None
    e._load_expression_spacy_model.cache_clear()


def test_load_expression_spacy_model_import_failure(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spacy":
            raise ImportError("missing spacy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    e._load_expression_spacy_model.cache_clear()
    assert e._load_expression_spacy_model() is None
    e._load_expression_spacy_model.cache_clear()


def test_load_expression_spacy_model_auto_download_success(monkeypatch):
    spacy_module = types.ModuleType("spacy")
    cli_module = types.ModuleType("spacy.cli")
    calls = {"load": 0, "download": 0}

    def load(_name, disable=None):
        _ = disable
        calls["load"] += 1
        if calls["load"] == 1:
            raise RuntimeError("model missing")
        return "loaded-model"

    def download(_name):
        calls["download"] += 1

    spacy_module.load = load  # type: ignore[attr-defined]
    cli_module.download = download  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "spacy", spacy_module)
    monkeypatch.setitem(sys.modules, "spacy.cli", cli_module)
    monkeypatch.setenv("SPACY_EN_MODEL_AUTO_DOWNLOAD", "1")
    e._load_expression_spacy_model.cache_clear()
    assert e._load_expression_spacy_model() == "loaded-model"
    assert calls["download"] == 1
    e._load_expression_spacy_model.cache_clear()


def test_load_dependency_matcher_and_idiom_matcher_success(monkeypatch):
    class NLP:
        vocab = "vocab"

    class DependencyMatcher:
        def __init__(self, vocab):
            self.vocab = vocab
            self.added = []

        def add(self, name, patterns):
            self.added.append((name, patterns))

    class Matcher:
        def __init__(self, vocab, validate=False):
            self.vocab = vocab
            self.validate = validate
            self.added = []

        def add(self, name, patterns):
            self.added.append((name, patterns))

    matcher_module = types.ModuleType("spacy.matcher")
    matcher_module.DependencyMatcher = DependencyMatcher  # type: ignore[attr-defined]
    matcher_module.Matcher = Matcher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy.matcher", matcher_module)

    monkeypatch.setattr(e, "_load_expression_spacy_model", lambda: NLP())
    monkeypatch.setattr(e, "_load_wordnet_phrase_inventory", lambda: (set(), {"kick the bucket"}))

    e._load_dependency_matcher.cache_clear()
    dep_matcher = e._load_dependency_matcher()
    assert dep_matcher is not None
    assert dep_matcher.added

    e._load_idiom_lemma_matcher.cache_clear()
    idiom_matcher = e._load_idiom_lemma_matcher()
    assert idiom_matcher is not None
    assert idiom_matcher.added


def test_load_idiom_matcher_returns_none_for_empty_inventory(monkeypatch):
    class NLP:
        vocab = "vocab"

    monkeypatch.setattr(e, "_load_expression_spacy_model", lambda: NLP())
    monkeypatch.setattr(e, "_load_wordnet_phrase_inventory", lambda: (set(), set()))
    e._load_idiom_lemma_matcher.cache_clear()
    assert e._load_idiom_lemma_matcher() is None


def test_load_wordnet_phrase_inventory_handles_synset_errors(monkeypatch):
    class WordNet:
        def all_synsets(self):
            raise RuntimeError("failed")

    monkeypatch.setattr(e, "_load_wordnet_corpus", lambda: WordNet())
    e._load_wordnet_phrase_inventory.cache_clear()
    phrasal, idioms = e._load_wordnet_phrase_inventory()
    assert phrasal == set()
    assert idioms == set()
    e._load_wordnet_phrase_inventory.cache_clear()


def test_load_wordnet_corpus_paths_and_download_modes(monkeypatch, tmp_path):
    nltk_module = types.ModuleType("nltk")
    corpus_module = types.ModuleType("nltk.corpus")

    class WordNet:
        def __init__(self, fail_once=False):
            self.fail_once = fail_once
            self.calls = 0

        def ensure_loaded(self):
            self.calls += 1
            if self.fail_once and self.calls == 1:
                raise LookupError("missing")
            return None

    wn = WordNet()
    nltk_module.data = types.SimpleNamespace(path=[])
    nltk_module.download = lambda *_args, **_kwargs: True  # type: ignore[attr-defined]
    corpus_module.wordnet = wn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nltk", nltk_module)
    monkeypatch.setitem(sys.modules, "nltk.corpus", corpus_module)
    monkeypatch.setenv("WORDNET_DATA_DIR", str(tmp_path / "nltk_data"))
    e._load_wordnet_corpus.cache_clear()
    assert e._load_wordnet_corpus() is wn
    e._load_wordnet_corpus.cache_clear()

    wn_missing = WordNet(fail_once=True)
    corpus_module.wordnet = wn_missing  # type: ignore[attr-defined]
    monkeypatch.setenv("WORDNET_AUTO_DOWNLOAD", "0")
    assert e._load_wordnet_corpus() is None
    e._load_wordnet_corpus.cache_clear()

    wn_download = WordNet(fail_once=True)
    corpus_module.wordnet = wn_download  # type: ignore[attr-defined]
    monkeypatch.setenv("WORDNET_AUTO_DOWNLOAD", "1")
    assert e._load_wordnet_corpus() is wn_download
    e._load_wordnet_corpus.cache_clear()


def test_load_wordnet_corpus_download_failure(monkeypatch, tmp_path):
    nltk_module = types.ModuleType("nltk")
    corpus_module = types.ModuleType("nltk.corpus")

    class WordNet:
        def ensure_loaded(self):
            raise LookupError("missing")

    def fail_download(*_args, **_kwargs):
        raise RuntimeError("download failed")

    wn = WordNet()
    nltk_module.data = types.SimpleNamespace(path=[])
    nltk_module.download = fail_download  # type: ignore[attr-defined]
    corpus_module.wordnet = wn  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "nltk", nltk_module)
    monkeypatch.setitem(sys.modules, "nltk.corpus", corpus_module)
    monkeypatch.setenv("WORDNET_AUTO_DOWNLOAD", "1")
    monkeypatch.setenv("WORDNET_DATA_DIR", str(tmp_path / "nltk_data"))
    e._load_wordnet_corpus.cache_clear()
    assert e._load_wordnet_corpus() is None
    e._load_wordnet_corpus.cache_clear()


def test_load_dependency_and_idiom_matcher_without_model(monkeypatch):
    monkeypatch.setattr(e, "_load_expression_spacy_model", lambda: None)
    e._load_dependency_matcher.cache_clear()
    e._load_idiom_lemma_matcher.cache_clear()
    assert e._load_dependency_matcher() is None
    assert e._load_idiom_lemma_matcher() is None

    # Keep the environment clean for other tests.
    os.environ.pop("SPACY_EN_MODEL_AUTO_DOWNLOAD", None)
