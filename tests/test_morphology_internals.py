import sys
from types import SimpleNamespace

from kokoro_tts.domain import morphology as m
from kokoro_tts.domain.morphology import RawToken, TokenAnnotation


def test_regex_tokenize_and_symbol_punct_helpers():
    tokens = m._regex_tokenize("Hello, 10 + world!")
    values = [token.text for token in tokens]
    assert "Hello" in values
    assert "," in values
    assert "+" in values
    assert m._is_symbol_token("+") is True
    assert m._is_punct_text("...") is True
    assert m._parse_ud_feats("Number=Plur|Case=Nom") == {"Number": "Plur", "Case": "Nom"}


def test_default_annotate_falls_back_when_annotation_missing(monkeypatch):
    tokens = [RawToken("foo", 0, 3, False, False, False)]
    monkeypatch.setattr(m, "_annotate_with_stanza", lambda _tokens: None)
    monkeypatch.setattr(m, "_annotate_with_spacy", lambda _text, _tokens: None)

    annotated = m._default_annotate("foo", tokens)
    assert annotated == [TokenAnnotation(lemma="foo", upos="X", feats={})]


def test_annotate_with_stanza_handles_mismatch_and_success(monkeypatch):
    class Word:
        def __init__(self, lemma, upos, feats):
            self.lemma = lemma
            self.upos = upos
            self.feats = feats

    tokens = [RawToken("Cats", 0, 4, False, False, False)]

    class PipelineOk:
        def __call__(self, payload):
            _ = payload
            return SimpleNamespace(sentences=[SimpleNamespace(words=[Word("cat", "NOUN", "Number=Plur")])])

    monkeypatch.setattr(m, "_load_stanza_pipeline", lambda: PipelineOk())
    result = m._annotate_with_stanza(tokens)
    assert result == [TokenAnnotation(lemma="cat", upos="NOUN", feats={"Number": "Plur"})]

    class PipelineMismatch:
        def __call__(self, payload):
            _ = payload
            return SimpleNamespace(sentences=[SimpleNamespace(words=[])])

    monkeypatch.setattr(m, "_load_stanza_pipeline", lambda: PipelineMismatch())
    assert m._annotate_with_stanza(tokens) is None


def test_annotate_with_spacy_maps_tokens_by_offsets(monkeypatch):
    class Morph:
        def to_dict(self):
            return {"Number": "Sing"}

    class Token:
        def __init__(self, text, idx, is_space=False):
            self.text = text
            self.idx = idx
            self.is_space = is_space
            self.lemma_ = text.lower()
            self.pos_ = "NOUN"
            self.morph = Morph()

    class NLP:
        def __call__(self, text):
            _ = text
            return [
                Token("alpha", 0),
                Token(" ", 5, is_space=True),
                Token("beta", 6),
            ]

    monkeypatch.setattr(m, "_load_spacy_model", lambda: NLP())
    raw_tokens = [
        RawToken("alpha", 0, 5, False, False, False),
        RawToken("beta", 6, 10, False, False, False),
    ]
    annotated = m._annotate_with_spacy("alpha beta", raw_tokens)
    assert [item.lemma for item in annotated] == ["alpha", "beta"]


def test_default_tokenize_uses_regex_when_spacy_unavailable(monkeypatch):
    monkeypatch.setattr(m, "_load_spacy_model", lambda: None)
    tokens = m._default_tokenize("Hello world")
    assert [token.text for token in tokens if not token.is_space] == ["Hello", "world"]


def test_load_spacy_model_fallback_to_blank(monkeypatch):
    m._load_spacy_model.cache_clear()

    class FakeSpacy:
        def load(self, _name, disable=None):
            _ = disable
            raise RuntimeError("missing model")

        def blank(self, lang):
            return f"blank:{lang}"

    monkeypatch.setitem(sys.modules, "spacy", FakeSpacy())
    model = m._load_spacy_model()
    assert model == "blank:en"
    m._load_spacy_model.cache_clear()


def test_load_stanza_pipeline_returns_none_on_error(monkeypatch):
    m._load_stanza_pipeline.cache_clear()

    class FakeStanza:
        class Pipeline:
            def __init__(self, **kwargs):
                _ = kwargs
                raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "stanza", FakeStanza())
    assert m._load_stanza_pipeline() is None
    m._load_stanza_pipeline.cache_clear()


def test_merge_annotation_sources_prefers_informative_upos_and_lemma():
    tokens = [RawToken("looked", 0, 6, False, False, False)]
    first = [TokenAnnotation("looked", "X", {})]
    second = [TokenAnnotation("look", "VERB", {"Tense": "Past"})]

    merged = m._merge_annotation_sources(tokens, first, second)
    assert merged is not None
    assert merged[0] == TokenAnnotation(lemma="look", upos="VERB", feats={"Tense": "Past"})


def test_default_annotate_uses_flair_backfill_for_uninformative_tags(monkeypatch):
    tokens = [RawToken("alpha", 0, 5, False, False, False)]
    monkeypatch.setattr(
        m,
        "_annotate_with_stanza",
        lambda _tokens: [TokenAnnotation("alpha", "X", {})],
    )
    monkeypatch.setattr(
        m,
        "_annotate_with_spacy",
        lambda _text, _tokens: [TokenAnnotation("alpha", "X", {})],
    )
    monkeypatch.setattr(
        m,
        "_annotate_with_flair",
        lambda _tokens: [TokenAnnotation("alpha", "NOUN", {})],
    )

    annotated = m._default_annotate("alpha", tokens)
    assert annotated[0].upos == "NOUN"


def test_normalize_flair_upos_maps_penn_and_subordinator_hint():
    because = RawToken("because", 0, 7, False, False, False)
    assert m._normalize_flair_upos("IN", because) == "SCONJ"

    noun = RawToken("cats", 0, 4, False, False, False)
    assert m._normalize_flair_upos("NNS", noun) == "NOUN"


def test_load_flair_tagger_respects_disable_flag(monkeypatch):
    monkeypatch.setenv("MORPH_FLAIR_ENABLED", "0")
    m._load_flair_tagger.cache_clear()
    assert m._load_flair_tagger() is None
    m._load_flair_tagger.cache_clear()
