from kokoro_tts.domain.morphology import RawToken, TokenAnnotation, analyze_english_text


def test_analyze_english_text_empty():
    assert analyze_english_text("") == {"language": "en", "items": []}


def test_analyze_english_text_ignores_punctuation_and_keeps_offsets():
    text = "Cats, run!"
    tokens = [
        RawToken("Cats", 0, 4, False, False, False),
        RawToken(",", 4, 5, False, True, False),
        RawToken(" ", 5, 6, True, False, False),
        RawToken("run", 6, 9, False, False, False),
        RawToken("!", 9, 10, False, True, False),
    ]

    def tokenizer(_: str):
        return tokens

    def annotator(_: str, raw_tokens):
        assert [token.text for token in raw_tokens] == ["Cats", ",", "run", "!"]
        return [
            TokenAnnotation("cat", "NOUN", {"Number": "Plur"}),
            TokenAnnotation(",", "PUNCT", {}),
            TokenAnnotation("run", "VERB", {"Tense": "Pres"}),
            TokenAnnotation("!", "PUNCT", {}),
        ]

    payload = analyze_english_text(text, tokenizer=tokenizer, annotator=annotator)
    assert payload == {
        "language": "en",
        "items": [
            {
                "token": "Cats",
                "lemma": "cat",
                "upos": "NOUN",
                "feats": {"Number": "Plur"},
                "start": 0,
                "end": 4,
                "key": "cat|noun",
            },
            {
                "token": "run",
                "lemma": "run",
                "upos": "VERB",
                "feats": {"Tense": "Pres"},
                "start": 6,
                "end": 9,
                "key": "run|verb",
            },
        ],
    }


def test_analyze_english_text_fallback_num_sym_x():
    text = "10 + foo ..."
    tokens = [
        RawToken("10", 0, 2, False, False, True),
        RawToken("+", 3, 4, False, False, False),
        RawToken("foo", 5, 8, False, False, False),
        RawToken("...", 9, 12, False, True, False),
    ]

    payload = analyze_english_text(
        text,
        tokenizer=lambda _: tokens,
        annotator=lambda *_: None,
    )
    assert payload["language"] == "en"
    assert payload["items"] == [
        {
            "token": "10",
            "lemma": "10",
            "upos": "NUM",
            "feats": {},
            "start": 0,
            "end": 2,
            "key": "10|num",
        },
        {
            "token": "+",
            "lemma": "+",
            "upos": "SYM",
            "feats": {},
            "start": 3,
            "end": 4,
            "key": "+|sym",
        },
        {
            "token": "foo",
            "lemma": "foo",
            "upos": "X",
            "feats": {},
            "start": 5,
            "end": 8,
            "key": "foo|x",
        },
    ]


def test_analyze_english_text_maps_aux_to_verb_and_normalizes_feats():
    tokens = [RawToken("Runs", 0, 4, False, False, False)]
    payload = analyze_english_text(
        "Runs",
        tokenizer=lambda _: tokens,
        annotator=lambda *_: [TokenAnnotation(" Run ", "AUX", {"Person": 3, "Number": "Sing"})],
    )
    assert payload["items"] == [
        {
            "token": "Runs",
            "lemma": "Run",
            "upos": "VERB",
            "feats": {"Person": "3", "Number": "Sing"},
            "start": 0,
            "end": 4,
            "key": "run|verb",
        }
    ]


def test_analyze_english_text_uses_fallback_when_annotation_mismatch():
    tokens = [RawToken("foo", 0, 3, False, False, False)]
    payload = analyze_english_text(
        "foo",
        tokenizer=lambda _: tokens,
        annotator=lambda *_: [],
    )
    assert payload["items"][0]["upos"] == "X"
    assert payload["items"][0]["key"] == "foo|x"


def test_analyze_english_text_supports_diverse_upos_word_types():
    tokens = [
        RawToken("Wow", 0, 3, False, False, False),
        RawToken("I", 4, 5, False, False, False),
        RawToken("can", 6, 9, False, False, False),
        RawToken("not", 10, 13, False, False, False),
        RawToken("write", 14, 19, False, False, False),
        RawToken("2", 20, 21, False, False, True),
        RawToken("the", 22, 25, False, False, False),
        RawToken("clean", 26, 31, False, False, False),
        RawToken("tests", 32, 37, False, False, False),
        RawToken("in", 38, 40, False, False, False),
        RawToken("Python", 41, 47, False, False, False),
        RawToken("and", 48, 51, False, False, False),
        RawToken("because", 52, 59, False, False, False),
        RawToken("they", 60, 64, False, False, False),
        RawToken("work", 65, 69, False, False, False),
        RawToken(".", 69, 70, False, True, False),
    ]

    annotations = [
        TokenAnnotation("wow", "INTJ", {}),
        TokenAnnotation("I", "PRON", {}),
        TokenAnnotation("can", "VERB", {}),
        TokenAnnotation("not", "PART", {}),
        TokenAnnotation("write", "VERB", {}),
        TokenAnnotation("2", "NUM", {}),
        TokenAnnotation("the", "DET", {}),
        TokenAnnotation("clean", "ADJ", {}),
        TokenAnnotation("test", "NOUN", {}),
        TokenAnnotation("in", "ADP", {}),
        TokenAnnotation("Python", "PROPN", {}),
        TokenAnnotation("and", "CCONJ", {}),
        TokenAnnotation("because", "SCONJ", {}),
        TokenAnnotation("they", "PRON", {}),
        TokenAnnotation("work", "VERB", {}),
        TokenAnnotation(".", "PUNCT", {}),
    ]

    payload = analyze_english_text(
        "Wow I can not write 2 the clean tests in Python and because they work.",
        tokenizer=lambda _: tokens,
        annotator=lambda *_: annotations,
    )

    assert [item["upos"] for item in payload["items"]] == [
        "INTJ",
        "PRON",
        "VERB",
        "PART",
        "VERB",
        "NUM",
        "DET",
        "ADJ",
        "NOUN",
        "ADP",
        "PROPN",
        "CCONJ",
        "SCONJ",
        "PRON",
        "VERB",
    ]
