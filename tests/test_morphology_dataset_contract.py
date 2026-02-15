import os

os.environ.setdefault("KOKORO_SKIP_APP_INIT", "1")

import app
from kokoro_tts.domain.morphology_datasets import (
    morphology_primary_key,
    normalize_morphology_dataset,
)
from kokoro_tts.ui import gradio_app


def test_dataset_aliases_are_canonical_across_layers():
    expected = {
        "": "occurrences",
        None: "occurrences",
        "occurrence": "occurrences",
        "occurrences": "occurrences",
        "token_occurrences": "occurrences",
        "lexeme": "lexemes",
        "lexemes": "lexemes",
        "expression": "expressions",
        "expressions": "expressions",
        "mwe": "expressions",
        "mwes": "expressions",
        "idioms": "expressions",
        "review": "reviews",
        "reviews": "reviews",
        "lm_reviews": "reviews",
        "unknown": "occurrences",
    }
    for raw_value, canonical in expected.items():
        assert normalize_morphology_dataset(raw_value) == canonical
        assert app._normalize_morph_dataset(raw_value) == canonical
        assert gradio_app._normalize_morph_dataset(raw_value) == canonical


def test_primary_key_contract_by_dataset():
    assert morphology_primary_key("lexemes") == "dedup_key"
    assert morphology_primary_key("lexeme") == "dedup_key"
    assert morphology_primary_key("occurrences") == "id"
    assert morphology_primary_key("reviews") == "id"
    assert morphology_primary_key("unknown") == "id"
