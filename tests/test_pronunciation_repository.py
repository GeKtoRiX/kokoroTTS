import json
from pathlib import Path

from kokoro_tts.storage.pronunciation_repository import PronunciationRepository


class _Logger:
    def __init__(self):
        self.warnings = []
        self.exceptions = []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)


def test_pronunciation_repository_roundtrip_and_aliases(tmp_path: Path):
    path = tmp_path / "rules.json"
    repo = PronunciationRepository(str(path), logger_instance=_Logger())

    saved = repo.save_rules(
        {
            "a": {"OpenAI": "oʊpənˈeɪ aɪ"},
            "en-us": {"Kokoro": "kˈOkəɹO"},
            "unknown": {"word": "x"},
        }
    )

    assert "a" in saved
    assert saved["a"]["OpenAI"] == "oʊpənˈeɪ aɪ"
    assert saved["a"]["Kokoro"] == "kˈOkəɹO"
    assert "unknown" not in saved

    loaded = repo.load_rules()
    assert loaded == saved
    assert path.is_file()

    json_payload = repo.to_pretty_json()
    parsed = json.loads(json_payload)
    assert parsed == saved


def test_pronunciation_repository_parse_and_file_import(tmp_path: Path):
    path = tmp_path / "rules.json"
    source_path = tmp_path / "import.json"
    source_path.write_text(
        '{"b": {"OpenAI": "əʊpənˈeɪ aɪ"}, "fr-fr": {"bonjour": "bɔ̃ʒuʁ"}}',
        encoding="utf-8",
    )
    repo = PronunciationRepository(str(path), logger_instance=_Logger())

    parsed = repo.parse_rules_json(source_path.read_text(encoding="utf-8"))
    assert parsed["b"]["OpenAI"] == "əʊpənˈeɪ aɪ"
    assert parsed["f"]["bonjour"] == "bɔ̃ʒuʁ"

    imported = repo.load_rules_from_file(str(source_path))
    assert imported == parsed


def test_pronunciation_repository_rejects_invalid_json(tmp_path: Path):
    repo = PronunciationRepository(str(tmp_path / "rules.json"), logger_instance=_Logger())

    try:
        repo.parse_rules_json("{ invalid")
    except ValueError as exc:
        assert "Invalid JSON" in str(exc)
    else:
        raise AssertionError("Expected invalid JSON error")
