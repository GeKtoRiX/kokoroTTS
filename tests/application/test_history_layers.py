from pathlib import Path
from datetime import datetime

from kokoro_tts.application.history_service import HistoryService
from kokoro_tts.storage.history_repository import HistoryRepository


class _Logger:
    def __init__(self):
        self.warnings = []
        self.exceptions = []
        self.infos = []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)

    def info(self, message, *args):
        self.infos.append(message % args if args else message)


def test_history_repository_deletes_only_files_inside_output_dir(tmp_path: Path):
    logger = _Logger()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    inside = output_dir / "inside.wav"
    outside = tmp_path / "outside.wav"
    inside.write_bytes(b"a")
    outside.write_bytes(b"b")

    repository = HistoryRepository(str(output_dir), logger)
    deleted = repository.delete_paths([str(inside), str(outside), ""])

    assert deleted == 1
    assert inside.exists() is False
    assert outside.exists() is True
    assert logger.warnings
    assert logger.exceptions == []


def test_history_repository_deletes_only_records_in_current_date_folder(tmp_path: Path):
    logger = _Logger()
    output_dir = tmp_path / "outputs"
    date_dir = output_dir / datetime.now().strftime("%Y-%m-%d")
    records_dir = date_dir / "records"
    vocab_dir = date_dir / "vocabulary"
    records_dir.mkdir(parents=True)
    vocab_dir.mkdir(parents=True)

    today_audio = records_dir / "today.wav"
    today_csv = vocab_dir / "today.csv"
    today_audio.write_bytes(b"a")
    today_csv.write_bytes(b"b")

    old_dir = output_dir / "2000-01-01" / "records"
    old_dir.mkdir(parents=True)
    old_audio = old_dir / "old.wav"
    old_audio.write_bytes(b"c")

    repository = HistoryRepository(str(output_dir), logger)
    deleted = repository.delete_current_date_files()

    assert deleted == 1
    assert today_audio.exists() is False
    assert today_csv.exists() is True
    assert old_audio.exists() is True


def test_history_service_updates_and_truncates_history(tmp_path: Path):
    logger = _Logger()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    path_1 = output_dir / "one.wav"
    path_2 = output_dir / "two.wav"
    path_1.write_bytes(b"a")
    path_2.write_bytes(b"b")

    class State:
        last_saved_paths = [str(path_1), str(path_2)]

    service = HistoryService(
        history_limit=2,
        repository=HistoryRepository(str(output_dir), logger),
        state=State(),
        logger=logger,
    )

    updated = service.update_history(["existing.wav"])

    assert updated == [str(path_1), str(path_2)]


def test_history_service_clear_history_resets_state(tmp_path: Path):
    logger = _Logger()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    audio = output_dir / "to_delete.wav"
    audio.write_bytes(b"abc")
    date_dir = output_dir / datetime.now().strftime("%Y-%m-%d") / "records"
    date_dir.mkdir(parents=True)
    today_generated = date_dir / "today_generated.wav"
    today_generated.write_bytes(b"xyz")
    vocabulary_dir = output_dir / datetime.now().strftime("%Y-%m-%d") / "vocabulary"
    vocabulary_dir.mkdir(parents=True)
    vocab_file = vocabulary_dir / "today.csv"
    vocab_file.write_bytes(b"vocab")

    class State:
        last_saved_paths = [str(audio)]

    service = HistoryService(
        history_limit=5,
        repository=HistoryRepository(str(output_dir), logger),
        state=State(),
        logger=logger,
    )

    cleared = service.clear_history([str(audio)])

    assert cleared == []
    assert service.state.last_saved_paths == []
    assert today_generated.exists() is False
    assert vocab_file.exists() is True
    assert logger.infos


def test_history_service_remove_selected_history_deletes_only_selected(tmp_path: Path):
    logger = _Logger()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    first = output_dir / "first.wav"
    second = output_dir / "second.wav"
    third = output_dir / "third.wav"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    third.write_bytes(b"c")

    class State:
        last_saved_paths = [str(first), str(second), str(third)]

    service = HistoryService(
        history_limit=10,
        repository=HistoryRepository(str(output_dir), logger),
        state=State(),
        logger=logger,
    )

    updated = service.remove_selected_history(
        [str(first), str(second), str(third)],
        [1, 2],
    )

    assert updated == [str(first)]
    assert first.exists() is True
    assert second.exists() is False
    assert third.exists() is False
    assert service.state.last_saved_paths == [str(first)]
