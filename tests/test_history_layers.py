from pathlib import Path

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
    assert logger.infos
