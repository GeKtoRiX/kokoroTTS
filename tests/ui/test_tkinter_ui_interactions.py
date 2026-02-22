import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tkinter as tk
from tkinter import ttk

from kokoro_tts.config import AppConfig
import kokoro_tts.ui.features.audio_player_feature as audio_feature_mod
from kokoro_tts.ui import tkinter_app as tkapp_mod
from kokoro_tts.ui.tkinter_app import VlcAudioBackend, create_tkinter_app


class _InlineThread:
    def __init__(self, target, daemon=True):
        self._target = target
        self._alive = False
        self.daemon = daemon

    def start(self):
        self._alive = True
        self._target()
        self._alive = False

    def join(self, timeout=None):
        _ = timeout

    def is_alive(self):
        return self._alive


class _Logger:
    def __init__(self):
        self.debugs = []
        self.exceptions = []

    def debug(self, message, *args):
        self.debugs.append(message % args if args else message)

    def exception(self, message, *args):
        self.exceptions.append(message % args if args else message)


class _HistoryService:
    def update_history(self, history):
        return list(history or [])

    def clear_history(self, _history):
        return []

    def remove_selected_history(self, history, selected_indices):
        selected = {int(index) for index in selected_indices or []}
        return [value for index, value in enumerate(list(history or [])) if index not in selected]


def _build_config(tmp_path: Path) -> AppConfig:
    logs = tmp_path / "logs"
    outputs = tmp_path / "outputs"
    logs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        log_level="INFO",
        file_log_level="DEBUG",
        log_dir=str(logs),
        log_file=str(logs / "app.log"),
        repo_id="repo/x",
        output_dir=str(outputs),
        output_dir_abs=str(outputs.resolve()),
        app_state_path=str(tmp_path / "data" / "app_state.json"),
        max_chunk_chars=250,
        history_limit=3,
        normalize_times=True,
        normalize_numbers=True,
        default_output_format="wav",
        default_concurrency_limit=None,
        log_segment_every=5,
        morph_db_enabled=True,
        morph_db_path=str(tmp_path / "data" / "morph.sqlite3"),
        morph_db_table_prefix="morph_",
        space_id="",
        is_duplicate=True,
        char_limit=None,
    )


def _sync_threaded(monkeypatch, ui_app):
    monkeypatch.setattr(
        ui_app,
        "_threaded",
        lambda work, on_success=None: on_success(work()) if on_success is not None else work(),
    )


def _normalized_color(value: str) -> str:
    return str(value or "").strip().lower()


def _assert_menu_palette(menu: tk.Menu, *, surface: str, accent: str) -> None:
    assert _normalized_color(menu.cget("background")) == _normalized_color(surface)
    assert _normalized_color(menu.cget("activebackground")) == _normalized_color(accent)
    assert _normalized_color(menu.cget("activeforeground")) == _normalized_color(surface)


@pytest.fixture
def ui_app(tmp_path: Path):
    logger = _Logger()
    os.environ.pop("TCL_LIBRARY", None)
    os.environ.pop("TK_LIBRARY", None)

    def _generate_first(*_args, **_kwargs):
        return (24000, [0.1, 0.2]), "tokens"

    def _tokenize_first(*_args, **_kwargs):
        return "tok"

    def _generate_all(*_args, **_kwargs):
        yield 24000, [0.1]

    app_instance = create_tkinter_app(
        config=_build_config(tmp_path),
        cuda_available=True,
        logger=logger,
        generate_first=_generate_first,
        tokenize_first=_tokenize_first,
        generate_all=_generate_all,
        predict=lambda *_args, **_kwargs: (24000, [0.2]),
        export_morphology_sheet=lambda dataset, file_format="ods": (
            str(tmp_path / f"{dataset}.{file_format}"),
            "ok",
        ),
        morphology_db_view=lambda dataset, limit, offset: (
            {"headers": ["token_text", "lemma", "upos"], "value": [["hello", "hello", "NOUN"]]},
            f"{dataset}:{limit}:{offset}",
        ),
        load_pronunciation_rules=lambda: ('{"a":{"x":"y"}}', "loaded"),
        apply_pronunciation_rules=lambda raw: (raw, "applied"),
        import_pronunciation_rules=lambda path: (f'{{"path":"{path}"}}', "imported"),
        export_pronunciation_rules=lambda: (str(tmp_path / "rules.json"), "exported"),
        set_tts_only_mode=lambda enabled: f"tts_only={enabled}",
        tts_only_mode_default=False,
        history_service=_HistoryService(),
        choices={},
    )
    app_instance._threaded = (
        lambda work, on_success=None: on_success(work()) if on_success is not None else work()
    )
    app_instance._run_on_ui = lambda callback: callback()
    root = None
    last_exc = None
    for _ in range(2):
        try:
            root = app_instance.build_for_test()
            break
        except tk.TclError as exc:
            last_exc = exc
            os.environ.pop("TCL_LIBRARY", None)
            os.environ.pop("TK_LIBRARY", None)
    if root is None:
        pytest.skip(f"Tk runtime unavailable: {last_exc}")
    yield app_instance, root, logger, tmp_path
    if root.winfo_exists():
        root.withdraw()
        root.update_idletasks()
        root.destroy()


def test_runtime_language_and_mix_interactions(ui_app, monkeypatch):
    app_instance, _root, _logger, _tmp_path = ui_app

    assert app_instance.input_text.cget("selectforeground") == app_instance.ui_surface
    assert app_instance.history_listbox.cget("selectforeground") == app_instance.ui_surface
    assert int(app_instance.input_text.cget("height")) >= 20
    assert str(app_instance.input_text.cget("undo")).lower() in {"1", "true"}
    assert app_instance.input_text.bind("<Control-z>")
    assert app_instance.input_text.bind("<Control-y>")
    assert app_instance.input_text.bind("<Button-3>")

    app_instance.input_text.delete("1.0", tk.END)
    app_instance.input_text.insert("1.0", "hello")
    app_instance.input_text.edit_separator()
    app_instance.input_text.insert(tk.END, " world")
    assert (
        app_instance._on_input_text_undo(SimpleNamespace(widget=app_instance.input_text)) == "break"
    )
    assert app_instance._read_text(app_instance.input_text) == "hello"
    assert (
        app_instance._on_input_text_redo(SimpleNamespace(widget=app_instance.input_text)) == "break"
    )
    assert app_instance._read_text(app_instance.input_text) == "hello world"

    popup_calls = []
    monkeypatch.setattr(
        tk.Menu,
        "tk_popup",
        lambda _self, x, y: popup_calls.append((int(x), int(y))),
    )
    context_result = app_instance._on_input_text_context_menu(
        SimpleNamespace(
            widget=app_instance.input_text,
            x=0,
            y=0,
            x_root=11,
            y_root=22,
        )
    )
    assert context_result == "break"
    assert popup_calls and popup_calls[-1] == (11, 22)
    assert app_instance.input_text_context_menu is not None
    assert app_instance.input_text_voice_menu is not None
    assert app_instance.input_text_number_menu is not None
    _assert_menu_palette(
        app_instance.input_text_context_menu,
        surface=app_instance.ui_surface,
        accent=app_instance.select_color,
    )
    _assert_menu_palette(
        app_instance.input_text_voice_menu,
        surface=app_instance.ui_surface,
        accent=app_instance.select_color,
    )
    _assert_menu_palette(
        app_instance.input_text_number_menu,
        surface=app_instance.ui_surface,
        accent=app_instance.select_color,
    )
    end_index = app_instance.input_text_voice_menu.index("end")
    assert end_index is not None and int(end_index) >= 0
    top_menu = app_instance.input_text_voice_menu
    cascade_indexes = [
        index for index in range(int(end_index) + 1) if str(top_menu.type(index)) == "cascade"
    ]
    assert cascade_indexes
    voice_ids: list[str] = []
    first_voice_label = None
    first_voice_menu = None
    for index in cascade_indexes:
        submenu_name = top_menu.entrycget(index, "menu")
        submenu = top_menu.nametowidget(submenu_name)
        submenu_end = submenu.index("end")
        if submenu_end is None:
            continue
        for entry_index in range(int(submenu_end) + 1):
            label = submenu.entrycget(entry_index, "label")
            if label.startswith("[voice=") and label.endswith("]"):
                voice_ids.append(label[len("[voice=") : -1])
                if first_voice_label is None:
                    first_voice_label = label
                    first_voice_menu = submenu
    assert any(not voice_id.startswith("a") for voice_id in voice_ids)
    assert first_voice_label is not None
    assert first_voice_menu is not None
    assert first_voice_label.startswith("[voice=")
    _assert_menu_palette(
        first_voice_menu,
        surface=app_instance.ui_surface,
        accent=app_instance.select_color,
    )
    app_instance.input_text.delete("1.0", tk.END)
    app_instance.input_text.insert("1.0", "hello")
    app_instance.input_text.mark_set(tk.INSERT, "1.0")
    first_voice_menu.invoke(0)
    assert app_instance._read_text(app_instance.input_text).startswith(first_voice_label)
    assert "Inserted [voice=" in app_instance.generate_status_var.get()
    number_menu = app_instance.input_text_number_menu
    assert number_menu is not None
    number_end = number_menu.index("end")
    assert number_end is not None and int(number_end) >= 0
    number_labels = []
    for index in range(int(number_end) + 1):
        if str(number_menu.type(index)) == "separator":
            continue
        number_labels.append(str(number_menu.entrycget(index, "label")))
    assert number_labels == ["[date]", "[tnumber]"]
    app_instance.input_text.delete("1.0", tk.END)
    app_instance.input_text.insert("1.0", "hello")
    app_instance.input_text.mark_set(tk.INSERT, "1.0")
    for index in range(int(number_end) + 1):
        if str(number_menu.type(index)) == "separator":
            continue
        if str(number_menu.entrycget(index, "label")) == "[date]":
            number_menu.invoke(index)
            break
    assert app_instance._read_text(app_instance.input_text).startswith("[date]")
    assert "Inserted [date]" in app_instance.generate_status_var.get()
    faq_widget = app_instance._create_readonly_faq_text(app_instance.root, height=6)
    app_instance._populate_faq_dialog_text(faq_widget)
    faq_content = faq_widget.get("1.0", tk.END)
    assert "[date]" in faq_content
    assert "[tnumber]" in faq_content

    assert app_instance._language_code_from_display("") == app_instance.default_lang
    assert app_instance._language_code_from_display("z") == "z"
    assert app_instance._language_code_from_display("Unknown (b)") == "b"
    assert app_instance._language_code_from_display("???") == app_instance.default_lang
    assert app_instance._language_display_from_code("unknown").endswith(
        f"({app_instance.default_lang})"
    )

    app_instance.language_display_var.set(app_instance.language_code_to_display["b"])
    app_instance._on_language_change()
    assert app_instance.language_var.get() == "b"
    assert app_instance.voice_var.get() in app_instance.current_voice_ids

    app_instance.mix_enabled_var.set(True)
    app_instance._on_mix_toggle()
    assert app_instance.voice_mix_listbox.cget("state") == "normal"
    assert str(app_instance.voice_combo.cget("state")) == "disabled"
    app_instance.mix_enabled_var.set(False)
    app_instance._on_mix_toggle()
    assert app_instance.voice_mix_listbox.cget("state") == "disabled"

    app_instance.mix_enabled_var.set(True)
    app_instance._on_mix_toggle()
    app_instance._set_mix_listbox_values(
        app_instance.current_voice_ids, selected=app_instance.current_voice_ids[:1]
    )
    selected_mix = app_instance._selected_mix_voices()
    assert selected_mix
    app_instance._on_mix_change()
    app_instance.voice_var.set(selected_mix[0])
    app_instance._on_voice_change()

    app_instance._set_runtime_mode("tts_morph", apply_backend=True)
    assert "Morphology mode" in app_instance.runtime_mode_status_var.get()
    app_instance._on_runtime_mode_change()
    app_instance._set_tab_visible("morph", False)
    app_instance._set_tab_visible("morph", True)

    app_instance.cuda_available = False
    app_instance.hardware_var.set("GPU")
    app_instance._sync_hardware_selector_visibility()
    assert app_instance.hardware_var.get() == "CPU"
    assert "true" in app_instance._set_tts_only_mode_wrapped(True).lower()

    app_instance.set_tts_only_mode = None
    fallback = app_instance._set_tts_only_mode_wrapped(False)
    assert "TTS-only mode is OFF" in fallback

    app_instance.accordion_setters["custom"] = (
        lambda expanded: app_instance.generate_status_var.set(str(expanded))
    )
    app_instance._set_accordion_expanded("custom", True)
    assert app_instance.generate_status_var.get() == "True"
    app_instance._set_accordion_expanded("missing", False)


def test_generation_tokenization_and_history(ui_app, monkeypatch):
    app_instance, _root, _logger, tmp_path = ui_app
    _sync_threaded(monkeypatch, app_instance)

    app_instance.input_text.delete("1.0", tk.END)
    app_instance.input_text.insert("1.0", "hello world")
    app_instance._on_tokenize()
    assert "Tokenization complete" in app_instance.generate_status_var.get()
    assert app_instance._read_text(app_instance.token_output_text) == "tok"

    app_instance.generate_first = lambda **_kwargs: (None, "none")
    app_instance._on_generate()
    assert "No audio generated" in app_instance.generate_status_var.get()

    playable = tmp_path / "outputs" / "play.wav"
    playable.parent.mkdir(parents=True, exist_ok=True)
    playable.write_bytes(b"audio")
    app_instance.history_service = SimpleNamespace(update_history=lambda _history: [str(playable)])
    app_instance.generate_first = lambda **_kwargs: ((24000, [0.1]), "ok")
    original_autoplay = app_instance._autoplay_latest_history
    monkeypatch.setattr(app_instance, "_autoplay_latest_history", lambda: False)
    app_instance._on_generate()
    assert "no playable file" in app_instance.generate_status_var.get().lower()
    monkeypatch.setattr(app_instance, "_autoplay_latest_history", original_autoplay)

    app_instance.generate_in_progress = True
    app_instance._set_error_status("boom")
    assert app_instance.generate_status_var.get() == "Error: boom"
    assert app_instance.generate_in_progress is False

    app_instance.history_state = [str(playable)]
    app_instance._render_history()
    assert not app_instance.history_listbox.bind("<<ListboxSelect>>")
    calls = []
    monkeypatch.setattr(
        app_instance,
        "_load_audio_file_async",
        lambda path, *, autoplay, history_index=None, resume_seconds=None: calls.append(
            (path, autoplay, history_index, resume_seconds)
        ),
    )
    app_instance.history_listbox.selection_clear(0, tk.END)
    assert app_instance._selected_history_item(show_errors=True) is None
    assert "Select a history item first" in app_instance.generate_status_var.get()

    app_instance.history_listbox.selection_set(0)
    selected = app_instance._selected_history_item(show_errors=True)
    assert selected and selected[1] == playable
    app_instance.history_listbox.event_generate("<<ListboxSelect>>")
    assert calls == []
    app_instance._on_history_select_autoplay()
    assert calls and calls[-1][1] is True
    calls_before_multi = len(calls)
    second_playable = tmp_path / "outputs" / "play_2.wav"
    second_playable.write_bytes(b"audio2")
    app_instance.history_state = [str(playable), str(second_playable)]
    app_instance._render_history()
    app_instance.history_listbox.selection_clear(0, tk.END)
    app_instance.history_listbox.selection_set(0, 1)
    app_instance._on_history_select_autoplay()
    assert len(calls) == calls_before_multi

    app_instance._on_history_double_click(SimpleNamespace(y=0))
    assert len(calls) >= 2

    app_instance.history_service = None
    app_instance._on_clear_history()
    assert app_instance.history_state == []

    app_instance.history_state = [str(playable)]
    app_instance.history_service = SimpleNamespace(clear_history=lambda _history: [])
    app_instance._on_clear_history()
    assert app_instance.history_state == []
    assert "cleared" in app_instance.generate_status_var.get().lower()

    app_instance.history_state = ["missing.wav", str(playable)]
    autoplay_calls = []
    monkeypatch.setattr(
        app_instance,
        "_load_audio_file_async",
        lambda path, *, autoplay, history_index=None, resume_seconds=None: autoplay_calls.append(
            (path, autoplay, history_index, resume_seconds)
        ),
    )
    assert app_instance._autoplay_latest_history() is True
    assert autoplay_calls and autoplay_calls[0][2] == 1

    app_instance.history_state = [str(playable), str(second_playable)]
    app_instance._render_history()
    app_instance.history_listbox.selection_clear(0, tk.END)
    app_instance.history_listbox.selection_set(0, 1)
    app_instance.history_service = SimpleNamespace(
        remove_selected_history=lambda history, selected_indices: [
            value
            for index, value in enumerate(list(history or []))
            if index not in set(int(i) for i in selected_indices or [])
        ]
    )
    app_instance._on_history_delete_selected()
    assert app_instance.history_state == []
    assert "deleted 2 selected history item" in app_instance.generate_status_var.get().lower()

    opened = {}
    app_instance.history_state = [str(playable)]
    app_instance._render_history()
    app_instance.history_context_index = 0
    monkeypatch.setattr(
        app_instance, "_open_path_in_file_manager", lambda path: opened.setdefault("path", path)
    )
    app_instance._on_history_open_selected_folder()
    assert opened.get("path") == playable.parent

    third_playable = tmp_path / "outputs" / "play_3.wav"
    third_playable.write_bytes(b"audio3")
    app_instance.history_state = [str(playable), str(second_playable), str(third_playable)]
    app_instance._render_history()
    app_instance.history_listbox.selection_clear(0, tk.END)
    app_instance.history_listbox.selection_set(1, 2)
    app_instance.history_listbox.update_idletasks()
    bbox = app_instance.history_listbox.bbox(1)
    event_y = int(bbox[1] + 1) if bbox else 20
    event = SimpleNamespace(y=event_y, x_root=40, y_root=40)
    assert app_instance._on_history_context_menu(event) == "break"
    assert app_instance.history_context_menu is not None
    menu = app_instance.history_context_menu
    _assert_menu_palette(
        menu,
        surface=app_instance.ui_surface,
        accent=app_instance.select_color,
    )
    labels = []
    for i in range(int(menu.index("end")) + 1):
        if str(menu.type(i)) == "separator":
            continue
        labels.append(str(menu.entrycget(i, "label")))
    assert "Delete selected" in labels
    assert "Open Containing Folder" in labels
    app_instance.history_listbox.selection_clear(0, tk.END)
    app_instance.history_listbox.selection_set(1, 2)
    for i in range(int(menu.index("end")) + 1):
        if str(menu.type(i)) == "separator":
            continue
        if str(menu.entrycget(i, "label")) == "Delete selected":
            menu.invoke(i)
            break
    assert app_instance.history_state == [str(playable)]

    app_instance._write_text(None, "x")
    assert app_instance._read_text(None) == ""


def test_pronunciation_morphology_and_stream_handlers(ui_app, monkeypatch):
    app_instance, _root, _logger, tmp_path = ui_app
    _sync_threaded(monkeypatch, app_instance)

    app_instance.load_pronunciation_rules = None
    app_instance._on_pronunciation_load()
    assert "not configured" in app_instance.pronunciation_status_var.get()

    app_instance.apply_pronunciation_rules = None
    app_instance._on_pronunciation_apply()
    assert "not configured" in app_instance.pronunciation_status_var.get()

    app_instance.import_pronunciation_rules = None
    app_instance._on_pronunciation_import()
    assert "not configured" in app_instance.pronunciation_status_var.get()

    app_instance.export_pronunciation_rules = None
    app_instance._on_pronunciation_export()
    assert "not configured" in app_instance.pronunciation_status_var.get()

    app_instance.load_pronunciation_rules = lambda: ('{"a":{"x":"y"}}', "loaded")
    app_instance._on_pronunciation_load()
    assert "loaded" in app_instance.pronunciation_status_var.get().lower()

    app_instance.apply_pronunciation_rules = lambda raw: (raw, "applied")
    app_instance._write_text(app_instance.pronunciation_json_text, '{"b":{"z":"q"}}')
    app_instance._on_pronunciation_apply()
    assert "applied" in app_instance.pronunciation_status_var.get().lower()

    import_path = tmp_path / "import.json"
    import_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tkapp_mod.filedialog, "askopenfilename", lambda **_kwargs: str(import_path))
    app_instance.import_pronunciation_rules = lambda path: (f'{{"path":"{path}"}}', "imported")
    app_instance._on_pronunciation_import()
    assert "imported" in app_instance.pronunciation_status_var.get().lower()

    app_instance.export_pronunciation_rules = lambda: (str(tmp_path / "export.json"), "exported")
    app_instance._on_pronunciation_export()
    assert "exported" in app_instance.pronunciation_status_var.get().lower()

    app_instance.export_supports_format = True
    app_instance.export_morphology_sheet = lambda dataset, file_format: (
        str(tmp_path / f"{dataset}.{file_format}"),
        "ready",
    )
    app_instance.export_dataset_var.set("lexemes")
    app_instance.export_format_var.set("csv")
    app_instance._on_export_morphology()
    assert app_instance.export_path_var.get().endswith(".csv")
    assert "ready" in app_instance.export_status_var.get().lower()

    app_instance.export_supports_format = False
    app_instance.export_morphology_sheet = lambda dataset: (
        str(tmp_path / f"{dataset}.ods"),
        "ready2",
    )
    app_instance._on_export_morphology()
    assert app_instance.export_path_var.get().endswith(".ods")

    app_instance.morphology_db_view = None
    app_instance._on_morphology_preview_dataset_change()
    assert "Rows: 0" in app_instance.morph_preview_status_var.get()

    app_instance.morphology_db_view = lambda dataset, limit, offset: (
        {
            "headers": ["token_text", "lemma", "upos"],
            "value": [["hello", "hello", "NOUN"], ["run", "run", "VERB"]],
        },
        "ok",
    )
    app_instance.export_dataset_var.set("pos_table")
    app_instance._on_morphology_preview_dataset_change()
    assert "Rows:" in app_instance.morph_preview_status_var.get()

    app_instance.morphology_db_view = None
    app_instance._on_morph_refresh()
    assert "not configured" in app_instance.morph_status_var.get()

    app_instance.morphology_db_view = lambda dataset, limit, offset: (
        {"headers": ["h1"], "value": ["bad-row"]},
        "loaded",
    )
    app_instance._on_morph_refresh()
    assert app_instance.morph_headers == ["h1"]
    assert "loaded" in app_instance.morph_status_var.get().lower()

    headers, rows = app_instance._project_morphology_preview_rows(
        "unknown",
        {"headers": ["a", "b"], "value": [["1", "2"], ["3", "4"]]},
    )
    assert headers == ["a", "b"]
    assert rows
    preview = app_instance._build_pos_table_preview_from_lexemes({"headers": [], "value": []})
    assert preview["headers"] == ["No data"]

    class _DeadThread:
        def is_alive(self):
            return False

    class _AliveThread:
        def is_alive(self):
            return True

    app_instance.stream_thread = _AliveThread()
    app_instance._on_stream_start()
    assert "already running" in app_instance.stream_status_var.get().lower()

    original_sd = tkapp_mod.sd
    original_thread = tkapp_mod.threading.Thread
    try:

        class _InlineThread:
            def __init__(self, target, daemon=True):
                self._target = target
                self._alive = False

            def start(self):
                self._alive = True
                self._target()
                self._alive = False

            def join(self, timeout=None):
                _ = timeout

            def is_alive(self):
                return self._alive

        tkapp_mod.sd = None
        tkapp_mod.threading.Thread = _InlineThread
        app_instance.stream_thread = _DeadThread()
        app_instance._on_stream_start()
        assert "unavailable" in app_instance.stream_status_var.get().lower()
    finally:
        tkapp_mod.sd = original_sd
        tkapp_mod.threading.Thread = original_thread

    app_instance._on_stream_stop()
    assert "stopping stream" in app_instance.stream_status_var.get().lower()


def test_audio_player_interactions_and_state(ui_app, monkeypatch):
    app_instance, _root, _logger, tmp_path = ui_app

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    app_instance.history_state = [str(audio_path)]

    load_calls = []
    monkeypatch.setattr(
        app_instance,
        "_load_audio_file_async",
        lambda path, *, autoplay, history_index=None, resume_seconds=None: load_calls.append(
            (path, autoplay, history_index, resume_seconds)
        ),
    )
    monkeypatch.setattr(tkapp_mod.filedialog, "askopenfilename", lambda **_kwargs: str(audio_path))
    app_instance._on_audio_player_open(autoplay=True)
    assert load_calls and load_calls[0][2] == 0

    opened = {"called": False}
    monkeypatch.setattr(
        app_instance,
        "_on_audio_player_open",
        lambda autoplay=False: opened.__setitem__("called", bool(autoplay)),
    )
    app_instance.audio_player_loaded_path = None
    app_instance._on_audio_player_play()
    assert opened["called"] is True

    app_instance.audio_player_loaded_path = audio_path
    app_instance.audio_player_total_frames = 100
    app_instance.audio_player_current_frame = 0
    original_start_playback = app_instance._audio_player_start_playback
    monkeypatch.setattr(app_instance, "_audio_player_start_playback", lambda _frame: True)
    app_instance._on_audio_player_play()
    assert "Playing sample.wav" in app_instance.audio_player_status_var.get()
    monkeypatch.setattr(app_instance, "_audio_player_start_playback", original_start_playback)

    app_instance.audio_player_loaded_path = None
    assert app_instance._audio_player_start_playback(0) is False

    app_instance.audio_player_loaded_path = audio_path
    app_instance.audio_player_pcm_data = np.array([], dtype=np.float32)
    app_instance.audio_player_sample_rate = 24000
    app_instance.audio_player_total_frames = 0
    assert app_instance._audio_player_start_playback_sounddevice(0) is False

    class _FakeSd:
        def __init__(self):
            self.play_calls = []
            self.stop_calls = 0

        def play(self, audio, samplerate, blocking):
            self.play_calls.append((len(audio), samplerate, blocking))

        def stop(self):
            self.stop_calls += 1

    fake_sd = _FakeSd()
    original_sd = tkapp_mod.sd
    try:
        tkapp_mod.sd = fake_sd
        app_instance.audio_player_pcm_data = np.asarray([0.0, 0.1, -0.1], dtype=np.float32)
        app_instance.audio_player_sample_rate = 24000
        app_instance.audio_player_total_frames = 3
        app_instance.audio_player_volume_var.set(1.2)
        assert app_instance._audio_player_start_playback_sounddevice(0) is True
        assert fake_sd.play_calls

        app_instance.audio_player_is_playing = False
        app_instance._on_audio_player_pause()
        assert "Nothing is currently playing" in app_instance.audio_player_status_var.get()

        app_instance.audio_player_is_playing = True
        app_instance.audio_player_backend = "sounddevice"
        app_instance._on_audio_player_pause()
        assert "Paused at" in app_instance.audio_player_status_var.get()
    finally:
        tkapp_mod.sd = original_sd

    app_instance.audio_player_loaded_path = audio_path
    app_instance._on_audio_player_stop()
    assert "Stopped: sample.wav" in app_instance.audio_player_status_var.get()

    app_instance.audio_player_loaded_path = None
    app_instance._on_audio_player_stop()
    assert app_instance.audio_player_status_var.get() == "Stopped."

    app_instance.audio_player_loaded_path = audio_path
    app_instance.audio_player_backend = "vlc"
    seek_targets = []
    monkeypatch.setattr(
        app_instance, "_audio_player_seek_vlc_ms", lambda ms: seek_targets.append(ms)
    )
    app_instance.audio_player_total_frames = 1000
    app_instance.audio_player_sample_rate = 100
    app_instance.audio_player_media_length_ms = 10_000
    app_instance._audio_player_seek_to_seconds(3.2)
    assert seek_targets and seek_targets[-1] == 3200
    assert "Seek:" in app_instance.audio_player_status_var.get()

    app_instance.audio_player_seek_dragging = True
    app_instance._on_audio_player_seek_change("2.5")
    assert "/" in app_instance.audio_player_time_var.get()
    app_instance._on_audio_player_seek_change("bad")

    app_instance.audio_player_waveform = np.asarray([0.0, 0.2, 0.5, 1.0], dtype=np.float32)
    app_instance.audio_player_waveform_canvas.configure(width=120, height=40)
    app_instance._audio_player_redraw_waveform()
    app_instance._on_audio_player_waveform_seek(SimpleNamespace(x=60))

    app_instance.audio_player_auto_next_var.set(True)
    app_instance.audio_player_queue_index = 0
    app_instance.history_state = [str(audio_path), str(audio_path)]
    next_calls = []
    monkeypatch.setattr(
        app_instance,
        "_load_audio_file_async",
        lambda path, *, autoplay, history_index=None, resume_seconds=None: next_calls.append(
            (path, autoplay, history_index)
        ),
    )
    assert app_instance._audio_player_try_auto_next() is True
    assert next_calls and next_calls[0][2] == 1

    assert app_instance._audio_player_format_timestamp(125.7) == "02:05"
    assert app_instance._audio_player_progress_fraction() >= 0.0
    assert app_instance._find_history_index(audio_path) == 0
    assert app_instance._find_history_index(tmp_path / "nope.wav") is None

    text_widget = tk.Text(app_instance.root)
    assert app_instance._is_text_input_widget(text_widget) is True
    assert (
        app_instance._is_text_input_widget(SimpleNamespace(winfo_class=lambda: "Listbox")) is False
    )
    assert (
        app_instance._is_text_input_widget(SimpleNamespace(winfo_class=lambda: "TCombobox"))
        is False
    )
    assert (
        app_instance._is_text_input_widget(SimpleNamespace(winfo_class=lambda: "Button")) is False
    )
    assert app_instance._is_text_input_widget(None) is False

    app_instance.audio_player_volume_var.set(1.1)
    result = app_instance._on_audio_shortcut_volume(
        SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "Button")), 0.1
    )
    assert result == "break"
    assert app_instance._on_audio_shortcut_volume(SimpleNamespace(widget=text_widget), 0.1) is None
    app_instance.audio_player_loaded_path = audio_path
    assert (
        app_instance._on_audio_shortcut_seek(
            SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "Button")), 1.0
        )
        == "break"
    )
    app_instance.audio_player_is_playing = True
    assert (
        app_instance._on_audio_shortcut_play_pause(
            SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "Button"))
        )
        == "break"
    )
    app_instance.audio_player_is_playing = True
    assert (
        app_instance._on_audio_shortcut_play_pause(
            SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "Listbox"))
        )
        == "break"
    )

    assert app_instance.audio_player_shortcut_tooltips
    assert app_instance.audio_player_play_btn.bind("<Enter>")
    assert app_instance.audio_player_pause_btn.bind("<Enter>")
    assert app_instance.audio_player_stop_btn.bind("<Enter>")

    app_instance.audio_player_state_path = tmp_path / "outputs" / "state.json"
    app_instance.audio_player_loaded_path = audio_path
    app_instance.audio_player_queue_index = 1
    app_instance.audio_player_auto_next_var.set(True)
    app_instance._save_audio_player_state()
    assert app_instance.audio_player_state_path.is_file()
    raw_payload = json.loads(app_instance.audio_player_state_path.read_text(encoding="utf-8"))
    assert raw_payload["audio_player"]["queue_index"] == 1
    payload = app_instance._load_audio_player_state()
    assert payload["queue_index"] == 1
    app_instance.audio_player_state_path.write_text("{bad", encoding="utf-8")
    assert app_instance._load_audio_player_state() == {}
    app_instance.audio_player_restore_path = audio_path
    app_instance.audio_player_restore_position_seconds = 1.2
    restore_calls = []
    monkeypatch.setattr(
        app_instance,
        "_load_audio_file_async",
        lambda path, *, autoplay, history_index=None, resume_seconds=None: restore_calls.append(
            (path, autoplay, history_index, resume_seconds)
        ),
    )
    app_instance._restore_audio_player_from_saved_state()
    assert restore_calls and restore_calls[0][3] == 1.2

    assert app_instance._coerce_float("bad", default=0.7, min_value=0.1, max_value=1.0) == 0.7
    assert app_instance._coerce_bool("on", default=False) is True
    assert app_instance._coerce_bool("off", default=True) is False


def test_audio_player_space_shortcut_overrides_transport_button_focus(ui_app):
    app_instance, _root, _logger, tmp_path = ui_app
    audio_path = tmp_path / "focus_shortcut.wav"
    audio_path.write_bytes(b"audio")
    app_instance.audio_player_loaded_path = audio_path

    assert str(app_instance.audio_player_play_btn.cget("takefocus")).lower() in {"0", "false"}
    assert str(app_instance.audio_player_pause_btn.cget("takefocus")).lower() in {"0", "false"}
    assert str(app_instance.audio_player_stop_btn.cget("takefocus")).lower() in {"0", "false"}
    assert app_instance.audio_player_minimal_check_btn is not None
    assert app_instance.audio_player_auto_next_check_btn is not None
    assert app_instance.audio_player_play_btn.bind("<space>")
    assert app_instance.audio_player_pause_btn.bind("<space>")
    assert app_instance.audio_player_stop_btn.bind("<space>")
    assert app_instance.audio_player_minimal_check_btn.bind("<space>")
    assert app_instance.audio_player_auto_next_check_btn.bind("<space>")
    assert app_instance.audio_player_play_btn.bind("<KeyRelease-space>")
    assert app_instance.audio_player_pause_btn.bind("<KeyRelease-space>")
    assert app_instance.audio_player_stop_btn.bind("<KeyRelease-space>")
    assert app_instance.audio_player_minimal_check_btn.bind("<KeyRelease-space>")
    assert app_instance.audio_player_auto_next_check_btn.bind("<KeyRelease-space>")

    calls = {"pause": 0, "play": 0, "stop": 0}
    app_instance._on_audio_player_pause = lambda: calls.__setitem__("pause", calls["pause"] + 1)
    app_instance._on_audio_player_play = lambda: calls.__setitem__("play", calls["play"] + 1)
    app_instance._on_audio_player_stop = lambda: calls.__setitem__("stop", calls["stop"] + 1)

    app_instance.audio_player_is_playing = True
    app_instance.audio_player_is_paused = False
    space_result = app_instance._on_audio_shortcut_play_pause_from_transport_button(
        SimpleNamespace(widget=app_instance.audio_player_stop_btn)
    )
    key_release_result = app_instance._on_audio_shortcut_consume_key_release(
        SimpleNamespace(widget=app_instance.audio_player_stop_btn)
    )

    assert space_result == "break"
    assert key_release_result == "break"
    assert calls["pause"] == 1
    assert calls["stop"] == 0
    assert calls["play"] == 0

    app_instance.audio_player_last_space_pressed_at = 0.0
    minimal_before = bool(app_instance.audio_player_minimal_var.get())
    minimal_space_result = app_instance._on_audio_shortcut_play_pause_from_transport_button(
        SimpleNamespace(widget=app_instance.audio_player_minimal_check_btn)
    )
    assert bool(app_instance.audio_player_minimal_var.get()) == minimal_before
    assert minimal_space_result == "break"
    assert calls["pause"] == 2
    assert calls["stop"] == 0

    app_instance.audio_player_last_space_pressed_at = 0.0
    app_instance.audio_player_is_playing = False
    app_instance.audio_player_is_paused = True
    play_result = app_instance._on_audio_shortcut_play_pause_from_transport_button(
        SimpleNamespace(widget=app_instance.audio_player_play_btn)
    )

    assert play_result == "break"
    assert calls["play"] == 1
    assert calls["stop"] == 0

    app_instance.audio_player_last_space_pressed_at = 0.0
    minimal_space_result_paused = app_instance._on_audio_shortcut_play_pause_from_transport_button(
        SimpleNamespace(widget=app_instance.audio_player_minimal_check_btn)
    )
    assert bool(app_instance.audio_player_minimal_var.get()) == minimal_before
    assert minimal_space_result_paused == "break"
    assert calls["play"] == 2
    assert calls["stop"] == 0


def test_audio_player_double_space_shortcut_stops_playback(ui_app, monkeypatch):
    app_instance, _root, _logger, tmp_path = ui_app
    audio_path = tmp_path / "double_space_shortcut.wav"
    audio_path.write_bytes(b"audio")
    app_instance.audio_player_loaded_path = audio_path
    app_instance.audio_player_current_frame = 120

    calls = {"pause": 0, "play": 0, "stop": 0}

    def _pause():
        calls["pause"] += 1
        app_instance.audio_player_is_playing = False
        app_instance.audio_player_is_paused = True

    def _play():
        calls["play"] += 1
        app_instance.audio_player_is_playing = True
        app_instance.audio_player_is_paused = False

    def _stop():
        calls["stop"] += 1
        app_instance.audio_player_is_playing = False
        app_instance.audio_player_is_paused = False
        app_instance.audio_player_current_frame = 0

    app_instance._on_audio_player_pause = _pause
    app_instance._on_audio_player_play = _play
    app_instance._on_audio_player_stop = _stop

    monotonic_values = iter((10.0, 10.2, 20.0, 20.7))
    monkeypatch.setattr(audio_feature_mod.time, "monotonic", lambda: next(monotonic_values))

    app_instance.audio_player_is_playing = True
    app_instance.audio_player_is_paused = False
    assert (
        app_instance._on_audio_shortcut_play_pause(
            SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "Button"))
        )
        == "break"
    )
    assert calls["pause"] == 1
    assert calls["stop"] == 0
    assert calls["play"] == 0

    assert (
        app_instance._on_audio_shortcut_play_pause(
            SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "Button"))
        )
        == "break"
    )
    assert calls["pause"] == 1
    assert calls["stop"] == 1
    assert calls["play"] == 0

    app_instance.audio_player_current_frame = 90
    app_instance.audio_player_is_playing = True
    app_instance.audio_player_is_paused = False
    assert (
        app_instance._on_audio_shortcut_play_pause(
            SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "Button"))
        )
        == "break"
    )
    assert calls["pause"] == 2
    assert calls["stop"] == 1
    assert calls["play"] == 0

    assert (
        app_instance._on_audio_shortcut_play_pause(
            SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "Button"))
        )
        == "break"
    )
    assert calls["pause"] == 2
    assert calls["stop"] == 1
    assert calls["play"] == 1


def test_layout_helpers_close_and_thread_timer_branches(ui_app, monkeypatch):
    app_instance, root, _logger, _tmp_path = ui_app

    host = ttk.Frame(root)
    host.pack(fill="both", expand=True)
    section = app_instance._create_accordion_section(
        host,
        title="Extra",
        expanded=False,
        key="extra",
        show_outline=False,
    )
    assert isinstance(section, ttk.Frame)
    root.update_idletasks()
    app_instance._set_accordion_expanded("extra", True)
    root.update_idletasks()
    app_instance._set_accordion_expanded("extra", False)
    root.update_idletasks()

    canvas = tk.Canvas(root, width=20, height=20)
    canvas.pack()
    app_instance._draw_rounded_rect(
        canvas,
        0,
        0,
        1,
        1,
        radius=0,
        fill="#000",
        outline="#111",
        tags="rect",
    )
    app_instance._draw_rounded_rect(
        canvas,
        0,
        0,
        18,
        18,
        radius=6,
        fill="#000",
        outline="#111",
        tags="poly",
    )

    helper_parent = ttk.Frame(root)
    helper_parent.pack(fill="x")
    widget = app_instance._add_labeled_widget(
        helper_parent, "Label", lambda parent: ttk.Entry(parent)
    )
    assert widget.winfo_exists()
    scale_var = tk.DoubleVar(value=0.5)
    app_instance._add_labeled_scale(
        helper_parent,
        label="Scale",
        variable=scale_var,
        from_=0.0,
        to=1.0,
        resolution=0.1,
    )
    scale_var.set(0.7)
    root.update_idletasks()

    started = []
    errors = []

    monkeypatch.setattr(tkapp_mod.threading, "Thread", _InlineThread)
    tkapp_mod.TkinterDesktopApp._threaded(
        app_instance, lambda: "ok", lambda value: started.append(value)
    )
    assert started == ["ok"]
    tkapp_mod.TkinterDesktopApp._threaded(
        app_instance,
        lambda: (_ for _ in ()).throw(RuntimeError("thread-fail")),
    )
    errors.append(app_instance.generate_status_var.get())
    assert any("thread-fail" in value for value in errors)

    app_instance.root = None
    marker = {"value": 0}
    tkapp_mod.TkinterDesktopApp._run_on_ui(app_instance, lambda: marker.__setitem__("value", 1))
    assert marker["value"] == 0
    app_instance.root = root
    tkapp_mod.TkinterDesktopApp._run_on_ui(app_instance, lambda: marker.__setitem__("value", 2))
    root.update()
    assert marker["value"] == 2

    app_instance.generate_btn = None
    app_instance._set_generate_button_processing(True)

    app_instance.root = None
    app_instance._start_generate_timer()
    assert app_instance.generate_timer_job is None

    class _CancelRoot:
        def after_cancel(self, _job):
            raise RuntimeError("cancel-fail")

    app_instance.root = _CancelRoot()
    app_instance.generate_timer_job = "job-1"
    app_instance._stop_generate_timer()
    assert app_instance.generate_timer_job is None

    class _AfterRoot:
        def after(self, _delay, _callback):
            return "timer-job"

    app_instance.root = _AfterRoot()
    app_instance.generate_in_progress = True
    app_instance.generate_started_at = time.perf_counter() - 0.2
    app_instance._update_generate_timer()
    assert app_instance.generate_timer_job == "timer-job"
    assert app_instance.generate_status_var.get().endswith(" s")

    app_instance.generate_status_var = None
    app_instance._update_generate_timer()
    assert app_instance.generate_timer_job is None
    app_instance.generate_status_var = tk.StringVar(master=root, value="")
    app_instance.root = root

    called = {"saved": 0, "stopped": 0, "tick": 0, "released": 0}
    monkeypatch.setattr(
        app_instance,
        "_save_audio_player_state",
        lambda: called.__setitem__("saved", called["saved"] + 1),
    )
    monkeypatch.setattr(
        app_instance,
        "_stop_generate_timer",
        lambda: called.__setitem__("stopped", called["stopped"] + 1),
    )
    monkeypatch.setattr(
        app_instance,
        "_stop_audio",
        lambda preserve_player_position=False: called.__setitem__("stopped", called["stopped"] + 1),
    )
    monkeypatch.setattr(
        app_instance,
        "_audio_player_cancel_tick",
        lambda: called.__setitem__("tick", called["tick"] + 1),
    )
    monkeypatch.setattr(
        app_instance,
        "_audio_player_release_vlc",
        lambda: called.__setitem__("released", called["released"] + 1),
    )
    app_instance.root = SimpleNamespace(destroy=lambda: called.__setitem__("destroyed", 1))
    app_instance._on_close()
    assert called["saved"] == 1
    assert called["tick"] == 1
    assert called["released"] == 1
    assert called.get("destroyed") == 1

    app_instance.root = root
    app_instance._bind_audio_player_shortcuts()
    app_instance._bind_audio_player_shortcuts()
    assert app_instance.audio_player_shortcuts_bound is True

    app_instance.audio_player_seek_dragging = False
    app_instance._on_audio_player_seek_change("-5")
    app_instance._on_audio_player_waveform_seek(SimpleNamespace(x=0))


def test_audio_file_loading_stream_and_tick_branches(ui_app, monkeypatch):
    app_instance, root, _logger, tmp_path = ui_app
    audio_path = tmp_path / "async.wav"
    audio_path.write_bytes(b"a")

    captured_payloads = []
    app_instance._run_on_ui = lambda callback: callback()
    monkeypatch.setattr(tkapp_mod.threading, "Thread", _InlineThread)
    original_on_loaded = app_instance._on_audio_file_loaded
    original_audio_player_update_progress = app_instance._audio_player_update_progress
    monkeypatch.setattr(
        app_instance,
        "_on_audio_file_loaded",
        lambda **kwargs: captured_payloads.append(kwargs),
    )

    class _SfOk:
        @staticmethod
        def read(_path, dtype, always_2d):
            _ = (dtype, always_2d)
            return np.asarray([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32), 24000

    original_sf = tkapp_mod.sf
    tkapp_mod.sf = _SfOk()
    app_instance._load_audio_file_async(
        audio_path, autoplay=True, history_index=3, resume_seconds=1.25
    )
    assert app_instance.audio_player_status_var.get().startswith("Loading async.wav")
    assert captured_payloads and captured_payloads[-1]["sample_rate"] == 24000

    class _SfBadShape:
        @staticmethod
        def read(_path, dtype, always_2d):
            _ = (dtype, always_2d)
            return np.ones((2, 2, 2), dtype=np.float32), 24000

    tkapp_mod.sf = _SfBadShape()
    app_instance._load_audio_file_async(audio_path, autoplay=False)
    assert "Unsupported audio shape." in captured_payloads[-1]["waveform_warning"]
    tkapp_mod.sf = None
    app_instance._load_audio_file_async(audio_path, autoplay=False)
    assert captured_payloads[-1]["sample_rate"] == 0
    tkapp_mod.sf = original_sf
    app_instance._on_audio_file_loaded = original_on_loaded

    app_instance._stop_audio = lambda preserve_player_position=False: None
    app_instance._audio_player_update_progress = lambda: None
    app_instance._update_audio_player_buttons = lambda: None
    app_instance._save_audio_player_state = lambda: None
    app_instance._audio_player_rebuild_waveform = lambda audio: None
    app_instance._audio_player_redraw_waveform = lambda: None
    original_audio_player_set_media = app_instance._audio_player_set_media
    original_audio_player_can_use_sounddevice = app_instance._audio_player_can_use_sounddevice
    original_start_playback_sounddevice = app_instance._audio_player_start_playback_sounddevice
    original_select_history_index = app_instance._select_history_index
    app_instance._select_history_index = lambda _index: None
    app_instance._find_history_index = lambda _path: 1

    original_vlc = tkapp_mod.vlc
    original_sd = tkapp_mod.sd
    try:
        tkapp_mod.vlc = None
        tkapp_mod.sd = None
        app_instance._audio_player_set_media = lambda _path: False
        app_instance._audio_player_can_use_sounddevice = lambda: False
        app_instance._audio_player_total_seconds = lambda refresh=False: 0.0
        app_instance._on_audio_file_loaded(
            path=audio_path,
            waveform_audio=None,
            sample_rate=0,
            total_frames=0,
            autoplay=False,
            history_index=None,
            resume_seconds=None,
            waveform_warning="",
        )
        assert "No playback backend available" in app_instance.audio_player_status_var.get()

        tkapp_mod.vlc = SimpleNamespace()
        app_instance._audio_player_set_media = lambda _path: False
        app_instance._audio_player_can_use_sounddevice = lambda: True
        app_instance._audio_player_total_seconds = lambda refresh=False: 6.0
        app_instance._on_audio_file_loaded(
            path=audio_path,
            waveform_audio=None,
            sample_rate=0,
            total_frames=0,
            autoplay=False,
            history_index=0,
            resume_seconds=2.7,
            waveform_warning="warn",
        )
        assert app_instance.audio_player_backend == "sounddevice"
        assert "resume 00:02" in app_instance.audio_player_status_var.get()
        assert "Waveform unavailable" in app_instance.audio_player_status_var.get()

        played = {"count": 0}
        app_instance._on_audio_player_play = lambda: played.__setitem__(
            "count", played["count"] + 1
        )
        app_instance._audio_player_set_media = lambda _path: True
        app_instance._audio_player_total_seconds = lambda refresh=False: 0.0
        app_instance._on_audio_file_loaded(
            path=audio_path,
            waveform_audio=np.asarray([0.1, 0.2], dtype=np.float32),
            sample_rate=24000,
            total_frames=2,
            autoplay=True,
            history_index=0,
            resume_seconds=0.0,
            waveform_warning="",
        )
        assert played["count"] == 1
    finally:
        tkapp_mod.vlc = original_vlc
        tkapp_mod.sd = original_sd

    class _BadListbox:
        def selection_clear(self, *_args):
            raise RuntimeError("list-fail")

        def selection_set(self, *_args):
            return None

        def activate(self, *_args):
            return None

        def see(self, *_args):
            return None

    app_instance.history_listbox = _BadListbox()
    app_instance._select_history_index = original_select_history_index
    app_instance._select_history_index(0)
    assert any("history selection" in msg.lower() for msg in _logger.exceptions)

    class _VlcFailure:
        def release(self):
            raise RuntimeError("release-fail")

    app_instance.vlc_audio = _VlcFailure()
    app_instance._audio_player_release_vlc()
    assert app_instance.vlc_audio is None

    app_instance.audio_player_status_var.set("")
    original_vlc = tkapp_mod.vlc
    tkapp_mod.vlc = None
    assert app_instance._ensure_vlc_player() is False
    assert "python-vlc is not installed" in app_instance.audio_player_status_var.get()
    tkapp_mod.vlc = original_vlc

    app_instance.vlc_audio = None
    app_instance.audio_player_status_var.set("")
    tkapp_mod.vlc = object()
    monkeypatch.setattr(
        tkapp_mod, "VlcAudioBackend", lambda: (_ for _ in ()).throw(RuntimeError("vlc-init-fail"))
    )
    assert app_instance._ensure_vlc_player() is False
    assert "VLC init failed" in app_instance.audio_player_status_var.get()

    app_instance._ensure_vlc_player = lambda: True
    app_instance._audio_player_set_media = original_audio_player_set_media
    app_instance.audio_player_volume_var.set(1.0)
    app_instance.vlc_audio = SimpleNamespace(
        load=lambda _path: (_ for _ in ()).throw(RuntimeError("media-fail")),
        set_volume=lambda _value: None,
    )
    assert app_instance._audio_player_set_media(audio_path) is False
    assert "VLC media error" in app_instance.audio_player_status_var.get()

    app_instance.audio_player_loaded_path = audio_path
    app_instance._audio_player_start_playback_sounddevice = lambda _frame: True
    tkapp_mod.vlc = None
    assert app_instance._audio_player_start_playback(1) is True
    tkapp_mod.vlc = original_vlc

    tkapp_mod.vlc = object()
    app_instance._ensure_vlc_player = lambda: False
    app_instance._audio_player_can_use_sounddevice = lambda: False
    assert app_instance._audio_player_start_playback(1) is False
    app_instance._audio_player_can_use_sounddevice = lambda: True
    app_instance._audio_player_start_playback_sounddevice = lambda _frame: True
    assert app_instance._audio_player_start_playback(1) is True

    app_instance._ensure_vlc_player = lambda: True
    app_instance._audio_player_can_use_sounddevice = lambda: False
    app_instance.vlc_audio = SimpleNamespace(
        set_volume=lambda _value: None,
        play=lambda: (_ for _ in ()).throw(RuntimeError("play-fail")),
        set_time_ms=lambda _ms: None,
    )
    assert app_instance._audio_player_start_playback(1) is False
    assert "Playback failed" in app_instance.audio_player_status_var.get()
    tkapp_mod.vlc = original_vlc

    class _SdFail:
        def play(self, *_args, **_kwargs):
            raise RuntimeError("sd-play-fail")

        def stop(self):
            raise RuntimeError("sd-stop-fail")

    original_sd = tkapp_mod.sd
    tkapp_mod.sd = _SdFail()
    app_instance._audio_player_can_use_sounddevice = original_audio_player_can_use_sounddevice
    app_instance._audio_player_start_playback_sounddevice = original_start_playback_sounddevice
    app_instance.audio_player_pcm_data = np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    app_instance.audio_player_total_frames = 2
    app_instance.audio_player_sample_rate = 24000
    assert app_instance._audio_player_start_playback_sounddevice(0) is False
    assert "Playback failed" in app_instance.audio_player_status_var.get()
    tkapp_mod.sd = original_sd

    app_instance.audio_player_is_playing = True
    app_instance.audio_player_total_frames = 0
    app_instance.audio_player_sample_rate = 1000
    app_instance.audio_player_loaded_path = None
    app_instance._audio_player_total_seconds = lambda refresh=False: 10.0
    app_instance._audio_player_current_seconds = lambda: 10.0
    app_instance._audio_player_try_auto_next = lambda: False
    app_instance._audio_player_update_progress = lambda: None
    app_instance._update_audio_player_buttons = lambda: None
    app_instance._save_audio_player_state = lambda: None
    app_instance._on_audio_player_tick()
    assert app_instance.audio_player_status_var.get() == "Playback complete."

    app_instance.audio_player_is_playing = True
    app_instance.audio_player_loaded_path = audio_path
    app_instance._audio_player_try_auto_next = lambda: True
    app_instance._on_audio_player_tick()

    app_instance.audio_player_is_playing = True
    app_instance._audio_player_total_seconds = lambda refresh=False: 10.0
    app_instance._audio_player_current_seconds = lambda: 1.0
    scheduled = {"count": 0}
    app_instance._audio_player_schedule_tick = lambda: scheduled.__setitem__(
        "count", scheduled["count"] + 1
    )
    app_instance._on_audio_player_tick()
    assert scheduled["count"] == 1

    app_instance.audio_player_is_playing = False
    app_instance._on_audio_player_tick()

    app_instance._audio_player_update_progress = original_audio_player_update_progress
    app_instance.audio_player_progress_var = None
    app_instance._audio_player_update_progress()
    app_instance.audio_player_progress_var = tk.DoubleVar(master=root, value=0.0)
    app_instance.audio_player_time_var = tk.StringVar(master=root, value="")
    app_instance._audio_player_total_seconds = lambda refresh=False: 0.0
    app_instance._audio_player_current_seconds = lambda: 0.0
    app_instance._audio_player_update_progress()
    assert app_instance.audio_player_time_var.get() == "00:00 / 00:00"

    app_instance.audio_player_total_frames = 0
    app_instance.audio_player_seek_scale = ttk.Scale(
        root, from_=0, to=1, variable=tk.DoubleVar(master=root)
    )
    app_instance._audio_player_total_seconds = lambda refresh=False: 4.0
    app_instance._audio_player_current_seconds = lambda: 1.5
    app_instance.audio_player_seek_dragging = False
    app_instance._audio_player_update_progress()
    assert app_instance.audio_player_time_var.get().startswith("00:01")

    class _SdStream:
        def __init__(self):
            self.play_calls = []
            self.stop_calls = 0

        def play(self, audio, samplerate, blocking):
            self.play_calls.append((len(audio), samplerate, blocking))

        def stop(self):
            self.stop_calls += 1

    sd_stream = _SdStream()
    original_sd = tkapp_mod.sd
    original_thread = tkapp_mod.threading.Thread
    try:
        tkapp_mod.sd = sd_stream
        tkapp_mod.threading.Thread = _InlineThread
        app_instance.generate_all = lambda **_kwargs: iter([(24000, []), (24000, [0.1])])
        app_instance.stream_thread = None
        app_instance._on_stream_start()
        assert "stream complete" in app_instance.stream_status_var.get().lower()
        assert sd_stream.play_calls

        def _iter_stopped(**_kwargs):
            app_instance.stream_stop_event.set()
            yield (24000, [0.1])

        app_instance.generate_all = _iter_stopped
        app_instance._on_stream_start()
        assert "stream stopped" in app_instance.stream_status_var.get().lower()

        app_instance.stream_stop_event.clear()
        app_instance.generate_all = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("stream-fail")
        )
        app_instance._on_stream_start()
        assert "stream failed" in app_instance.stream_status_var.get().lower()

        class _SdStopFail(_SdStream):
            def stop(self):
                raise RuntimeError("stop-fail")

        tkapp_mod.sd = _SdStopFail()
        app_instance._on_stream_stop()
        assert "stopping stream" in app_instance.stream_status_var.get().lower()
    finally:
        tkapp_mod.sd = original_sd
        tkapp_mod.threading.Thread = original_thread


def test_audio_additional_edge_branches(ui_app, monkeypatch):
    app_instance, root, logger, tmp_path = ui_app
    audio_path = tmp_path / "edge.wav"
    audio_path.write_bytes(b"audio")
    app_instance.audio_player_loaded_path = audio_path
    original_start_playback = app_instance._audio_player_start_playback
    original_total_seconds = app_instance._audio_player_total_seconds
    original_current_seconds = app_instance._audio_player_current_seconds
    original_save_state = app_instance._save_audio_player_state

    app_instance.audio_player_total_frames = 10
    app_instance.audio_player_current_frame = 10
    app_instance.audio_player_is_playing = False
    app_instance._audio_player_start_playback = lambda _frame: False
    app_instance._on_audio_player_play()
    assert app_instance.audio_player_current_frame == 0

    app_instance.audio_player_is_playing = True
    app_instance._on_audio_player_play()
    app_instance._audio_player_start_playback = original_start_playback

    original_vlc = tkapp_mod.vlc
    original_sd = tkapp_mod.sd
    try:
        tkapp_mod.vlc = object()

        class _FakeVlcAudio:
            def __init__(self):
                self.played = 0
                self.times = []
                self.volumes = []

            def set_volume(self, value):
                self.volumes.append(value)

            def play(self):
                self.played += 1

            def set_time_ms(self, value):
                self.times.append(value)

            def get_length_ms(self):
                return 5000

            def get_time_ms(self):
                return 1234

        fake_vlc_audio = _FakeVlcAudio()
        app_instance.vlc_audio = fake_vlc_audio
        app_instance._ensure_vlc_player = lambda: True
        app_instance._audio_player_total_seconds = lambda refresh=False: 5.0
        app_instance._audio_player_update_progress = lambda: None
        app_instance._audio_player_schedule_tick = lambda: None
        app_instance._update_audio_player_buttons = lambda: None
        after_calls = []
        app_instance.root = SimpleNamespace(
            after=lambda delay, callback: after_calls.append(delay) or "after-id"
        )
        app_instance.audio_player_sample_rate = 100
        assert app_instance._audio_player_start_playback(10) is True
        assert fake_vlc_audio.played == 1
        assert fake_vlc_audio.times and fake_vlc_audio.times[-1] == 100
        assert after_calls

        tkapp_mod.sd = None
        app_instance.audio_player_pcm_data = None
        assert app_instance._audio_player_start_playback_sounddevice(0) is False
        assert "Playback backend unavailable" in app_instance.audio_player_status_var.get()

        app_instance.vlc_audio = SimpleNamespace(
            set_time_ms=lambda _ms: (_ for _ in ()).throw(RuntimeError("seek-fail"))
        )
        app_instance._audio_player_seek_vlc_ms(100)
        assert any("seek vlc" in message.lower() for message in logger.exceptions)

        app_instance.vlc_audio = SimpleNamespace(
            set_pause=lambda _on: (_ for _ in ()).throw(RuntimeError("pause-fail"))
        )
        tkapp_mod.sd = SimpleNamespace(
            stop=lambda: (_ for _ in ()).throw(RuntimeError("sd-pause-fail"))
        )
        app_instance.audio_player_backend = "sounddevice"
        app_instance.audio_player_is_playing = True
        app_instance.audio_player_sample_rate = 100
        app_instance.audio_player_total_frames = 100
        original_cancel_tick = app_instance._audio_player_cancel_tick
        app_instance._audio_player_cancel_tick = lambda: None
        app_instance._audio_player_update_progress = lambda: None
        app_instance._update_audio_player_buttons = lambda: None
        app_instance._save_audio_player_state = lambda: None
        app_instance._on_audio_player_pause()
        assert "Paused at" in app_instance.audio_player_status_var.get()

        app_instance.audio_player_loaded_path = audio_path
        original_seek_to_seconds = app_instance._audio_player_seek_to_seconds
        seeks = []
        app_instance._audio_player_seek_to_seconds = lambda value: seeks.append(value)
        app_instance._audio_player_current_seconds = lambda: 3.0
        app_instance._on_audio_player_seek_back()
        app_instance._on_audio_player_seek_forward()
        assert seeks
        app_instance._audio_player_seek_to_seconds = original_seek_to_seconds

        app_instance.audio_player_loaded_path = None
        app_instance._audio_player_seek_relative(1.0)

        app_instance.audio_player_loaded_path = audio_path
        app_instance.audio_player_backend = "sounddevice"
        app_instance.audio_player_sample_rate = 100
        app_instance.audio_player_total_frames = 1000
        app_instance.audio_player_is_playing = True
        app_instance._audio_player_total_seconds = lambda refresh=False: 10.0
        app_instance._audio_player_start_playback_sounddevice = lambda frame: False
        app_instance._audio_player_update_progress = lambda: None
        app_instance._audio_player_schedule_tick = lambda: None
        app_instance._update_audio_player_buttons = lambda: None
        app_instance._save_audio_player_state = lambda: None
        app_instance._audio_player_seek_to_seconds(9.5)
        assert app_instance.audio_player_is_playing is False

        app_instance.root = None
        app_instance._audio_player_cancel_tick = original_cancel_tick
        app_instance._audio_player_schedule_tick()
        app_instance.audio_player_tick_job = "t1"
        app_instance._audio_player_cancel_tick()
        assert app_instance.audio_player_tick_job is None

        app_instance.root = SimpleNamespace(
            after_cancel=lambda _job: (_ for _ in ()).throw(RuntimeError("cancel-fail"))
        )
        app_instance.audio_player_tick_job = "t2"
        app_instance._audio_player_cancel_tick()
        assert app_instance.audio_player_tick_job is None

        app_instance.audio_player_is_playing = True
        app_instance.audio_player_sample_rate = 100
        app_instance._audio_player_total_seconds = lambda refresh=False: 0.0
        app_instance._audio_player_current_seconds = lambda: 2.0
        app_instance._audio_player_update_progress = lambda: None
        app_instance._audio_player_schedule_tick = lambda: None
        tkapp_mod.vlc = None
        app_instance._on_audio_player_tick()

        app_instance.audio_player_is_playing = True
        app_instance.audio_player_loaded_path = audio_path
        app_instance._audio_player_total_seconds = lambda refresh=False: 2.0
        app_instance._audio_player_current_seconds = lambda: 2.0
        app_instance.vlc_audio = SimpleNamespace(get_state=lambda: "ended")
        tkapp_mod.vlc = SimpleNamespace(
            State=SimpleNamespace(Ended="ended", Stopped="stopped", Error="error")
        )
        app_instance._audio_player_try_auto_next = lambda: False
        app_instance._audio_player_update_progress = lambda: None
        app_instance._update_audio_player_buttons = lambda: None
        app_instance._save_audio_player_state = lambda: None
        app_instance._on_audio_player_tick()
        assert "Playback complete:" in app_instance.audio_player_status_var.get()

        app_instance.audio_player_volume_var.set(1.1)
        app_instance.vlc_audio = SimpleNamespace(
            set_volume=lambda _value: (_ for _ in ()).throw(RuntimeError("volume-fail"))
        )
        app_instance._save_audio_player_state = lambda: None
        app_instance._on_audio_player_volume_scale()
        assert any("update vlc volume" in message.lower() for message in logger.exceptions)

        app_instance._audio_player_redraw_waveform = lambda: None
        app_instance._audio_player_rebuild_waveform(np.asarray([], dtype=np.float32))
        assert app_instance.audio_player_waveform is None
        app_instance._audio_player_rebuild_waveform(
            np.asarray([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)
        )
        assert app_instance.audio_player_waveform is not None

        app_instance._audio_player_total_seconds = original_total_seconds
        app_instance.vlc_audio = SimpleNamespace(
            get_length_ms=lambda: (_ for _ in ()).throw(RuntimeError("len-fail"))
        )
        app_instance.audio_player_media_length_ms = 0
        app_instance.audio_player_total_frames = 0
        app_instance.audio_player_sample_rate = 0
        assert app_instance._audio_player_total_seconds(refresh=True) == 0.0

        app_instance._audio_player_current_seconds = original_current_seconds
        app_instance.vlc_audio = SimpleNamespace(
            get_time_ms=lambda: (_ for _ in ()).throw(RuntimeError("time-fail"))
        )
        app_instance.audio_player_backend = "sounddevice"
        app_instance.audio_player_is_playing = True
        app_instance.audio_player_sample_rate = 100
        app_instance.audio_player_sd_start_frame = 10
        app_instance.audio_player_sd_started_at = time.monotonic() - 0.1
        app_instance.audio_player_total_frames = 100
        assert app_instance._audio_player_current_seconds() > 0.0

        app_instance.audio_player_state_path = SimpleNamespace(
            parent=SimpleNamespace(mkdir=lambda parents=True, exist_ok=True: None),
            write_text=lambda _text, encoding="utf-8": (_ for _ in ()).throw(OSError("save-fail")),
            is_file=lambda: False,
        )
        app_instance.audio_player_volume_var = tk.DoubleVar(master=root, value=1.0)
        app_instance.audio_player_auto_next_var = tk.BooleanVar(master=root, value=True)
        app_instance._save_audio_player_state = original_save_state
        app_instance._save_audio_player_state()
        assert any("save audio player state" in message.lower() for message in logger.exceptions)

        app_instance.import_pronunciation_rules = lambda _path: ("{}", "ok")
        monkeypatch.setattr(tkapp_mod.filedialog, "askopenfilename", lambda **_kwargs: "")
        app_instance._on_pronunciation_import()

        app_instance.export_morphology_sheet = None
        app_instance._on_export_morphology()
        assert "not configured" in app_instance.export_status_var.get().lower()

        app_instance.morph_preview_status_var = None
        app_instance._on_morphology_preview_dataset_change()
        app_instance.morph_preview_status_var = tk.StringVar(master=root, value="")

        headers, rows = app_instance._project_morphology_preview_rows(
            "lexemes", {"headers": ["token_text"], "value": "bad"}
        )
        assert headers == ["token_text", "lemma", "upos"]
        assert rows == []
        headers, rows = app_instance._project_morphology_preview_rows(
            "unknown", {"headers": ["h"], "value": ["v"]}
        )
        assert headers == ["h"]
        assert rows == [["v"]]

        preview = app_instance._build_pos_table_preview_from_lexemes(
            {"headers": ["x"], "value": ["bad"]}
        )
        assert preview["headers"] == ["No data"]
        preview = app_instance._build_pos_table_preview_from_lexemes(
            {
                "headers": ["lemma", "upos"],
                "value": [["", ""], ["hello", "NOUN"], ["hello", "NOUN"], ["x", "ZZZ"]],
            }
        )
        assert "Noun" in preview["headers"]
        assert "ZZZ" in preview["headers"]

        app_instance.morph_preview_tree = None
        app_instance._set_morphology_preview_table(["A"], [["1"]], rows_count=1, unique_count=1)

        app_instance.morph_tree = None
        app_instance._apply_table_update({"headers": [], "value": "bad"})
    finally:
        tkapp_mod.vlc = original_vlc
        tkapp_mod.sd = original_sd


def test_vlc_backend_wrapper_branches(monkeypatch, tmp_path: Path):
    class _FakeMedia:
        def __init__(self, path):
            self.path = path
            self.released = False

        def release(self):
            self.released = True

    class _FakePlayer:
        def __init__(self):
            self.media = None
            self.play_rc = 0
            self.volume = 0
            self.time_ms = 0
            self.length_ms = 0
            self.playing = False
            self.state = "ok"

        def set_media(self, media):
            self.media = media

        def play(self):
            return self.play_rc

        def pause(self):
            self.playing = not self.playing

        def set_pause(self, value):
            self.playing = not bool(value)

        def stop(self):
            self.playing = False

        def is_playing(self):
            return self.playing

        def audio_set_volume(self, value):
            self.volume = value

        def get_time(self):
            return self.time_ms

        def get_length(self):
            return self.length_ms

        def set_time(self, value):
            self.time_ms = value

        def get_state(self):
            return self.state

        def release(self):
            return None

    class _FakeInstance:
        def __init__(self, args):
            self.args = args
            self.player = _FakePlayer()
            self.last_media = None

        def media_player_new(self):
            return self.player

        def media_new(self, path):
            self.last_media = _FakeMedia(path)
            return self.last_media

        def release(self):
            return None

    class _FakeVlc:
        def __init__(self):
            self.last_instance = None

        def Instance(self, args):
            self.last_instance = _FakeInstance(args)
            return self.last_instance

    fake_vlc = _FakeVlc()
    monkeypatch.setattr(tkapp_mod, "vlc", fake_vlc)
    monkeypatch.setattr(tkapp_mod.sys, "platform", "linux")

    backend = VlcAudioBackend()
    assert fake_vlc.last_instance.args == ["--no-xlib"]

    with pytest.raises(FileNotFoundError):
        backend.load(str(tmp_path / "missing.wav"))

    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"data")
    backend.load(str(wav_path))
    first_media = backend.media
    backend.load(str(wav_path))
    assert first_media.released is True

    backend.player.play_rc = -1
    with pytest.raises(RuntimeError, match="failed"):
        backend.play()
    backend.player.play_rc = 0
    backend.play()
    backend.pause_toggle()
    backend.set_pause(True)
    backend.stop()
    backend.player.time_ms = 2500
    backend.player.length_ms = 9999
    backend.set_volume(250)
    backend.set_time_ms(1234)
    assert backend.get_time_ms() == 1234
    assert backend.get_length_ms() == 9999
    assert backend.get_state() == "ok"
    backend.release()

    monkeypatch.setattr(tkapp_mod, "vlc", None)
    with pytest.raises(RuntimeError, match="not available"):
        VlcAudioBackend()
