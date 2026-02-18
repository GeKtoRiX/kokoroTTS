"""Generate tab feature wiring and actions."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from ...domain.style import DEFAULT_STYLE_PRESET


class GenerateTabFeature:
    """Build and handle the Generate tab and its nested views."""

    def __init__(self, host) -> None:
        self.host = host

    def build_tab(self, parent: ttk.Frame) -> None:
        ui = self.host
        assert ui.generate_status_var is not None
        assert ui.export_dataset_var is not None
        assert ui.export_format_var is not None
        assert ui.export_status_var is not None
        assert ui.export_path_var is not None
        assert ui.audio_player_status_var is not None
        assert ui.audio_player_track_var is not None
        assert ui.audio_player_progress_var is not None
        assert ui.audio_player_time_var is not None
        assert ui.audio_player_volume_var is not None
        assert ui.audio_player_auto_next_var is not None
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        top = ttk.Frame(parent, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)
        ui.generate_btn = ttk.Button(
            top,
            text="Generate",
            style="Primary.TButton",
            command=ui._on_generate,
        )
        ui.generate_btn.grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(top, textvariable=ui.generate_status_var, style="CardMuted.TLabel").grid(
            row=0,
            column=1,
            sticky="w",
        )

        details_notebook = ttk.Notebook(parent, style="App.TNotebook")
        details_notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        player_tab = ttk.Frame(details_notebook, style="Card.TFrame")
        history_tab = ttk.Frame(details_notebook, style="Card.TFrame")
        tokens_tab = ttk.Frame(details_notebook, style="Card.TFrame")
        morphology_tab = ttk.Frame(details_notebook, style="Card.TFrame")
        faq_tab = ttk.Frame(details_notebook, style="Card.TFrame")
        ui.generate_detail_notebook = details_notebook
        ui.generate_detail_tabs = {
            "player": player_tab,
            "history": history_tab,
            "tokens": tokens_tab,
            "morphology": morphology_tab,
            "faq": faq_tab,
        }
        details_notebook.add(player_tab, text="Player")
        details_notebook.add(history_tab, text="History")
        details_notebook.add(tokens_tab, text="Tokens")
        details_notebook.add(morphology_tab, text="Morphology")
        details_notebook.add(faq_tab, text="FAQ")

        self._build_player_tab(player_tab)
        self._build_history_tab(history_tab)
        self._build_tokens_tab(tokens_tab)
        self._build_morphology_preview_tab(morphology_tab)
        ui._build_faq_tab(faq_tab)

    def _build_player_tab(self, parent: ttk.Frame) -> None:
        ui = self.host
        parent.grid_columnconfigure(0, weight=1)
        player_shell = ttk.Frame(parent, style="PlayerShell.TFrame", padding=(8, 8))
        player_shell.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        player_shell.grid_columnconfigure(0, weight=1)
        for fixed_row in (0, 1, 2, 3, 4, 5, 6, 7):
            player_shell.grid_rowconfigure(fixed_row, weight=0)

        mode_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        mode_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(2, 4))
        mode_row.grid_columnconfigure(0, weight=1)
        ttk.Checkbutton(
            mode_row,
            text="Minimal",
            variable=ui.audio_player_minimal_var,
            style="Player.TCheckbutton",
            command=ui._on_audio_player_minimal_toggle,
        ).grid(row=0, column=1, sticky="e", padx=8, pady=6)

        track_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        track_row.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        track_row.grid_columnconfigure(0, weight=1)
        ttk.Label(
            track_row,
            textvariable=ui.audio_player_track_var,
            style="PlayerTitle.TLabel",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ui.audio_player_track_frame = track_row

        waveform_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        waveform_row.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        waveform_row.grid_columnconfigure(0, weight=1)
        ui.audio_player_waveform_canvas = tk.Canvas(
            waveform_row,
            height=100,
            background=getattr(ui, "ui_surface", "#031003"),
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
        )
        ui.audio_player_waveform_canvas.grid(row=0, column=0, sticky="ew")
        ui.audio_player_waveform_canvas.bind("<Configure>", lambda _e: ui._audio_player_redraw_waveform())
        ui.audio_player_waveform_canvas.bind("<Button-1>", ui._on_audio_player_waveform_seek)
        ui.audio_player_waveform_frame = waveform_row

        timeline_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        timeline_row.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        timeline_row.grid_columnconfigure(0, weight=1)
        timeline_row.grid_columnconfigure(1, weight=0)
        ui.audio_player_seek_scale = ttk.Scale(
            timeline_row,
            from_=0.0,
            to=1.0,
            variable=ui.audio_player_progress_var,
            command=ui._on_audio_player_seek_change,
        )
        ui.audio_player_seek_scale.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ui.audio_player_seek_scale.bind("<ButtonPress-1>", ui._on_audio_player_seek_press)
        ui.audio_player_seek_scale.bind("<ButtonRelease-1>", ui._on_audio_player_seek_release)
        ttk.Label(
            timeline_row,
            textvariable=ui.audio_player_time_var,
            style="PlayerTime.TLabel",
            anchor="e",
        ).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ui.audio_player_seek_frame = timeline_row

        controls_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        controls_row.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        controls_row.grid_columnconfigure(0, weight=1)
        ui.audio_player_controls_frame = controls_row

        controls_main = ttk.Frame(controls_row, style="PlayerShell.TFrame")
        controls_main.grid(row=0, column=0, sticky="ew")
        controls_main.grid_columnconfigure(0, weight=1)
        controls_main.grid_columnconfigure(4, weight=1)
        ui.audio_player_play_btn = ttk.Button(
            controls_main,
            text="\u25b6",
            style="TransportPrimary.TButton",
            command=ui._on_audio_player_play,
        )
        ui.audio_player_play_btn.grid(row=0, column=1, padx=8, pady=6)
        ui.audio_player_pause_btn = ttk.Button(
            controls_main,
            text="\u258c\u258c",
            style="Transport.TButton",
            command=ui._on_audio_player_pause,
        )
        ui.audio_player_pause_btn.grid(row=0, column=2, padx=8, pady=6)
        ui.audio_player_stop_btn = ttk.Button(
            controls_main,
            text="\u25a0",
            style="Transport.TButton",
            command=ui._on_audio_player_stop,
        )
        ui.audio_player_stop_btn.grid(row=0, column=3, padx=8, pady=6)

        controls_seek = ttk.Frame(controls_row, style="PlayerShell.TFrame")
        controls_seek.grid(row=1, column=0, sticky="ew")
        controls_seek.grid_columnconfigure(0, weight=1)
        controls_seek.grid_columnconfigure(3, weight=1)
        ttk.Button(
            controls_seek,
            text="-5s",
            style="Small.TButton",
            command=ui._on_audio_player_seek_back,
        ).grid(row=0, column=1, padx=8, pady=6)
        ttk.Button(
            controls_seek,
            text="+5s",
            style="Small.TButton",
            command=ui._on_audio_player_seek_forward,
        ).grid(row=0, column=2, padx=8, pady=6)

        volume_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        volume_row.grid(row=5, column=0, sticky="ew", padx=8, pady=6)
        volume_row.grid_columnconfigure(0, weight=1)
        volume_row.grid_columnconfigure(2, weight=1)
        ui.audio_player_volume_frame = volume_row

        volume_group = ttk.Frame(volume_row, style="PlayerShell.TFrame")
        volume_group.grid(row=0, column=1, padx=8, pady=6)
        ttk.Label(volume_group, text="Volume", style="PlayerMeta.TLabel").grid(
            row=0,
            column=0,
            padx=8,
            pady=6,
        )
        ttk.Scale(
            volume_group,
            from_=0.0,
            to=1.5,
            variable=ui.audio_player_volume_var,
            command=lambda _value: ui._on_audio_player_volume_scale(),
        ).grid(
            row=0,
            column=1,
            padx=8,
            pady=6,
        )
        ui.audio_player_volume_value_label = ttk.Label(
            volume_group,
            text="100%",
            style="PlayerTime.TLabel",
            width=5,
            anchor="e",
        )
        ui.audio_player_volume_value_label.grid(row=0, column=2, padx=8, pady=6)

        auto_next_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        auto_next_row.grid(row=6, column=0, sticky="ew", padx=8, pady=6)
        auto_next_row.grid_columnconfigure(0, weight=1)
        ttk.Checkbutton(
            auto_next_row,
            text="Auto-next",
            variable=ui.audio_player_auto_next_var,
            style="Player.TCheckbutton",
            command=ui._save_audio_player_state,
        ).grid(row=0, column=1, sticky="e", padx=8, pady=6)
        ui.audio_player_autonext_frame = auto_next_row

        status_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        status_row.grid(row=7, column=0, sticky="ew", padx=8, pady=(6, 2))
        status_row.grid_columnconfigure(0, weight=1)
        ttk.Label(
            status_row,
            textvariable=ui.audio_player_status_var,
            style="PlayerStatus.TLabel",
            justify="left",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ui.audio_player_status_frame = status_row
        ui._sync_audio_player_control_labels()
        ui._update_audio_player_buttons()
        ui._apply_audio_player_minimal_mode()

    def _build_history_tab(self, parent: ttk.Frame) -> None:
        ui = self.host
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)
        history_actions = ttk.Frame(parent, style="Card.TFrame")
        history_actions.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(
            history_actions,
            text="Clear history",
            style="Primary.TButton",
            command=ui._on_clear_history,
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ui.history_listbox = tk.Listbox(
            parent,
            height=6,
            exportselection=False,
            font=("Courier New", 10),
            relief="flat",
            borderwidth=1,
        )
        ui._style_listbox(ui.history_listbox)
        ui.history_listbox.grid(row=1, column=0, sticky="nsew", padx=8, pady=(6, 8))
        ui.history_listbox.bind("<<ListboxSelect>>", lambda _e: ui._on_history_select_autoplay())
        ui.history_listbox.bind("<Double-Button-1>", ui._on_history_double_click)

    def _build_tokens_tab(self, parent: ttk.Frame) -> None:
        ui = self.host
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)
        token_actions = ttk.Frame(parent, style="Card.TFrame")
        token_actions.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(token_actions, text="Tokenize", style="Primary.TButton", command=ui._on_tokenize).grid(
            row=0,
            column=0,
            sticky="w",
        )
        token_text_wrap, ui.token_output_text = ui._create_text_with_scrollbar(
            parent,
            wrap=tk.WORD,
            height=7,
            font=("Courier New", 10),
        )
        token_text_wrap.grid(row=1, column=0, sticky="nsew", padx=8, pady=(6, 8))

    def _build_morphology_preview_tab(self, parent: ttk.Frame) -> None:
        ui = self.host
        assert ui.morph_preview_status_var is not None
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_rowconfigure(2, weight=0)

        toolbar = ttk.Frame(parent, style="Card.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        toolbar.grid_columnconfigure(0, weight=0)
        toolbar.grid_columnconfigure(1, weight=0)
        toolbar.grid_columnconfigure(2, weight=0)
        toolbar.grid_columnconfigure(3, weight=0)
        toolbar.grid_columnconfigure(4, weight=1)
        toolbar.grid_columnconfigure(5, weight=0)
        ttk.Label(toolbar, text="Dataset", style="Card.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 6),
            pady=(0, 0),
        )
        dataset_combo = ttk.Combobox(
            toolbar,
            textvariable=ui.export_dataset_var,
            state="readonly",
            values=["lexemes", "occurrences", "expressions", "pos_table"],
        )
        dataset_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=(0, 0))
        dataset_combo.bind("<<ComboboxSelected>>", lambda _e: ui._on_morphology_preview_dataset_change())
        ttk.Label(toolbar, text="Format", style="Card.TLabel").grid(
            row=0,
            column=2,
            sticky="w",
            padx=(0, 6),
            pady=(0, 0),
        )
        ttk.Combobox(
            toolbar,
            textvariable=ui.export_format_var,
            state="readonly",
            values=["ods", "csv", "txt", "xlsx"],
        ).grid(row=0, column=3, sticky="ew", padx=(0, 12), pady=(0, 0))
        export_btn = ttk.Button(
            toolbar,
            text="Export file",
            style="Primary.TButton",
            command=ui._on_export_morphology,
        )
        export_btn.grid(row=0, column=5, sticky="e", pady=(0, 0))
        if not (ui.config.morph_db_enabled and callable(ui.export_morphology_sheet)):
            export_btn.state(["disabled"])

        table_wrap = ttk.Frame(parent, style="Card.TFrame")
        table_wrap.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        table_wrap.grid_columnconfigure(0, weight=1)
        table_wrap.grid_rowconfigure(0, weight=1)
        ui.morph_preview_tree = ttk.Treeview(
            table_wrap,
            show="headings",
            style="Treeview",
            selectmode="none",
        )
        preview_scroll = ui._create_scrollbar(table_wrap, orient=tk.VERTICAL, command=ui.morph_preview_tree.yview)
        ui.morph_preview_tree.configure(yscrollcommand=preview_scroll.set)
        ui.morph_preview_tree.grid(row=0, column=0, sticky="nsew")
        preview_scroll.grid(row=0, column=1, sticky="ns")

        ttk.Label(
            parent,
            textvariable=ui.morph_preview_status_var,
            style="CardMuted.TLabel",
            anchor="w",
        ).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=8,
            pady=(0, 8),
        )

        ui._set_morphology_preview_table(["No data"], [["No data"]], rows_count=0, unique_count=0)
        ui._on_morphology_preview_dataset_change()

    def base_generation_kwargs(self) -> dict[str, Any]:
        ui = self.host
        return {
            "text": ui._read_text(ui.input_text),
            "voice": ui.voice_var.get(),
            "mix_enabled": bool(ui.mix_enabled_var.get()),
            "voice_mix": ui._selected_mix_voices(),
            "speed": float(ui.speed_var.get()),
            "use_gpu": ui.hardware_var.get() == "GPU",
            "pause_seconds": float(ui.pause_var.get()),
            "normalize_times_enabled": bool(ui.normalize_times_var.get()),
            "normalize_numbers_enabled": bool(ui.normalize_numbers_var.get()),
            "style_preset": ui.style_var.get() or DEFAULT_STYLE_PRESET,
        }

    def on_generate(self) -> None:
        ui = self.host
        if ui.generate_in_progress:
            return
        ui._set_generate_button_processing(True)

        def work():
            kwargs = ui._base_generation_kwargs()
            kwargs["output_format"] = ui.output_format_var.get()
            result, tokens = ui.generate_first(**kwargs)
            updated_history = (
                ui.history_service.update_history(ui.history_state)
                if ui.history_service is not None
                else ui.history_state
            )
            return result, tokens, list(updated_history)

        def on_success(payload: tuple[Any, str, list[str]]) -> None:
            try:
                result, tokens, updated_history = payload
                ui.history_state = updated_history
                ui._render_history()
                ui._write_text(ui.token_output_text, tokens)
                if result is None:
                    ui.generate_status_var.set("No audio generated.")
                    return
                if ui.generate_detail_notebook is not None and ui.generate_detail_tabs.get("player") is not None:
                    ui.generate_detail_notebook.select(ui.generate_detail_tabs["player"])
                ui.generate_status_var.set("Generation complete. Loading latest audio...")
                if not ui._autoplay_latest_history():
                    ui.generate_status_var.set("Generation complete, but no playable file found in History.")
            finally:
                ui._set_generate_button_processing(False)

        ui._threaded(work, on_success)

    def on_tokenize(self) -> None:
        ui = self.host
        ui.generate_status_var.set("Tokenizing...")

        def work():
            kwargs = ui._base_generation_kwargs()
            kwargs.pop("use_gpu", None)
            kwargs.pop("pause_seconds", None)
            return ui.tokenize_first(**kwargs)

        def on_success(tokens: str) -> None:
            ui._write_text(ui.token_output_text, tokens)
            ui.generate_status_var.set("Tokenization complete.")

        ui._threaded(work, on_success)
