"""Tkinter desktop UI for Kokoro TTS."""
from __future__ import annotations

import json
import os
import threading
import time
import tkinter as tk
import sys
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable, Mapping

import numpy as np

from ..config import AppConfig
from ..constants import OUTPUT_FORMATS
from ..domain.style import DEFAULT_STYLE_PRESET, STYLE_PRESET_CHOICES
from ..domain.voice import (
    DEFAULT_VOICE,
    LANGUAGE_LABELS,
    default_voice_for_lang,
    get_voice_choices,
    normalize_lang_code,
    voice_language,
)
from .common import (
    APP_TITLE,
    RUNTIME_MODE_CHOICES,
    RUNTIME_MODE_DEFAULT,
    RUNTIME_MODE_FULL,
    RUNTIME_MODE_TTS_MORPH,
    build_morph_update_payload,
    extract_morph_headers,
    llm_only_mode_status_text,
    normalize_runtime_mode,
    resolve_morph_delete_confirmation,
    runtime_mode_from_flags,
    runtime_mode_status_text,
    runtime_mode_tab_visibility,
    supports_export_format_arg,
    tts_only_mode_status_text,
)
from .desktop_types import DesktopApp

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - dependency optional at import time
    sd = None

try:
    import vlc
except Exception:  # pragma: no cover - dependency optional at import time
    vlc = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - dependency optional at import time
    sf = None


class VlcAudioBackend:
    """Thin libVLC wrapper for audio-only playback."""

    def __init__(self) -> None:
        if vlc is None:
            raise RuntimeError("python-vlc is not available")
        args = ["--no-xlib"] if sys.platform.startswith("linux") else []
        self.instance = vlc.Instance(args)
        self.player = self.instance.media_player_new()
        self.media = None

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if self.media is not None:
            try:
                self.media.release()
            except Exception:
                pass
            self.media = None
        media = self.instance.media_new(os.path.abspath(path))
        self.player.set_media(media)
        self.media = media

    def play(self) -> None:
        rc = int(self.player.play())
        if rc == -1:
            raise RuntimeError("VLC failed to start playback.")

    def pause_toggle(self) -> None:
        self.player.pause()

    def set_pause(self, on: bool) -> None:
        self.player.set_pause(1 if on else 0)

    def stop(self) -> None:
        self.player.stop()

    def is_playing(self) -> bool:
        return bool(self.player.is_playing())

    def set_volume(self, vol_0_100: int) -> None:
        self.player.audio_set_volume(max(0, min(100, int(vol_0_100))))

    def get_time_ms(self) -> int:
        return int(self.player.get_time() or 0)

    def get_length_ms(self) -> int:
        return int(self.player.get_length() or 0)

    def set_time_ms(self, ms: int) -> None:
        self.player.set_time(int(ms))

    def get_state(self):
        return self.player.get_state()

    def release(self) -> None:
        try:
            self.player.stop()
        except Exception:
            pass
        if self.media is not None:
            try:
                self.media.release()
            except Exception:
                pass
            self.media = None
        try:
            self.player.release()
        except Exception:
            pass
        try:
            self.instance.release()
        except Exception:
            pass


class TkinterDesktopApp(DesktopApp):
    """Tkinter implementation of the Kokoro desktop UI."""

    def __init__(
        self,
        *,
        config: AppConfig,
        cuda_available: bool,
        logger,
        generate_first,
        tokenize_first,
        generate_all,
        predict,
        export_morphology_sheet=None,
        morphology_db_view=None,
        morphology_db_add=None,
        morphology_db_update=None,
        morphology_db_delete=None,
        load_pronunciation_rules=None,
        apply_pronunciation_rules=None,
        import_pronunciation_rules=None,
        export_pronunciation_rules=None,
        build_lesson_for_tts=None,
        set_tts_only_mode=None,
        set_llm_only_mode=None,
        tts_only_mode_default: bool = False,
        llm_only_mode_default: bool = False,
        history_service=None,
        choices: Mapping[str, str] | None = None,
    ) -> None:
        _ = choices  # Backward compatibility with existing call sites.
        self.title = APP_TITLE
        self.config = config
        self.cuda_available = bool(cuda_available)
        self.logger = logger
        self.generate_first = generate_first
        self.tokenize_first = tokenize_first
        self.generate_all = generate_all
        self.predict = predict
        self.export_morphology_sheet = export_morphology_sheet
        self.morphology_db_view = morphology_db_view
        self.morphology_db_add = morphology_db_add
        self.morphology_db_update = morphology_db_update
        self.morphology_db_delete = morphology_db_delete
        self.load_pronunciation_rules = load_pronunciation_rules
        self.apply_pronunciation_rules = apply_pronunciation_rules
        self.import_pronunciation_rules = import_pronunciation_rules
        self.export_pronunciation_rules = export_pronunciation_rules
        self.build_lesson_for_tts = build_lesson_for_tts
        self.set_tts_only_mode = set_tts_only_mode
        self.set_llm_only_mode = set_llm_only_mode
        self.history_service = history_service
        self.export_supports_format = bool(
            callable(export_morphology_sheet) and supports_export_format_arg(export_morphology_sheet)
        )

        self.default_lang = voice_language(DEFAULT_VOICE)
        self.default_voice = default_voice_for_lang(self.default_lang)
        self.current_voice_choices = get_voice_choices(self.default_lang)
        self.current_voice_ids = [voice_id for _, voice_id in self.current_voice_choices]
        self.language_order = ["a", "b", "e", "f", "h", "i", "j", "p", "z"]
        self.language_display_values = [
            f"{LANGUAGE_LABELS.get(code, code)} ({code})" for code in self.language_order
        ]
        self.language_code_to_display = {
            code: f"{LANGUAGE_LABELS.get(code, code)} ({code})" for code in self.language_order
        }
        self.language_display_to_code = {
            display: code for code, display in self.language_code_to_display.items()
        }

        self.runtime_tts_only_enabled = bool(tts_only_mode_default)
        self.runtime_llm_only_enabled = bool(llm_only_mode_default)
        if not self.runtime_tts_only_enabled and not self.runtime_llm_only_enabled:
            self.runtime_mode_value = RUNTIME_MODE_DEFAULT
            self.runtime_tts_only_enabled = True
        else:
            self.runtime_mode_value = runtime_mode_from_flags(
                tts_only_enabled=self.runtime_tts_only_enabled,
                llm_only_enabled=self.runtime_llm_only_enabled,
            )

        self.root: tk.Tk | None = None
        self.notebook: ttk.Notebook | None = None
        self.tabs: dict[str, tuple[ttk.Frame, str]] = {}
        self.generate_detail_notebook: ttk.Notebook | None = None
        self.generate_detail_tabs: dict[str, ttk.Frame] = {}
        self.history_state: list[str] = []
        self.morph_headers: list[str] = []
        self.morph_delete_armed = ""
        self.stream_stop_event = threading.Event()
        self.stream_thread: threading.Thread | None = None
        self.accordion_setters: dict[str, Callable[[bool], None]] = {}
        self.audio_player_loaded_path: Path | None = None
        self.audio_player_pcm_data: np.ndarray | None = None
        self.audio_player_sample_rate = 0
        self.audio_player_total_frames = 0
        self.audio_player_current_frame = 0
        self.audio_player_media_length_ms = 0
        self.audio_player_backend = "vlc"
        self.audio_player_sd_start_frame = 0
        self.audio_player_sd_started_at = 0.0
        self.audio_player_is_playing = False
        self.audio_player_is_paused = False
        self.audio_player_tick_job: str | None = None
        self.audio_player_seek_dragging = False
        self.audio_player_seek_programmatic = False
        self.audio_player_queue_index: int | None = None
        self.audio_player_waveform: np.ndarray | None = None
        self.audio_player_state_path = Path(self.config.output_dir) / ".audio_player_state.json"
        self.audio_player_restore_path: Path | None = None
        self.audio_player_restore_position_seconds = 0.0
        self.audio_player_shortcuts_bound = False
        self.audio_player_seek_step_seconds = 5.0
        self.vlc_audio: VlcAudioBackend | None = None
        self.generate_in_progress = False
        self.generate_started_at = 0.0
        self.generate_timer_job: str | None = None

        self.language_var: tk.StringVar | None = None
        self.language_display_var: tk.StringVar | None = None
        self.voice_var: tk.StringVar | None = None
        self.mix_enabled_var: tk.BooleanVar | None = None
        self.hardware_var: tk.StringVar | None = None
        self.speed_var: tk.DoubleVar | None = None
        self.style_var: tk.StringVar | None = None
        self.pause_var: tk.DoubleVar | None = None
        default_output_format = (
            config.default_output_format
            if config.default_output_format in OUTPUT_FORMATS
            else "wav"
        )
        self.default_output_format = default_output_format
        self.output_format_var: tk.StringVar | None = None
        self.normalize_times_var: tk.BooleanVar | None = None
        self.normalize_numbers_var: tk.BooleanVar | None = None
        self.advanced_var: tk.BooleanVar | None = None
        self.runtime_mode_var: tk.StringVar | None = None
        self.runtime_mode_status_var: tk.StringVar | None = None

        self.generate_status_var: tk.StringVar | None = None
        self.stream_status_var: tk.StringVar | None = None
        self.pronunciation_status_var: tk.StringVar | None = None
        self.export_status_var: tk.StringVar | None = None
        self.lesson_status_var: tk.StringVar | None = None
        self.morph_status_var: tk.StringVar | None = None
        self.selected_morph_row_var: tk.StringVar | None = None
        self.morph_preview_status_var: tk.StringVar | None = None
        self.audio_player_status_var: tk.StringVar | None = None
        self.audio_player_track_var: tk.StringVar | None = None
        self.audio_player_progress_var: tk.DoubleVar | None = None
        self.audio_player_time_var: tk.StringVar | None = None
        self.audio_player_volume_var: tk.DoubleVar | None = None
        self.audio_player_auto_next_var: tk.BooleanVar | None = None
        self.audio_player_minimal_var: tk.BooleanVar | None = None

        # Widgets assigned during UI build.
        self.input_text: tk.Text | None = None
        self.voice_mix_listbox: tk.Listbox | None = None
        self.voice_combo: ttk.Combobox | None = None
        self.token_output_text: tk.Text | None = None
        self.history_listbox: tk.Listbox | None = None
        self.pronunciation_json_text: tk.Text | None = None
        self.export_path_var: tk.StringVar | None = None
        self.export_dataset_var: tk.StringVar | None = None
        self.export_format_var: tk.StringVar | None = None
        self.lesson_raw_text: tk.Text | None = None
        self.lesson_output_text: tk.Text | None = None
        self.llm_base_url_var: tk.StringVar | None = None
        self.llm_api_key_var: tk.StringVar | None = None
        self.llm_model_var: tk.StringVar | None = None
        self.llm_temperature_var: tk.DoubleVar | None = None
        self.llm_max_tokens_var: tk.IntVar | None = None
        self.llm_timeout_seconds_var: tk.IntVar | None = None
        self.llm_extra_instructions_text: tk.Text | None = None
        self.morph_dataset_var: tk.StringVar | None = None
        self.morph_limit_var: tk.IntVar | None = None
        self.morph_offset_var: tk.IntVar | None = None
        self.morph_tree: ttk.Treeview | None = None
        self.morph_preview_tree: ttk.Treeview | None = None
        self.morph_preview_headers: list[str] = []
        self.morph_add_json_text: tk.Text | None = None
        self.morph_update_json_text: tk.Text | None = None
        self.left_canvas: tk.Canvas | None = None
        self.left_scroll: ttk.Scrollbar | None = None
        self.left_content: ttk.Frame | None = None
        self.audio_player_play_btn: ttk.Button | None = None
        self.audio_player_pause_btn: ttk.Button | None = None
        self.audio_player_stop_btn: ttk.Button | None = None
        self.audio_player_seek_scale: ttk.Scale | None = None
        self.audio_player_waveform_canvas: tk.Canvas | None = None
        self.audio_player_volume_value_label: ttk.Label | None = None
        self.audio_player_track_frame: ttk.Frame | None = None
        self.audio_player_waveform_frame: ttk.Frame | None = None
        self.audio_player_seek_frame: ttk.Frame | None = None
        self.audio_player_controls_frame: ttk.Frame | None = None
        self.audio_player_volume_frame: ttk.Frame | None = None
        self.audio_player_autonext_frame: ttk.Frame | None = None
        self.audio_player_status_frame: ttk.Frame | None = None
        self.generate_btn: ttk.Button | None = None
        self.hardware_section_frame: ttk.Frame | None = None
        self.hardware_combo_widget: ttk.Combobox | None = None

    def launch(self) -> None:
        self._ensure_root()
        assert self.root is not None
        self.root.mainloop()

    def build_for_test(self) -> tk.Tk:
        """Build root/widgets without entering mainloop (for tests)."""
        self._ensure_root()
        assert self.root is not None
        return self.root

    def _ensure_root(self) -> None:
        if self.root is not None:
            return
        root = tk.Tk()
        self._configure_high_dpi(root)
        root.title(APP_TITLE)
        root.geometry("1520x900")
        root.minsize(1120, 700)
        self.root = root
        self._configure_theme()
        self._init_tk_variables()
        self._build_layout()
        self._bind_audio_player_shortcuts()
        self._restore_audio_player_from_saved_state()
        assert self.runtime_mode_var is not None
        self._set_runtime_mode(self.runtime_mode_var.get(), apply_backend=True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.logger.debug("Tkinter UI wiring complete")

    def _configure_high_dpi(self, root: tk.Tk) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        try:
            pixels_per_inch = float(root.winfo_fpixels("1i"))
            scaling = max(1.0, min(2.5, pixels_per_inch / 72.0))
            root.tk.call("tk", "scaling", scaling)
        except Exception:
            pass

    def _configure_theme(self) -> None:
        assert self.root is not None
        style = ttk.Style(self.root)
        # Prefer cross-platform themes so custom dark palette is applied reliably.
        available = set(style.theme_names())
        for theme_name in ("clam", "alt", "default", "vista", "classic", "xpnative", "winnative"):
            if theme_name in available:
                style.theme_use(theme_name)
                break

        bg = "#020702"
        card_bg = "#061106"
        panel_bg = "#0a1a0a"
        text_primary = "#e6e8ef"
        text_muted = "#9aa3b2"
        accent = "#00ff41"
        accent_hover = "#3dff75"
        focus = "#36ff6f"
        disabled = "#6f7888"
        heading = "#f2f5fb"
        input_bg = "#031003"
        border = "#00a83a"
        button_bg = "#0d220d"
        button_hover = "#143514"
        button_pressed = "#0b1b0b"
        button_disabled_bg = "#081308"
        tab_inactive = "#091809"
        tab_active = "#112a11"
        accent_pressed = "#00d63a"
        contrast_text = "#ffffff"
        combo_border = "#1f3f1f"
        combo_focus = "#2a552a"
        self.root.configure(background=bg)
        style.configure(".", background=bg, foreground=text_primary, font=("Segoe UI", 10))
        style.configure("AppBg.TFrame", background=bg)
        style.configure("TFrame", background=card_bg)
        style.configure("TLabel", background=card_bg, foreground=text_primary, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=bg, foreground=heading, font=("Segoe UI", 28, "bold"))
        style.configure(
            "Card.TLabelframe",
            background=card_bg,
            borderwidth=1,
            relief="solid",
            padding=8,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=card_bg,
            foreground=text_primary,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("Card.TFrame", background=card_bg)
        style.configure("Card.TLabel", background=card_bg, foreground=text_primary, font=("Segoe UI", 10))
        style.configure("CardMuted.TLabel", background=card_bg, foreground=text_muted, font=("Segoe UI", 9))
        style.configure(
            "Inset.TFrame",
            background=card_bg,
            borderwidth=1,
            relief="solid",
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.configure("PlayerShell.TFrame", background=card_bg, borderwidth=0, relief="flat")
        style.configure("PlayerWave.TFrame", background=card_bg, borderwidth=0, relief="flat")
        style.configure("PlayerTitle.TLabel", background=card_bg, foreground=heading, font=("Segoe UI", 11, "bold"))
        style.configure("PlayerMeta.TLabel", background=card_bg, foreground=text_muted, font=("Segoe UI", 9))
        style.configure("PlayerTime.TLabel", background=card_bg, foreground=text_primary, font=("Consolas", 10, "bold"))
        style.configure("PlayerStatus.TLabel", background=card_bg, foreground=text_muted, font=("Segoe UI", 9))
        style.configure("AccordionSection.TFrame", background=card_bg, borderwidth=0, relief="flat")
        style.configure("AccordionHeader.TFrame", background=panel_bg)
        style.configure(
            "TButton",
            padding=(10, 6),
            font=("Segoe UI", 10),
            relief="solid",
            borderwidth=1,
            background=card_bg,
            foreground=text_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "TButton",
            background=[("active", card_bg), ("pressed", card_bg), ("disabled", card_bg)],
            bordercolor=[("active", accent_hover), ("pressed", accent), ("disabled", disabled), ("!disabled", border)],
            lightcolor=[("active", accent_hover), ("pressed", accent), ("disabled", disabled), ("!disabled", border)],
            darkcolor=[("active", accent_hover), ("pressed", accent), ("disabled", disabled), ("!disabled", border)],
            foreground=[("disabled", disabled)],
        )
        style.configure(
            "Accordion.TButton",
            padding=(10, 7),
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            relief="solid",
            borderwidth=1,
            background=card_bg,
            foreground=text_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "Accordion.TButton",
            background=[("active", card_bg), ("pressed", card_bg), ("disabled", card_bg)],
            foreground=[("disabled", disabled)],
        )
        style.configure(
            "Primary.TButton",
            padding=(12, 6),
            font=("Segoe UI", 10, "bold"),
            relief="solid",
            borderwidth=1,
            background=card_bg,
            foreground=text_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.configure(
            "Transport.TButton",
            padding=(12, 7),
            font=("Segoe UI", 10, "bold"),
            relief="solid",
            borderwidth=1,
            background=card_bg,
            foreground=text_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "Transport.TButton",
            background=[("active", card_bg), ("pressed", card_bg), ("disabled", card_bg)],
            foreground=[("disabled", disabled), ("!disabled", text_primary)],
        )
        style.configure(
            "TransportPrimary.TButton",
            padding=(12, 7),
            font=("Segoe UI", 10, "bold"),
            relief="solid",
            borderwidth=1,
            background=card_bg,
            foreground=text_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "TransportPrimary.TButton",
            background=[("active", card_bg), ("pressed", card_bg), ("disabled", card_bg)],
            foreground=[("disabled", disabled), ("!disabled", text_primary)],
        )
        style.configure(
            "Small.TButton",
            padding=(8, 4),
            font=("Segoe UI", 9, "bold"),
            relief="solid",
            borderwidth=1,
            background=card_bg,
            foreground=text_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "Small.TButton",
            background=[("active", card_bg), ("pressed", card_bg), ("disabled", card_bg)],
            foreground=[("disabled", disabled), ("!disabled", text_primary)],
        )
        style.map(
            "Primary.TButton",
            background=[("active", card_bg), ("pressed", card_bg), ("disabled", card_bg)],
            foreground=[("disabled", disabled), ("!disabled", text_primary)],
        )
        for button_style in (
            "Accordion.TButton",
            "Primary.TButton",
            "Transport.TButton",
            "TransportPrimary.TButton",
            "Small.TButton",
        ):
            style.map(
                button_style,
                bordercolor=[("active", accent_hover), ("pressed", accent), ("disabled", disabled), ("!disabled", border)],
                lightcolor=[("active", accent_hover), ("pressed", accent), ("disabled", disabled), ("!disabled", border)],
                darkcolor=[("active", accent_hover), ("pressed", accent), ("disabled", disabled), ("!disabled", border)],
            )
        for button_style, radius in (
            ("TButton", 8),
            ("Accordion.TButton", 9),
            ("Primary.TButton", 9),
            ("Transport.TButton", 10),
            ("TransportPrimary.TButton", 10),
            ("Small.TButton", 8),
        ):
            try:
                style.configure(button_style, borderradius=radius)
            except tk.TclError:
                # Older Tk builds may not support native border radius.
                pass
        style.configure(
            "App.TNotebook",
            background=bg,
            borderwidth=1,
            relief="flat",
            tabmargins=(0, 0, 0, 0),
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            focuscolor=border,
        )
        style.configure(
            "TNotebook",
            background=bg,
            borderwidth=1,
            relief="flat",
            tabmargins=(0, 0, 0, 0),
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            focuscolor=border,
        )
        style.configure(
            "App.TNotebook.Tab",
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
            borderwidth=1,
            width=14,
            anchor="center",
            relief="flat",
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            focuscolor=border,
        )
        style.configure(
            "TNotebook.Tab",
            borderwidth=1,
            relief="flat",
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            focuscolor=border,
        )
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", card_bg), ("active", tab_active), ("!selected", tab_inactive)],
            foreground=[("selected", heading), ("!selected", text_muted)],
            expand=[("selected", (0, 0, 0, 0)), ("active", (0, 0, 0, 0)), ("!selected", (0, 0, 0, 0))],
            padding=[("selected", (14, 8)), ("active", (14, 8)), ("!selected", (14, 8))],
        )
        style.configure("TLabelframe", background=card_bg, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=card_bg, foreground=text_primary, font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", fieldbackground=input_bg, foreground=text_primary, bordercolor=border)
        style.map("TEntry", fieldbackground=[("disabled", button_disabled_bg)], foreground=[("disabled", disabled)])
        style.configure(
            "TSpinbox",
            fieldbackground=input_bg,
            background=input_bg,
            foreground=text_primary,
            arrowcolor=text_primary,
        )
        style.configure(
            "TCombobox",
            fieldbackground=input_bg,
            background=input_bg,
            foreground=text_primary,
            bordercolor=combo_border,
            darkcolor=combo_border,
            lightcolor=combo_border,
            focuscolor=combo_focus,
            arrowcolor=text_primary,
            arrowsize=14,
            padding=3,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", input_bg), ("!disabled", input_bg), ("disabled", button_disabled_bg)],
            background=[("readonly", input_bg), ("active", input_bg), ("!disabled", input_bg)],
            foreground=[("readonly", text_primary), ("!disabled", text_primary), ("disabled", disabled)],
            arrowcolor=[("readonly", text_primary), ("active", text_primary), ("disabled", disabled)],
            bordercolor=[("focus", combo_focus), ("active", combo_border), ("!disabled", combo_border)],
            lightcolor=[("focus", combo_focus), ("active", combo_border), ("!disabled", combo_border)],
            darkcolor=[("focus", combo_focus), ("active", combo_border), ("!disabled", combo_border)],
        )
        style.configure(
            "Card.TCheckbutton",
            background=card_bg,
            foreground=text_primary,
            font=("Segoe UI", 10),
            padding=(0, 2),
        )
        style.configure(
            "Player.TCheckbutton",
            background=card_bg,
            foreground=text_primary,
            font=("Segoe UI", 10),
            padding=(0, 2),
        )
        style.configure(
            "Section.TCheckbutton",
            background=panel_bg,
            foreground=heading,
            font=("Segoe UI", 10, "bold"),
            padding=(8, 6),
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "Card.TCheckbutton",
            background=[("selected", card_bg), ("active", card_bg), ("!active", card_bg)],
            foreground=[("disabled", disabled), ("!disabled", text_primary)],
        )
        style.map(
            "Player.TCheckbutton",
            background=[("selected", card_bg), ("active", card_bg), ("!active", card_bg)],
            foreground=[("disabled", disabled), ("!disabled", text_primary)],
        )
        style.map(
            "Section.TCheckbutton",
            background=[("selected", tab_active), ("active", tab_active), ("!selected", panel_bg)],
            foreground=[("disabled", disabled), ("!disabled", heading)],
        )
        try:
            style.layout(
                "Section.TCheckbutton",
                [
                    (
                        "Checkbutton.padding",
                        {
                            "sticky": "nswe",
                            "children": [("Checkbutton.label", {"sticky": "nswe"})],
                        },
                    )
                ],
            )
        except tk.TclError:
            pass
        style.configure(
            "Card.TRadiobutton",
            background=card_bg,
            foreground=text_primary,
            font=("Segoe UI", 10),
            padding=(0, 2),
        )
        style.map(
            "Card.TRadiobutton",
            background=[("selected", card_bg), ("active", card_bg), ("!active", card_bg)],
            foreground=[("disabled", disabled), ("!disabled", text_primary)],
        )
        style.configure(
            "Treeview",
            rowheight=24,
            font=("Segoe UI", 10),
            background=input_bg,
            fieldbackground=input_bg,
            foreground=text_primary,
            borderwidth=1,
            relief="sunken",
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "Treeview",
            background=[("selected", accent)],
            foreground=[("selected", contrast_text)],
        )
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            borderwidth=1,
            background=panel_bg,
            foreground=heading,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=button_bg,
            background=accent,
            borderwidth=1,
            relief="flat",
        )
        style.configure("TScale", background=card_bg, troughcolor=button_bg)
        for scrollbar_style in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
            try:
                style.configure(
                    scrollbar_style,
                    background=panel_bg,
                    troughcolor=input_bg,
                    bordercolor=border,
                    arrowcolor=text_primary,
                    darkcolor=panel_bg,
                    lightcolor=panel_bg,
                    relief="flat",
                    borderwidth=0,
                    arrowsize=14,
                )
                style.map(
                    scrollbar_style,
                    background=[("active", button_hover), ("!active", panel_bg)],
                    arrowcolor=[("active", text_primary), ("disabled", disabled)],
                )
            except tk.TclError:
                pass
        try:
            style.configure(
                "App.TPanedwindow",
                background=border,
                borderwidth=0,
                relief="flat",
                sashrelief="flat",
                sashthickness=6,
            )
        except tk.TclError:
            pass

        self.root.option_add("*TCombobox*Listbox.background", input_bg)
        self.root.option_add("*TCombobox*Listbox.foreground", text_primary)
        self.root.option_add("*TCombobox*Listbox.selectBackground", button_hover)
        self.root.option_add("*TCombobox*Listbox.selectForeground", text_primary)
        self.root.option_add("*TCombobox*Listbox.selectBorderWidth", 0)
        self.root.option_add("*TCombobox*Listbox.activeStyle", "none")
        self.root.option_add("*TCombobox*Listbox.highlightColor", input_bg)
        self.root.option_add("*TCombobox*Listbox.highlightBackground", input_bg)
        self.root.option_add("*TCombobox*Listbox.highlightThickness", 0)
        self.root.option_add("*TCombobox*Listbox.font", "{Segoe UI} 10")
        self.root.option_add("*TCombobox*Listbox.relief", "flat")
        self.root.option_add("*TCombobox*Listbox.borderWidth", 0)

        self.ui_bg = bg
        self.ui_card_bg = card_bg
        self.ui_border = border
        self.ui_radius = 11
        self.ui_surface = input_bg
        self.focus_color = focus
        self.select_color = accent

    def _init_tk_variables(self) -> None:
        assert self.root is not None
        player_state = self._load_audio_player_state()
        saved_volume = self._coerce_float(player_state.get("volume"), default=1.0, min_value=0.0, max_value=1.5)
        saved_auto_next = self._coerce_bool(player_state.get("auto_next"), default=True)
        saved_position = self._coerce_float(
            player_state.get("last_position_seconds"),
            default=0.0,
            min_value=0.0,
            max_value=86400.0,
        )
        saved_path_raw = str(player_state.get("last_path", "") or "").strip()
        self.audio_player_restore_path = Path(saved_path_raw) if saved_path_raw else None
        self.audio_player_restore_position_seconds = float(saved_position)

        self.language_var = tk.StringVar(master=self.root, value=self.default_lang)
        self.language_display_var = tk.StringVar(
            master=self.root,
            value=self.language_code_to_display.get(self.default_lang, f"{self.default_lang}"),
        )
        self.voice_var = tk.StringVar(master=self.root, value=self.default_voice)
        self.mix_enabled_var = tk.BooleanVar(master=self.root, value=False)
        self.hardware_var = tk.StringVar(
            master=self.root,
            value="GPU" if self.cuda_available else "CPU",
        )
        self.speed_var = tk.DoubleVar(master=self.root, value=0.8)
        self.style_var = tk.StringVar(master=self.root, value=DEFAULT_STYLE_PRESET)
        self.pause_var = tk.DoubleVar(master=self.root, value=0.3)
        self.output_format_var = tk.StringVar(master=self.root, value=self.default_output_format)
        self.normalize_times_var = tk.BooleanVar(master=self.root, value=self.config.normalize_times)
        self.normalize_numbers_var = tk.BooleanVar(master=self.root, value=self.config.normalize_numbers)
        self.advanced_var = tk.BooleanVar(master=self.root, value=False)
        self.runtime_mode_var = tk.StringVar(master=self.root, value=self.runtime_mode_value)
        self.runtime_mode_status_var = tk.StringVar(
            master=self.root,
            value=runtime_mode_status_text(self.runtime_mode_value),
        )
        self.generate_status_var = tk.StringVar(master=self.root, value="Ready.")
        self.stream_status_var = tk.StringVar(master=self.root, value="Ready.")
        self.pronunciation_status_var = tk.StringVar(master=self.root, value="")
        self.export_status_var = tk.StringVar(master=self.root, value="")
        self.lesson_status_var = tk.StringVar(master=self.root, value="")
        self.morph_status_var = tk.StringVar(master=self.root, value="")
        self.selected_morph_row_var = tk.StringVar(master=self.root, value="")
        self.morph_preview_status_var = tk.StringVar(
            master=self.root,
            value="Rows: 0 | Unique: 0 | Last updated: -",
        )
        self.audio_player_status_var = tk.StringVar(master=self.root, value="No file loaded.")
        self.audio_player_track_var = tk.StringVar(master=self.root, value="No file loaded.")
        self.audio_player_progress_var = tk.DoubleVar(master=self.root, value=0.0)
        self.audio_player_time_var = tk.StringVar(master=self.root, value="00:00 / 00:00")
        self.audio_player_volume_var = tk.DoubleVar(master=self.root, value=saved_volume)
        self.audio_player_auto_next_var = tk.BooleanVar(master=self.root, value=saved_auto_next)
        self.audio_player_minimal_var = tk.BooleanVar(master=self.root, value=True)
        self.export_path_var = tk.StringVar(master=self.root, value="")
        self.export_dataset_var = tk.StringVar(master=self.root, value="lexemes")
        self.export_format_var = tk.StringVar(master=self.root, value="ods")
        self.llm_base_url_var = tk.StringVar(master=self.root, value=self.config.lm_studio_base_url)
        self.llm_api_key_var = tk.StringVar(master=self.root, value=self.config.lm_studio_api_key)
        self.llm_model_var = tk.StringVar(master=self.root, value=self.config.lm_studio_model)
        self.llm_temperature_var = tk.DoubleVar(master=self.root, value=self.config.lm_studio_temperature)
        self.llm_max_tokens_var = tk.IntVar(master=self.root, value=self.config.lm_studio_max_tokens)
        self.llm_timeout_seconds_var = tk.IntVar(
            master=self.root,
            value=self.config.lm_studio_timeout_seconds,
        )
        self.morph_dataset_var = tk.StringVar(master=self.root, value="occurrences")
        self.morph_limit_var = tk.IntVar(master=self.root, value=100)
        self.morph_offset_var = tk.IntVar(master=self.root, value=0)

        self.audio_player_volume_var.trace_add("write", lambda *_args: self._on_audio_player_volume_var_updated())
        self.audio_player_auto_next_var.trace_add(
            "write",
            lambda *_args: self._save_audio_player_state(),
        )

    def _build_layout(self) -> None:
        assert self.root is not None
        outer = ttk.Frame(self.root, padding=12, style="AppBg.TFrame")
        outer.pack(fill="both", expand=True)
        title = ttk.Label(outer, text=APP_TITLE, style="Title.TLabel")
        title.pack(anchor="w", pady=(0, 12))

        pane = ttk.Panedwindow(outer, orient=tk.HORIZONTAL, style="App.TPanedwindow")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, style="Card.TFrame")
        right = ttk.Frame(pane, style="Card.TFrame")
        pane.add(left, weight=3)
        pane.add(right, weight=3)

        self.left_canvas = tk.Canvas(
            left,
            background=getattr(self, "ui_bg", "#020702"),
            highlightthickness=0,
            borderwidth=0,
        )
        self.left_scroll = self._create_scrollbar(left, orient=tk.VERTICAL, command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=self.left_scroll.set)
        self.left_canvas.pack(side="left", fill="both", expand=True)
        self.left_scroll.pack(side="right", fill="y", padx=(6, 0))

        self.left_content = ttk.Frame(self.left_canvas, style="Card.TFrame")
        window_id = self.left_canvas.create_window((0, 0), window=self.left_content, anchor="nw")

        def _sync_left_scroll(_event: tk.Event[Any]) -> None:
            if self.left_canvas is None or self.left_content is None:
                return
            content_w = max(1, int(self.left_content.winfo_reqwidth()))
            content_h = max(1, int(self.left_content.winfo_reqheight()))
            view_w = max(1, int(self.left_canvas.winfo_width()))
            view_h = max(1, int(self.left_canvas.winfo_height()))
            region_w = max(content_w, view_w)
            region_h = max(content_h, view_h)
            self.left_canvas.configure(scrollregion=(0, 0, region_w, region_h))
            if content_h <= view_h:
                self.left_canvas.yview_moveto(0.0)
                return
            top, bottom = self.left_canvas.yview()
            span = max(0.0, float(bottom) - float(top))
            if span <= 0.0:
                self.left_canvas.yview_moveto(0.0)
                return
            max_top = max(0.0, 1.0 - span)
            if top < 0.0:
                self.left_canvas.yview_moveto(0.0)
            elif top > max_top:
                self.left_canvas.yview_moveto(max_top)

        def _sync_left_width(event: tk.Event[Any]) -> None:
            if self.left_canvas is None:
                return
            self.left_canvas.itemconfigure(window_id, width=event.width)

        self.left_content.bind("<Configure>", _sync_left_scroll)
        self.left_canvas.bind("<Configure>", _sync_left_width)

        self._build_left_panel(self.left_content)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        assert self.language_var is not None
        assert self.language_display_var is not None
        assert self.voice_var is not None
        assert self.mix_enabled_var is not None
        assert self.hardware_var is not None
        assert self.speed_var is not None
        assert self.style_var is not None
        assert self.pause_var is not None
        assert self.output_format_var is not None
        assert self.normalize_times_var is not None
        assert self.normalize_numbers_var is not None
        assert self.advanced_var is not None
        assert self.runtime_mode_var is not None
        assert self.runtime_mode_status_var is not None
        assert self.pronunciation_status_var is not None
        parent.grid_columnconfigure(0, weight=1)

        input_shell = ttk.Frame(parent, style="Card.TFrame")
        input_shell.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        input_shell.grid_columnconfigure(0, weight=1)
        input_text_wrap, self.input_text = self._create_text_with_scrollbar(
            input_shell,
            wrap=tk.WORD,
            height=8,
            font=("Courier New", 10),
        )
        input_text_wrap.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 8))

        quick_shell = ttk.Frame(parent, style="Card.TFrame")
        quick_shell.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        quick_shell.grid_columnconfigure(1, weight=1)

        ttk.Label(quick_shell, text="Language", style="Card.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
            pady=(8, 6),
        )
        language_combo = ttk.Combobox(
            quick_shell,
            textvariable=self.language_display_var,
            state="readonly",
            values=self.language_display_values,
        )
        language_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 6))
        language_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_language_change())

        ttk.Label(quick_shell, text="Voice", style="Card.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            padx=8,
            pady=6,
        )
        self.voice_combo = ttk.Combobox(
            quick_shell,
            textvariable=self.voice_var,
            state="readonly",
            values=self.current_voice_ids,
        )
        self.voice_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        self.voice_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_voice_change())

        ttk.Label(quick_shell, text="Speed", style="Card.TLabel").grid(
            row=2,
            column=0,
            sticky="w",
            padx=8,
            pady=(6, 8),
        )
        speed_row = ttk.Frame(quick_shell, style="Card.TFrame")
        speed_row.grid(row=2, column=1, sticky="ew", padx=8, pady=(6, 8))
        speed_row.grid_columnconfigure(0, weight=1)
        ttk.Scale(
            speed_row,
            from_=0.5,
            to=2.0,
            variable=self.speed_var,
        ).grid(row=0, column=0, sticky="ew")
        speed_value_label = ttk.Label(speed_row, text=f"{float(self.speed_var.get()):.2f}", style="Card.TLabel")
        speed_value_label.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.speed_var.trace_add(
            "write",
            lambda *_args: speed_value_label.configure(
                text=f"{self._coerce_float(self.speed_var.get(), default=0.8, min_value=0.5, max_value=2.0):.2f}"
            ),
        )
        ttk.Label(quick_shell, text="Pause between sentences (s)", style="Card.TLabel").grid(
            row=3,
            column=0,
            sticky="w",
            padx=8,
            pady=(0, 8),
        )
        pause_row = ttk.Frame(quick_shell, style="Card.TFrame")
        pause_row.grid(row=3, column=1, sticky="ew", padx=8, pady=(0, 8))
        pause_row.grid_columnconfigure(0, weight=1)
        ttk.Scale(
            pause_row,
            from_=0.0,
            to=2.0,
            variable=self.pause_var,
        ).grid(row=0, column=0, sticky="ew")
        pause_value_label = ttk.Label(pause_row, text=f"{float(self.pause_var.get()):.2f}", style="Card.TLabel")
        pause_value_label.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.pause_var.trace_add(
            "write",
            lambda *_args: pause_value_label.configure(
                text=f"{self._coerce_float(self.pause_var.get(), default=0.3, min_value=0.0, max_value=2.0):.2f}"
            ),
        )

        advanced_shell = ttk.Frame(parent, style="Card.TFrame")
        advanced_shell.grid(row=2, column=0, sticky="ew")
        advanced_shell.grid_columnconfigure(0, weight=1)
        advanced_shell.grid_rowconfigure(0, weight=0)
        advanced_shell.grid_rowconfigure(1, weight=0)
        advanced_content = ttk.Frame(advanced_shell, style="Card.TFrame")
        advanced_content.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        advanced_content.grid_columnconfigure(0, weight=1)
        advanced_toggle = ttk.Checkbutton(
            advanced_shell,
            variable=self.advanced_var,
            style="Section.TCheckbutton",
            cursor="hand2",
            takefocus=False,
        )
        if "indicatoron" in advanced_toggle.keys():
            advanced_toggle.configure(indicatoron=False)
        advanced_toggle.grid(row=0, column=0, sticky="ew", padx=8, pady=(2, 6))

        def _sync_advanced_toggle_state() -> None:
            expanded = bool(self.advanced_var.get())
            marker = "▾" if expanded else "▸"
            advanced_toggle.configure(text=f"{marker} Advanced settings")
            if expanded:
                advanced_content.grid()
            else:
                advanced_content.grid_remove()

        advanced_toggle.configure(command=_sync_advanced_toggle_state)
        _sync_advanced_toggle_state()

        hardware_section = ttk.Frame(advanced_content, style="Card.TFrame")
        hardware_section.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        hardware_section.grid_columnconfigure(1, weight=1)
        self.hardware_section_frame = hardware_section
        ttk.Label(hardware_section, text="Hardware", style="Card.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
            pady=(6, 2),
        )
        hardware_combo = ttk.Combobox(
            hardware_section,
            textvariable=self.hardware_var,
            state="readonly",
            values=["GPU", "CPU"],
        )
        hardware_combo.grid(row=0, column=1, sticky="w", padx=8, pady=(6, 2))
        self.hardware_combo_widget = hardware_combo
        if not self.cuda_available:
            self.hardware_var.set("CPU")
            hardware_combo.configure(state="disabled")
        self._sync_hardware_selector_visibility()

        generation_section = ttk.Frame(advanced_content, style="Card.TFrame")
        generation_section.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        generation_section.grid_columnconfigure(1, weight=1)
        ttk.Label(generation_section, text="Generation settings", style="Card.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
            pady=(6, 2),
        )
        ttk.Label(generation_section, text="Style preset", style="Card.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            padx=8,
            pady=4,
        )
        ttk.Combobox(
            generation_section,
            textvariable=self.style_var,
            state="readonly",
            values=[value for _label, value in STYLE_PRESET_CHOICES],
        ).grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(generation_section, text="Output format", style="Card.TLabel").grid(
            row=2,
            column=0,
            sticky="w",
            padx=8,
            pady=4,
        )
        ttk.Combobox(
            generation_section,
            textvariable=self.output_format_var,
            state="readonly",
            values=OUTPUT_FORMATS,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=4)
        ttk.Checkbutton(
            generation_section,
            text="Mix voices",
            variable=self.mix_enabled_var,
            style="Card.TCheckbutton",
            command=self._on_mix_toggle,
        ).grid(row=3, column=0, sticky="w", padx=8, pady=(6, 2))
        mix_list_wrap = ttk.Frame(generation_section, style="Inset.TFrame")
        mix_list_wrap.grid(row=3, column=1, sticky="ew", padx=8, pady=(6, 2))
        mix_list_wrap.grid_columnconfigure(0, weight=1)
        mix_list_wrap.grid_rowconfigure(0, weight=1)
        self.voice_mix_listbox = tk.Listbox(
            mix_list_wrap,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=4,
            font=("Tahoma", 9),
            relief="flat",
            borderwidth=1,
        )
        self._style_voice_mix_listbox(self.voice_mix_listbox)
        mix_scroll = self._create_scrollbar(mix_list_wrap, orient=tk.VERTICAL, command=self.voice_mix_listbox.yview)
        self.voice_mix_listbox.configure(yscrollcommand=mix_scroll.set)
        self.voice_mix_listbox.grid(row=0, column=0, sticky="ew", padx=(6, 0), pady=6)
        mix_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 6), pady=6)
        self.voice_mix_listbox.bind("<<ListboxSelect>>", lambda _e: self._on_mix_change())
        self._set_mix_listbox_values(self.current_voice_ids, selected=[])
        self._on_mix_toggle()

        normalization_section = ttk.Frame(advanced_content, style="Card.TFrame")
        normalization_section.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        normalization_section.grid_columnconfigure(0, weight=1)
        ttk.Label(normalization_section, text="Text normalization", style="Card.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
            pady=(6, 2),
        )
        ttk.Checkbutton(
            normalization_section,
            text="Normalize times (12:30 -> twelve thirty)",
            variable=self.normalize_times_var,
            style="Card.TCheckbutton",
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 2))
        ttk.Checkbutton(
            normalization_section,
            text="Normalize numbers (0-9999, decimals, %, ordinals)",
            variable=self.normalize_numbers_var,
            style="Card.TCheckbutton",
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(2, 6))

        runtime_section = ttk.Frame(advanced_content, style="Card.TFrame")
        runtime_section.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        runtime_section.grid_columnconfigure(0, weight=1)
        ttk.Label(runtime_section, text="Runtime mode", style="Card.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
            pady=(6, 2),
        )
        runtime_row = 1
        for label, value in RUNTIME_MODE_CHOICES:
            ttk.Radiobutton(
                runtime_section,
                text=label,
                value=value,
                variable=self.runtime_mode_var,
                style="Card.TRadiobutton",
                command=self._on_runtime_mode_change,
            ).grid(row=runtime_row, column=0, sticky="w", padx=8, pady=2)
            runtime_row += 1
        ttk.Label(
            runtime_section,
            textvariable=self.runtime_mode_status_var,
            wraplength=420,
            foreground="#b7c1d6",
        ).grid(row=runtime_row, column=0, sticky="ew", padx=8, pady=(4, 6))

        pronunciation_section = ttk.Frame(advanced_content, style="Card.TFrame")
        pronunciation_section.grid(row=4, column=0, sticky="ew")
        pronunciation_section.grid_columnconfigure(0, weight=1)
        pronunciation_content = ttk.Frame(pronunciation_section, style="Card.TFrame")
        pronunciation_content.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        pronunciation_content.grid_columnconfigure(0, weight=1)
        pronunciation_state = {"expanded": False}
        pronunciation_toggle_btn = ttk.Button(pronunciation_section, style="Accordion.TButton")
        pronunciation_toggle_btn.grid(row=0, column=0, sticky="ew", padx=8, pady=(2, 6))

        def _set_pronunciation_expanded(expanded: bool) -> None:
            pronunciation_state["expanded"] = bool(expanded)
            marker = "v" if pronunciation_state["expanded"] else ">"
            pronunciation_toggle_btn.configure(text=f"{marker} Pronunciation dictionary")
            if pronunciation_state["expanded"]:
                pronunciation_content.grid()
            else:
                pronunciation_content.grid_remove()

        def _toggle_pronunciation() -> None:
            _set_pronunciation_expanded(not pronunciation_state["expanded"])

        pronunciation_toggle_btn.configure(command=_toggle_pronunciation)
        _set_pronunciation_expanded(False)

        pronunciation_text_wrap, self.pronunciation_json_text = self._create_text_with_scrollbar(
            pronunciation_content,
            wrap=tk.WORD,
            height=8,
            font=("Courier New", 10),
        )
        self.pronunciation_json_text.insert("1.0", "{}")
        pronunciation_text_wrap.grid(row=0, column=0, sticky="ew", padx=8, pady=(2, 6))
        ttk.Label(pronunciation_content, textvariable=self.pronunciation_status_var, wraplength=420).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=8,
            pady=(0, 6),
        )
        pron_btn_row = ttk.Frame(pronunciation_content, style="Card.TFrame")
        pron_btn_row.grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))
        ttk.Button(pron_btn_row, text="Load current", command=self._on_pronunciation_load).grid(
            row=0,
            column=0,
            padx=(0, 8),
        )
        ttk.Button(pron_btn_row, text="Apply", command=self._on_pronunciation_apply).grid(
            row=0,
            column=1,
            padx=(0, 8),
        )
        ttk.Button(pron_btn_row, text="Import file", command=self._on_pronunciation_import).grid(
            row=0,
            column=2,
            padx=(0, 8),
        )
        ttk.Button(pron_btn_row, text="Export JSON", command=self._on_pronunciation_export).grid(
            row=0,
            column=3,
        )

    def _create_accordion_section(
        self,
        parent: ttk.Frame,
        *,
        title: str,
        expanded: bool,
        key: str | None = None,
        content_fill: str = "x",
        content_expand: bool = False,
        pady: tuple[int, int] = (0, 8),
        show_outline: bool = True,
    ) -> ttk.Frame:
        shell_bg = getattr(self, "ui_bg", "#020702")
        card_bg = getattr(self, "ui_card_bg", "#061106")
        border = getattr(self, "ui_border", "#00a83a")
        radius = int(getattr(self, "ui_radius", 10))
        shell = tk.Frame(parent, background=shell_bg, borderwidth=0, highlightthickness=0)
        shell.pack(fill="x", pady=pady)
        canvas = tk.Canvas(
            shell,
            background=shell_bg,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            height=1,
        )
        canvas.pack(fill="x", expand=True)

        section = ttk.Frame(canvas, style="AccordionSection.TFrame")
        section_window = canvas.create_window((1, 1), window=section, anchor="nw")
        header = ttk.Frame(section, style="AccordionHeader.TFrame")
        header.pack(fill="x")
        content = ttk.Frame(section, style="Card.TFrame")
        expanded_state = [bool(expanded)]

        def _redraw(*_args: Any) -> None:
            if not canvas.winfo_exists():
                return
            canvas.update_idletasks()
            width = max(canvas.winfo_width(), 2)
            height = max(section.winfo_reqheight(), 2)
            canvas.configure(height=height + 2)
            canvas.coords(section_window, 1, 1)
            canvas.itemconfigure(section_window, width=max(width - 2, 1))
            canvas.delete("rounded-bg")
            outline_color = border if show_outline else card_bg
            self._draw_rounded_rect(
                canvas,
                1,
                1,
                width - 1,
                height + 1,
                radius=radius,
                fill=card_bg,
                outline=outline_color,
                tags="rounded-bg",
            )
            canvas.tag_lower("rounded-bg")

        def _apply_state() -> None:
            marker = "v" if expanded_state[0] else ">"
            toggle_btn.configure(text=f"{marker} {title}")
            managed = content.winfo_manager() != ""
            if expanded_state[0] and not managed:
                content.pack(fill=content_fill, expand=content_expand, padx=10, pady=(6, 10))
            elif not expanded_state[0] and managed:
                content.pack_forget()
            canvas.after_idle(_redraw)

        def _set_expanded(value: bool) -> None:
            expanded_state[0] = bool(value)
            _apply_state()

        def _toggle() -> None:
            _set_expanded(not expanded_state[0])

        toggle_btn = ttk.Button(header, style="Accordion.TButton", command=_toggle)
        toggle_btn.pack(fill="x", padx=8, pady=8)
        section.bind("<Configure>", _redraw)
        canvas.bind("<Configure>", _redraw)
        if key:
            self.accordion_setters[key] = _set_expanded
        _apply_state()
        return content

    def _set_accordion_expanded(self, key: str, expanded: bool) -> None:
        setter = self.accordion_setters.get(str(key))
        if setter is None:
            return
        setter(bool(expanded))

    @staticmethod
    def _draw_rounded_rect(
        canvas: tk.Canvas,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        radius: int,
        fill: str,
        outline: str,
        tags: str,
    ) -> None:
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        r = max(0, min(int(radius), width // 2, height // 2))
        if r <= 0:
            canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1, tags=tags)
            return
        points = [
            x1 + r,
            y1,
            x2 - r,
            y1,
            x2,
            y1,
            x2,
            y1 + r,
            x2,
            y2 - r,
            x2,
            y2,
            x2 - r,
            y2,
            x1 + r,
            y2,
            x1,
            y2,
            x1,
            y2 - r,
            x1,
            y1 + r,
            x1,
            y1,
        ]
        canvas.create_polygon(
            points,
            smooth=True,
            splinesteps=24,
            fill=fill,
            outline=outline,
            width=1,
            tags=tags,
        )

    def _style_scrolled_text(self, widget: tk.Text) -> None:
        focus = getattr(self, "focus_color", "#36ff6f")
        select = getattr(self, "select_color", "#00ff41")
        surface = getattr(self, "ui_surface", "#031003")
        widget.configure(
            background=surface,
            foreground="#e6e8ef",
            insertbackground="#e6e8ef",
            relief="sunken",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=focus,
            highlightcolor=focus,
            selectbackground=select,
            selectforeground="#ffffff",
        )

    def _create_scrollbar(self, parent: tk.Widget, *, orient: str, command) -> ttk.Scrollbar:
        style_name = "Vertical.TScrollbar" if str(orient).lower() == "vertical" else "Horizontal.TScrollbar"
        return ttk.Scrollbar(parent, orient=orient, command=command, style=style_name)

    def _create_text_with_scrollbar(
        self,
        parent: tk.Widget,
        *,
        wrap: str = tk.WORD,
        height: int = 6,
        font: tuple[str, int] = ("Courier New", 10),
    ) -> tuple[ttk.Frame, tk.Text]:
        container = ttk.Frame(parent, style="Card.TFrame")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        text_widget = tk.Text(
            container,
            wrap=wrap,
            height=height,
            font=font,
            relief="flat",
            borderwidth=1,
        )
        self._style_scrolled_text(text_widget)
        y_scroll = self._create_scrollbar(container, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=y_scroll.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        return container, text_widget

    def _style_listbox(self, widget: tk.Listbox) -> None:
        focus = getattr(self, "focus_color", "#36ff6f")
        select = getattr(self, "select_color", "#00ff41")
        surface = getattr(self, "ui_surface", "#031003")
        widget.configure(
            background=surface,
            foreground="#e6e8ef",
            disabledforeground="#6f7888",
            relief="sunken",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=focus,
            highlightcolor=focus,
            selectbackground=select,
            selectforeground="#ffffff",
            activestyle="none",
        )

    def _style_voice_mix_listbox(self, widget: tk.Listbox) -> None:
        self._style_listbox(widget)
        widget.configure(
            background="#071407",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(parent, style="App.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=(8, 0))

        generate_tab = ttk.Frame(self.notebook)
        stream_tab = ttk.Frame(self.notebook)
        lesson_tab = ttk.Frame(self.notebook)
        morph_tab = ttk.Frame(self.notebook)
        self.tabs = {
            "generate": (generate_tab, "Generate"),
            "stream": (stream_tab, "Stream"),
            "lesson": (lesson_tab, "Lesson Builder (LLM)"),
            "morph": (morph_tab, "Morphology DB"),
        }
        self.notebook.add(generate_tab, text="Generate")
        self.notebook.add(stream_tab, text="Stream")
        self.notebook.add(lesson_tab, text="Lesson Builder (LLM)")
        self.notebook.add(morph_tab, text="Morphology DB")

        self._build_generate_tab(generate_tab)
        self._build_stream_tab(stream_tab)
        self._build_lesson_tab(lesson_tab)
        self._build_morph_tab(morph_tab)

    def _build_generate_tab(self, parent: ttk.Frame) -> None:
        assert self.generate_status_var is not None
        assert self.export_dataset_var is not None
        assert self.export_format_var is not None
        assert self.export_status_var is not None
        assert self.export_path_var is not None
        assert self.audio_player_status_var is not None
        assert self.audio_player_track_var is not None
        assert self.audio_player_progress_var is not None
        assert self.audio_player_time_var is not None
        assert self.audio_player_volume_var is not None
        assert self.audio_player_auto_next_var is not None
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        top = ttk.Frame(parent, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)
        self.generate_btn = ttk.Button(
            top,
            text="Generate",
            style="Primary.TButton",
            command=self._on_generate,
        )
        self.generate_btn.grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(top, textvariable=self.generate_status_var, style="CardMuted.TLabel").grid(
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
        self.generate_detail_notebook = details_notebook
        self.generate_detail_tabs = {
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

        player_tab.grid_columnconfigure(0, weight=1)
        player_shell = ttk.Frame(player_tab, style="PlayerShell.TFrame", padding=(8, 8))
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
            variable=self.audio_player_minimal_var,
            style="Player.TCheckbutton",
            command=self._on_audio_player_minimal_toggle,
        ).grid(row=0, column=1, sticky="e", padx=8, pady=6)

        track_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        track_row.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        track_row.grid_columnconfigure(0, weight=1)
        ttk.Label(
            track_row,
            textvariable=self.audio_player_track_var,
            style="PlayerTitle.TLabel",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.audio_player_track_frame = track_row

        waveform_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        waveform_row.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        waveform_row.grid_columnconfigure(0, weight=1)
        self.audio_player_waveform_canvas = tk.Canvas(
            waveform_row,
            height=100,
            background=getattr(self, "ui_surface", "#031003"),
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
        )
        self.audio_player_waveform_canvas.grid(row=0, column=0, sticky="ew")
        self.audio_player_waveform_canvas.bind("<Configure>", lambda _e: self._audio_player_redraw_waveform())
        self.audio_player_waveform_canvas.bind("<Button-1>", self._on_audio_player_waveform_seek)
        self.audio_player_waveform_frame = waveform_row

        timeline_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        timeline_row.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        timeline_row.grid_columnconfigure(0, weight=1)
        timeline_row.grid_columnconfigure(1, weight=0)
        self.audio_player_seek_scale = ttk.Scale(
            timeline_row,
            from_=0.0,
            to=1.0,
            variable=self.audio_player_progress_var,
            command=self._on_audio_player_seek_change,
        )
        self.audio_player_seek_scale.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.audio_player_seek_scale.bind("<ButtonPress-1>", self._on_audio_player_seek_press)
        self.audio_player_seek_scale.bind("<ButtonRelease-1>", self._on_audio_player_seek_release)
        ttk.Label(
            timeline_row,
            textvariable=self.audio_player_time_var,
            style="PlayerTime.TLabel",
            anchor="e",
        ).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.audio_player_seek_frame = timeline_row

        controls_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        controls_row.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        controls_row.grid_columnconfigure(0, weight=1)
        self.audio_player_controls_frame = controls_row

        controls_main = ttk.Frame(controls_row, style="PlayerShell.TFrame")
        controls_main.grid(row=0, column=0, sticky="ew")
        controls_main.grid_columnconfigure(0, weight=1)
        controls_main.grid_columnconfigure(4, weight=1)
        self.audio_player_play_btn = ttk.Button(
            controls_main,
            text="Play",
            style="TransportPrimary.TButton",
            command=self._on_audio_player_play,
        )
        self.audio_player_play_btn.grid(row=0, column=1, padx=8, pady=6)
        self.audio_player_pause_btn = ttk.Button(
            controls_main,
            text="Pause",
            style="Transport.TButton",
            command=self._on_audio_player_pause,
        )
        self.audio_player_pause_btn.grid(row=0, column=2, padx=8, pady=6)
        self.audio_player_stop_btn = ttk.Button(
            controls_main,
            text="Stop",
            style="Transport.TButton",
            command=self._on_audio_player_stop,
        )
        self.audio_player_stop_btn.grid(row=0, column=3, padx=8, pady=6)

        controls_seek = ttk.Frame(controls_row, style="PlayerShell.TFrame")
        controls_seek.grid(row=1, column=0, sticky="ew")
        controls_seek.grid_columnconfigure(0, weight=1)
        controls_seek.grid_columnconfigure(3, weight=1)
        ttk.Button(
            controls_seek,
            text="-5s",
            style="Small.TButton",
            command=self._on_audio_player_seek_back,
        ).grid(row=0, column=1, padx=8, pady=6)
        ttk.Button(
            controls_seek,
            text="+5s",
            style="Small.TButton",
            command=self._on_audio_player_seek_forward,
        ).grid(row=0, column=2, padx=8, pady=6)

        volume_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        volume_row.grid(row=5, column=0, sticky="ew", padx=8, pady=6)
        volume_row.grid_columnconfigure(0, weight=1)
        volume_row.grid_columnconfigure(2, weight=1)
        self.audio_player_volume_frame = volume_row

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
            variable=self.audio_player_volume_var,
            command=lambda _value: self._on_audio_player_volume_scale(),
        ).grid(
            row=0,
            column=1,
            padx=8,
            pady=6,
        )
        self.audio_player_volume_value_label = ttk.Label(
            volume_group,
            text="100%",
            style="PlayerTime.TLabel",
            width=5,
            anchor="e",
        )
        self.audio_player_volume_value_label.grid(row=0, column=2, padx=8, pady=6)

        auto_next_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        auto_next_row.grid(row=6, column=0, sticky="ew", padx=8, pady=6)
        auto_next_row.grid_columnconfigure(0, weight=1)
        ttk.Checkbutton(
            auto_next_row,
            text="Auto-next",
            variable=self.audio_player_auto_next_var,
            style="Player.TCheckbutton",
            command=self._save_audio_player_state,
        ).grid(row=0, column=1, sticky="e", padx=8, pady=6)
        self.audio_player_autonext_frame = auto_next_row

        status_row = ttk.Frame(player_shell, style="PlayerShell.TFrame")
        status_row.grid(row=7, column=0, sticky="ew", padx=8, pady=(6, 2))
        status_row.grid_columnconfigure(0, weight=1)
        ttk.Label(
            status_row,
            textvariable=self.audio_player_status_var,
            style="PlayerStatus.TLabel",
            justify="left",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.audio_player_status_frame = status_row
        self._sync_audio_player_control_labels()
        self._update_audio_player_buttons()
        self._apply_audio_player_minimal_mode()

        history_tab.grid_columnconfigure(0, weight=1)
        history_tab.grid_rowconfigure(1, weight=1)
        history_actions = ttk.Frame(history_tab, style="Card.TFrame")
        history_actions.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(history_actions, text="Clear history", style="Primary.TButton", command=self._on_clear_history).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.history_listbox = tk.Listbox(
            history_tab,
            height=6,
            exportselection=False,
            font=("Courier New", 10),
            relief="flat",
            borderwidth=1,
        )
        self._style_listbox(self.history_listbox)
        self.history_listbox.grid(row=1, column=0, sticky="nsew", padx=8, pady=(6, 8))
        self.history_listbox.bind("<<ListboxSelect>>", lambda _e: self._on_history_select_autoplay())
        self.history_listbox.bind("<Double-Button-1>", self._on_history_double_click)

        tokens_tab.grid_columnconfigure(0, weight=1)
        tokens_tab.grid_rowconfigure(1, weight=1)
        token_actions = ttk.Frame(tokens_tab, style="Card.TFrame")
        token_actions.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(token_actions, text="Tokenize", style="Primary.TButton", command=self._on_tokenize).grid(
            row=0,
            column=0,
            sticky="w",
        )
        token_text_wrap, self.token_output_text = self._create_text_with_scrollbar(
            tokens_tab,
            wrap=tk.WORD,
            height=7,
            font=("Courier New", 10),
        )
        token_text_wrap.grid(row=1, column=0, sticky="nsew", padx=8, pady=(6, 8))

        assert self.morph_preview_status_var is not None
        morphology_tab.grid_columnconfigure(0, weight=1)
        morphology_tab.grid_rowconfigure(0, weight=0)
        morphology_tab.grid_rowconfigure(1, weight=1)
        morphology_tab.grid_rowconfigure(2, weight=0)

        toolbar = ttk.Frame(morphology_tab, style="Card.TFrame")
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
            textvariable=self.export_dataset_var,
            state="readonly",
            values=["lexemes", "occurrences", "expressions", "reviews", "pos_table"],
        )
        dataset_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=(0, 0))
        dataset_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_morphology_preview_dataset_change())
        ttk.Label(toolbar, text="Format", style="Card.TLabel").grid(
            row=0,
            column=2,
            sticky="w",
            padx=(0, 6),
            pady=(0, 0),
        )
        ttk.Combobox(
            toolbar,
            textvariable=self.export_format_var,
            state="readonly",
            values=["ods", "csv", "txt", "xlsx"],
        ).grid(row=0, column=3, sticky="ew", padx=(0, 12), pady=(0, 0))
        export_btn = ttk.Button(toolbar, text="Export file", style="Primary.TButton", command=self._on_export_morphology)
        export_btn.grid(row=0, column=5, sticky="e", pady=(0, 0))
        if not (self.config.morph_db_enabled and callable(self.export_morphology_sheet)):
            export_btn.state(["disabled"])

        table_wrap = ttk.Frame(morphology_tab, style="Card.TFrame")
        table_wrap.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        table_wrap.grid_columnconfigure(0, weight=1)
        table_wrap.grid_rowconfigure(0, weight=1)
        self.morph_preview_tree = ttk.Treeview(
            table_wrap,
            show="headings",
            style="Treeview",
            selectmode="none",
        )
        preview_scroll = self._create_scrollbar(table_wrap, orient=tk.VERTICAL, command=self.morph_preview_tree.yview)
        self.morph_preview_tree.configure(yscrollcommand=preview_scroll.set)
        self.morph_preview_tree.grid(row=0, column=0, sticky="nsew")
        preview_scroll.grid(row=0, column=1, sticky="ns")

        ttk.Label(
            morphology_tab,
            textvariable=self.morph_preview_status_var,
            style="CardMuted.TLabel",
            anchor="w",
        ).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=8,
            pady=(0, 8),
        )

        self._set_morphology_preview_table(["No data"], [["No data"]], rows_count=0, unique_count=0)
        self._on_morphology_preview_dataset_change()

        self._build_faq_tab(faq_tab)

    def _build_faq_tab(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        container = ttk.Frame(parent, style="Card.TFrame")
        container.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=0)
        container.grid_rowconfigure(1, weight=0)
        container.grid_rowconfigure(2, weight=1)

        pronunciation_text = self._create_readonly_faq_text(container, height=10)
        pronunciation_text.grid(row=0, column=0, sticky="ew")
        self._populate_faq_pronunciation_text(pronunciation_text)

        ttk.Separator(container, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky="ew", pady=6)

        dialog_text = self._create_readonly_faq_text(container, height=10)
        dialog_text.grid(row=2, column=0, sticky="nsew")
        self._populate_faq_dialog_text(dialog_text)

    def _create_readonly_faq_text(self, parent: ttk.Frame, *, height: int) -> tk.Text:
        text_widget = tk.Text(
            parent,
            wrap=tk.WORD,
            height=height,
            padx=20,
            pady=10,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            cursor="arrow",
            takefocus=False,
        )
        surface = getattr(self, "ui_surface", "#031003")
        text_widget.configure(
            background=surface,
            foreground="#e6e8ef",
            insertwidth=0,
        )
        text_widget.tag_configure(
            "heading",
            font=("Segoe UI", 12, "bold"),
            foreground="#f2f5fb",
            justify="left",
            spacing1=2,
            spacing3=8,
        )
        text_widget.tag_configure(
            "code",
            font=("Consolas", 10),
            foreground="#d6e4ff",
            justify="left",
            lmargin1=12,
            lmargin2=12,
            rmargin=8,
            spacing1=2,
            spacing3=4,
        )
        text_widget.tag_configure(
            "section_space",
            justify="left",
            spacing1=4,
            spacing3=10,
        )
        text_widget.tag_configure(
            "body",
            font=("Segoe UI", 10),
            foreground="#e6e8ef",
            justify="left",
        )
        return text_widget

    def _populate_faq_pronunciation_text(self, widget: tk.Text) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, "Customize pronunciation\n", ("heading",))
        widget.insert(
            tk.END,
            "Use Markdown link syntax to define exact pronunciation.\n",
            ("body", "section_space"),
        )
        widget.insert(tk.END, "[Kokoro](/k o k o r o/)\n", ("code", "section_space"))
        widget.insert(
            tk.END,
            "Adjust intonation with punctuation and stress markers.\n",
            ("body",),
        )
        widget.insert(
            tk.END,
            'Punctuation: ; : , . ! ? " ( )\n',
            ("code",),
        )
        widget.insert(
            tk.END,
            "Lower stress:\n",
            ("body",),
        )
        widget.insert(
            tk.END,
            "[1 level](-1)  |  [2 levels](-2)\n",
            ("code",),
        )
        widget.insert(
            tk.END,
            "Raise stress:\n",
            ("body",),
        )
        widget.insert(
            tk.END,
            "[or](+2)  |  (+1 where supported)\n",
            ("code",),
        )
        widget.configure(state=tk.DISABLED)

    def _populate_faq_dialog_text(self, widget: tk.Text) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, "Dialog tags\n", ("heading",))
        widget.insert(
            tk.END,
            "Switch speakers directly inside the text:\n",
            ("body", "section_space"),
        )
        widget.insert(tk.END, "[voice=af_heart]\n", ("code", "section_space"))
        widget.insert(
            tk.END,
            "Switch style per segment:\n",
            ("body",),
        )
        widget.insert(
            tk.END,
            "[style=neutral|narrator|energetic]\n",
            ("code",),
        )
        widget.insert(
            tk.END,
            "Control pauses per segment:\n",
            ("body",),
        )
        widget.insert(
            tk.END,
            "[pause=0.35]  |  [pause=350ms]  |  [pause=default]\n",
            ("code",),
        )
        widget.insert(
            tk.END,
            "Mix voices with commas:\n",
            ("body",),
        )
        widget.insert(
            tk.END,
            "[voice=af_heart,am_michael]\n",
            ("code",),
        )
        widget.configure(state=tk.DISABLED)

    def _build_stream_tab(self, parent: ttk.Frame) -> None:
        assert self.stream_status_var is not None
        frame = ttk.Frame(parent, padding=8)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, textvariable=self.stream_status_var, wraplength=760).pack(anchor="w")
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", pady=(8, 8))
        self.stream_btn = ttk.Button(btn_row, text="Stream", style="Primary.TButton", command=self._on_stream_start)
        self.stop_stream_btn = ttk.Button(btn_row, text="Stop", command=self._on_stream_stop)
        self.stream_btn.pack(side="left")
        self.stop_stream_btn.pack(side="left", padx=(8, 0))
        self.stop_stream_btn.state(["disabled"])
        ttk.Label(
            frame,
            text="Desktop stream runs in worker threads with Stop support.",
            wraplength=760,
            justify="left",
        ).pack(anchor="w")

    def _build_lesson_tab(self, parent: ttk.Frame) -> None:
        assert self.llm_base_url_var is not None
        assert self.llm_api_key_var is not None
        assert self.llm_model_var is not None
        assert self.llm_temperature_var is not None
        assert self.llm_max_tokens_var is not None
        assert self.llm_timeout_seconds_var is not None
        assert self.lesson_status_var is not None
        frame = ttk.Frame(parent, padding=8)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=(
                "Transform raw material into an English lesson script with detailed "
                "exercise explanations for TTS narration."
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w")
        lesson_raw_wrap, self.lesson_raw_text = self._create_text_with_scrollbar(
            frame,
            wrap=tk.WORD,
            height=10,
            font=("Courier New", 10),
        )
        lesson_raw_wrap.pack(fill="x", pady=(8, 8))

        settings = self._create_accordion_section(
            frame,
            title="LM Studio settings",
            expanded=False,
        )
        self._add_labeled_widget(
            settings,
            "Base URL",
            lambda row: ttk.Entry(row, textvariable=self.llm_base_url_var, width=50),
        )
        self._add_labeled_widget(
            settings,
            "Model",
            lambda row: ttk.Entry(row, textvariable=self.llm_model_var, width=50),
        )
        self._add_labeled_widget(
            settings,
            "API key",
            lambda row: ttk.Entry(row, textvariable=self.llm_api_key_var, width=50, show="*"),
        )
        self._add_labeled_widget(
            settings,
            "Temperature",
            lambda row: ttk.Spinbox(
                row,
                from_=0.0,
                to=2.0,
                increment=0.05,
                textvariable=self.llm_temperature_var,
                width=12,
            ),
        )
        self._add_labeled_widget(
            settings,
            "Max tokens",
            lambda row: ttk.Spinbox(
                row,
                from_=64,
                to=32768,
                increment=64,
                textvariable=self.llm_max_tokens_var,
                width=12,
            ),
        )
        self._add_labeled_widget(
            settings,
            "Timeout (s)",
            lambda row: ttk.Spinbox(
                row,
                from_=0,
                to=86400,
                increment=5,
                textvariable=self.llm_timeout_seconds_var,
                width=12,
            ),
        )
        ttk.Label(settings, text="Extra instructions").pack(anchor="w", padx=8, pady=(0, 2))
        extra_wrap, self.llm_extra_instructions_text = self._create_text_with_scrollbar(
            settings,
            wrap=tk.WORD,
            height=4,
            font=("Courier New", 10),
        )
        extra_wrap.pack(fill="x", padx=8, pady=(0, 8))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", pady=(0, 8))
        ttk.Button(btn_row, text="Generate lesson", style="Primary.TButton", command=self._on_lesson_generate).pack(
            side="left"
        )
        ttk.Button(btn_row, text="Use lesson as TTS input", command=self._on_lesson_copy_to_input).pack(
            side="left",
            padx=(8, 0),
        )
        lesson_output_wrap, self.lesson_output_text = self._create_text_with_scrollbar(
            frame,
            wrap=tk.WORD,
            height=14,
            font=("Courier New", 10),
        )
        lesson_output_wrap.pack(fill="both", expand=True)
        ttk.Label(frame, textvariable=self.lesson_status_var, wraplength=760).pack(
            fill="x",
            pady=(6, 0),
        )

    def _build_morph_tab(self, parent: ttk.Frame) -> None:
        assert self.morph_dataset_var is not None
        assert self.morph_limit_var is not None
        assert self.morph_offset_var is not None
        assert self.morph_status_var is not None
        assert self.selected_morph_row_var is not None
        frame = ttk.Frame(parent, padding=8)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=(
                "Simple CRUD for morphology.sqlite3: browse rows, add JSON row, "
                "select a row in the table, then click Update/Delete."
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w")

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(8, 8))
        ttk.Label(controls, text="Dataset").pack(side="left")
        dataset_combo = ttk.Combobox(
            controls,
            textvariable=self.morph_dataset_var,
            state="readonly",
            values=["occurrences", "lexemes", "expressions", "reviews"],
            width=16,
        )
        dataset_combo.pack(side="left", padx=(8, 12))
        dataset_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_morph_refresh())
        ttk.Label(controls, text="Limit").pack(side="left")
        ttk.Spinbox(
            controls,
            from_=1,
            to=1000,
            increment=1,
            textvariable=self.morph_limit_var,
            width=8,
        ).pack(side="left", padx=(8, 12))
        ttk.Label(controls, text="Offset").pack(side="left")
        ttk.Spinbox(
            controls,
            from_=0,
            to=1000000,
            increment=1,
            textvariable=self.morph_offset_var,
            width=8,
        ).pack(side="left", padx=(8, 12))
        ttk.Button(controls, text="Refresh", command=self._on_morph_refresh).pack(side="left")

        table_wrap = ttk.Frame(frame)
        table_wrap.pack(fill="both", expand=True)
        self.morph_tree = ttk.Treeview(table_wrap, show="headings", style="Treeview")
        y_scroll = self._create_scrollbar(table_wrap, orient=tk.VERTICAL, command=self.morph_tree.yview)
        x_scroll = self._create_scrollbar(table_wrap, orient=tk.HORIZONTAL, command=self.morph_tree.xview)
        self.morph_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.morph_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_wrap.rowconfigure(0, weight=1)
        table_wrap.columnconfigure(0, weight=1)
        self.morph_tree.bind("<<TreeviewSelect>>", self._on_morph_select_row)

        ttk.Label(frame, textvariable=self.morph_status_var, wraplength=760).pack(
            fill="x",
            pady=(6, 6),
        )
        add_box = self._create_accordion_section(frame, title="Add row", expanded=False)
        morph_add_wrap, self.morph_add_json_text = self._create_text_with_scrollbar(
            add_box,
            wrap=tk.WORD,
            height=5,
            font=("Courier New", 10),
        )
        self.morph_add_json_text.insert("1.0", '{"source":"manual"}')
        morph_add_wrap.pack(fill="x", padx=8, pady=8)
        ttk.Button(add_box, text="Add row", style="Primary.TButton", command=self._on_morph_add).pack(
            anchor="w",
            padx=8,
            pady=(0, 8),
        )

        update_box = self._create_accordion_section(frame, title="Update / Delete row", expanded=False)
        selected_row = ttk.Frame(update_box)
        selected_row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(selected_row, text="Selected row id/key").pack(side="left")
        ttk.Entry(selected_row, textvariable=self.selected_morph_row_var, state="readonly", width=30).pack(
            side="left",
            padx=(8, 0),
        )
        morph_update_wrap, self.morph_update_json_text = self._create_text_with_scrollbar(
            update_box,
            wrap=tk.WORD,
            height=5,
            font=("Courier New", 10),
        )
        self.morph_update_json_text.insert("1.0", '{"source":"manual"}')
        morph_update_wrap.pack(fill="x", padx=8, pady=(0, 8))
        btn_row = ttk.Frame(update_box)
        btn_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btn_row, text="Update row", command=self._on_morph_update).pack(side="left")
        ttk.Button(btn_row, text="Delete row", command=self._on_morph_delete).pack(side="left", padx=(8, 0))

        self._apply_table_update({"headers": ["No data"], "value": [[]]})

    @staticmethod
    def _add_labeled_widget(
        parent: ttk.LabelFrame | ttk.Frame,
        label: str,
        widget_factory: Callable[[ttk.Frame], tk.Widget],
    ) -> tk.Widget:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", padx=14, pady=(6, 8))
        ttk.Label(row, text=label, width=22, anchor="w", style="Card.TLabel").pack(side="left")
        widget = widget_factory(row)
        widget.pack(side="left", fill="x", expand=True, padx=(12, 12), ipady=2)
        return widget

    @staticmethod
    def _add_labeled_scale(
        parent: ttk.LabelFrame | ttk.Frame,
        *,
        label: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float,
    ) -> None:
        _ = resolution
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="x", padx=14, pady=(6, 8))
        ttk.Label(frame, text=label, width=22, anchor="w", style="Card.TLabel").pack(side="left")
        scale = ttk.Scale(frame, from_=from_, to=to, variable=variable)
        scale.pack(side="left", fill="x", expand=True, padx=(12, 12))
        value_label = ttk.Label(frame, text=f"{variable.get():.2f}", width=6, style="Card.TLabel")
        value_label.pack(side="left", padx=(0, 12))
        variable.trace_add(
            "write",
            lambda *_args: value_label.configure(text=f"{float(variable.get()):.2f}"),
        )

    def _on_close(self) -> None:
        self._save_audio_player_state()
        self.stream_stop_event.set()
        self._stop_generate_timer()
        self._stop_audio(preserve_player_position=False)
        self._audio_player_cancel_tick()
        self._audio_player_release_vlc()
        if self.root is not None:
            self.root.destroy()

    def _language_code_from_display(self, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            return self.default_lang
        if normalized in self.language_display_to_code:
            return self.language_display_to_code[normalized]
        if normalized in self.language_order:
            return normalized
        if "(" in normalized and normalized.endswith(")"):
            code = normalized.rsplit("(", 1)[-1].rstrip(")").strip().lower()
            if code in self.language_order:
                return code
        return self.default_lang

    def _language_display_from_code(self, code: str) -> str:
        normalized = normalize_lang_code(code, default=self.default_lang)
        return self.language_code_to_display.get(normalized, f"{normalized}")

    def _on_language_change(self) -> None:
        assert self.language_var is not None
        assert self.language_display_var is not None
        assert self.voice_var is not None
        selected_lang = normalize_lang_code(
            self._language_code_from_display(self.language_display_var.get()),
            default=self.default_lang,
        )
        options = get_voice_choices(selected_lang)
        voice_ids = [voice_id for _, voice_id in options]
        if not voice_ids:
            voice_ids = [self.default_voice]
        current_voice = self.voice_var.get()
        if current_voice not in voice_ids:
            current_voice = default_voice_for_lang(selected_lang)
        old_mix = set(self._selected_mix_voices())
        self.current_voice_choices = options
        self.current_voice_ids = voice_ids
        if self.voice_combo is not None:
            self.voice_combo.configure(values=voice_ids)
        self.language_var.set(selected_lang)
        self.language_display_var.set(self._language_display_from_code(selected_lang))
        self.voice_var.set(current_voice)
        self._set_mix_listbox_values(voice_ids, selected=[voice for voice in voice_ids if voice in old_mix])

    def _on_voice_change(self) -> None:
        assert self.language_var is not None
        assert self.language_display_var is not None
        assert self.voice_var is not None
        selected_voice = self.voice_var.get()
        language_code = voice_language(selected_voice, default=self.default_lang)
        self.language_var.set(language_code)
        self.language_display_var.set(self._language_display_from_code(language_code))

    def _on_mix_change(self) -> None:
        assert self.language_var is not None
        assert self.language_display_var is not None
        selected_mix = self._selected_mix_voices()
        if selected_mix:
            language_code = voice_language(selected_mix[0], default=self.default_lang)
            self.language_var.set(language_code)
            self.language_display_var.set(self._language_display_from_code(language_code))

    def _on_mix_toggle(self) -> None:
        enabled = bool(self.mix_enabled_var.get())
        if self.voice_mix_listbox is not None:
            self.voice_mix_listbox.configure(state="normal" if enabled else "disabled")
        if self.voice_combo is not None:
            self.voice_combo.configure(state="disabled" if enabled else "readonly")

    def _set_mix_listbox_values(self, voice_ids: list[str], *, selected: list[str]) -> None:
        if self.voice_mix_listbox is None:
            return
        self.voice_mix_listbox.delete(0, tk.END)
        for voice_id in voice_ids:
            self.voice_mix_listbox.insert(tk.END, voice_id)
        selected_set = set(selected)
        for index, voice_id in enumerate(voice_ids):
            if voice_id in selected_set:
                self.voice_mix_listbox.selection_set(index)

    def _selected_mix_voices(self) -> list[str]:
        if self.voice_mix_listbox is None:
            return []
        selected = []
        for index in self.voice_mix_listbox.curselection():
            if 0 <= index < len(self.current_voice_ids):
                selected.append(self.current_voice_ids[index])
        return selected

    def _on_runtime_mode_change(self) -> None:
        mode = self.runtime_mode_var.get()
        self._set_runtime_mode(mode, apply_backend=True)

    def _set_runtime_mode(self, mode_value: Any, *, apply_backend: bool) -> None:
        selected_mode = normalize_runtime_mode(mode_value)
        self.runtime_mode_var.set(selected_mode)
        if apply_backend:
            if selected_mode == RUNTIME_MODE_DEFAULT:
                self._set_tts_only_mode_wrapped(True)
                self._set_llm_only_mode_wrapped(False)
            elif selected_mode == RUNTIME_MODE_TTS_MORPH:
                self._set_tts_only_mode_wrapped(False)
                self._set_llm_only_mode_wrapped(True)
            else:
                self._set_tts_only_mode_wrapped(False)
                self._set_llm_only_mode_wrapped(False)
        self.runtime_mode_status_var.set(runtime_mode_status_text(selected_mode))
        self._sync_runtime_tabs(selected_mode)
        self._sync_hardware_selector_visibility()

    def _llm_runs_on_cpu(self) -> bool:
        llm_device = str(
            os.getenv("LM_STUDIO_DEVICE", "") or os.getenv("LLM_DEVICE", "")
        ).strip().lower()
        if llm_device in {"gpu", "cuda"}:
            return False
        if llm_device:
            return True
        return not self.cuda_available

    def _sync_hardware_selector_visibility(self) -> None:
        if self.hardware_section_frame is None or self.hardware_var is None:
            return
        if self._llm_runs_on_cpu():
            self.hardware_var.set("CPU")
            self.hardware_section_frame.grid_remove()
            return
        self.hardware_section_frame.grid()

    def _set_tts_only_mode_wrapped(self, enabled: bool) -> str:
        self.runtime_tts_only_enabled = bool(enabled)
        if callable(self.set_tts_only_mode):
            status = self.set_tts_only_mode(self.runtime_tts_only_enabled)
            return str(status or tts_only_mode_status_text(self.runtime_tts_only_enabled))
        return tts_only_mode_status_text(self.runtime_tts_only_enabled)

    def _set_llm_only_mode_wrapped(self, enabled: bool) -> str:
        self.runtime_llm_only_enabled = bool(enabled)
        if callable(self.set_llm_only_mode):
            status = self.set_llm_only_mode(self.runtime_llm_only_enabled)
            return str(
                status
                or llm_only_mode_status_text(
                    self.runtime_llm_only_enabled,
                    tts_only_enabled=self.runtime_tts_only_enabled,
                )
            )
        return llm_only_mode_status_text(
            self.runtime_llm_only_enabled,
            tts_only_enabled=self.runtime_tts_only_enabled,
        )

    def _sync_runtime_tabs(self, selected_mode: str) -> None:
        if self.notebook is None:
            return
        lesson_visible, morph_visible = runtime_mode_tab_visibility(selected_mode)
        self._set_tab_visible("lesson", lesson_visible)
        self._set_tab_visible("morph", morph_visible)

    def _set_tab_visible(self, tab_key: str, visible: bool) -> None:
        if self.notebook is None:
            return
        frame, title = self.tabs[tab_key]
        tab_id = str(frame)
        existing = set(self.notebook.tabs())
        if visible and tab_id not in existing:
            self.notebook.add(frame, text=title)
        elif not visible and tab_id in existing:
            self.notebook.forget(frame)

    def _threaded(self, work: Callable[[], Any], on_success: Callable[[Any], None] | None = None) -> None:
        def _runner() -> None:
            try:
                result = work()
            except Exception as exc:  # pragma: no cover - defensive UI path
                self.logger.exception("Tkinter UI action failed")
                message = str(exc)
                self._run_on_ui(lambda message=message: self._set_error_status(message))
                return
            if on_success is not None:
                self._run_on_ui(lambda: on_success(result))

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

    def _run_on_ui(self, callback: Callable[[], None]) -> None:
        if self.root is None:
            return
        self.root.after(0, callback)

    def _set_error_status(self, message: str) -> None:
        self.generate_status_var.set(f"Error: {message}")
        if self.generate_in_progress:
            self._set_generate_button_processing(False)

    def _set_generate_button_processing(self, in_progress: bool) -> None:
        self.generate_in_progress = bool(in_progress)
        if self.generate_btn is None:
            return
        if self.generate_in_progress:
            self.generate_btn.configure(text="in process...")
            self.generate_btn.state(["disabled"])
            self._start_generate_timer()
        else:
            self.generate_btn.configure(text="Generate")
            self.generate_btn.state(["!disabled"])
            self._stop_generate_timer()

    def _start_generate_timer(self) -> None:
        self._stop_generate_timer()
        if self.root is None:
            return
        self.generate_started_at = time.perf_counter()
        self._update_generate_timer()

    def _stop_generate_timer(self) -> None:
        if self.root is not None and self.generate_timer_job:
            try:
                self.root.after_cancel(self.generate_timer_job)
            except Exception:
                pass
        self.generate_timer_job = None
        self.generate_started_at = 0.0

    def _update_generate_timer(self) -> None:
        if (
            self.root is None
            or self.generate_status_var is None
            or not self.generate_in_progress
            or self.generate_started_at <= 0.0
        ):
            self.generate_timer_job = None
            return
        elapsed = max(0.0, time.perf_counter() - self.generate_started_at)
        self.generate_status_var.set(f"{elapsed:.1f} s")
        self.generate_timer_job = self.root.after(100, self._update_generate_timer)

    def _read_text(self, widget: tk.Text | None) -> str:
        if widget is None:
            return ""
        return widget.get("1.0", tk.END).strip()

    def _write_text(self, widget: tk.Text | None, text: str) -> None:
        if widget is None:
            return
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text or "")

    def _base_generation_kwargs(self) -> dict[str, Any]:
        return {
            "text": self._read_text(self.input_text),
            "voice": self.voice_var.get(),
            "mix_enabled": bool(self.mix_enabled_var.get()),
            "voice_mix": self._selected_mix_voices(),
            "speed": float(self.speed_var.get()),
            "use_gpu": self.hardware_var.get() == "GPU",
            "pause_seconds": float(self.pause_var.get()),
            "normalize_times_enabled": bool(self.normalize_times_var.get()),
            "normalize_numbers_enabled": bool(self.normalize_numbers_var.get()),
            "style_preset": self.style_var.get() or DEFAULT_STYLE_PRESET,
        }

    def _on_generate(self) -> None:
        if self.generate_in_progress:
            return
        self._set_generate_button_processing(True)

        def work():
            kwargs = self._base_generation_kwargs()
            kwargs["output_format"] = self.output_format_var.get()
            result, tokens = self.generate_first(**kwargs)
            updated_history = (
                self.history_service.update_history(self.history_state)
                if self.history_service is not None
                else self.history_state
            )
            return result, tokens, list(updated_history)

        def on_success(payload: tuple[Any, str, list[str]]) -> None:
            try:
                result, tokens, updated_history = payload
                self.history_state = updated_history
                self._render_history()
                self._write_text(self.token_output_text, tokens)
                if result is None:
                    self.generate_status_var.set("No audio generated.")
                    return
                if self.generate_detail_notebook is not None and self.generate_detail_tabs.get("player") is not None:
                    self.generate_detail_notebook.select(self.generate_detail_tabs["player"])
                self.generate_status_var.set("Generation complete. Loading latest audio...")
                if not self._autoplay_latest_history():
                    self.generate_status_var.set("Generation complete, but no playable file found in History.")
            finally:
                self._set_generate_button_processing(False)

        self._threaded(work, on_success)

    def _on_tokenize(self) -> None:
        self.generate_status_var.set("Tokenizing...")

        def work():
            kwargs = self._base_generation_kwargs()
            kwargs.pop("use_gpu", None)
            kwargs.pop("pause_seconds", None)
            return self.tokenize_first(**kwargs)

        def on_success(tokens: str) -> None:
            self._write_text(self.token_output_text, tokens)
            self.generate_status_var.set("Tokenization complete.")

        self._threaded(work, on_success)

    def _stop_audio(self, *, preserve_player_position: bool = True) -> None:
        if self.vlc_audio is not None:
            try:
                if preserve_player_position:
                    if self.audio_player_is_playing:
                        self.vlc_audio.set_pause(True)
                else:
                    self.vlc_audio.stop()
            except Exception:
                self.logger.exception("Failed to control VLC player")

        if preserve_player_position:
            current_seconds = self._audio_player_current_seconds()
            self.audio_player_current_frame = int(current_seconds * float(max(self.audio_player_sample_rate, 1)))
        if self.audio_player_is_playing:
            self.audio_player_is_playing = False
            self.audio_player_is_paused = bool(preserve_player_position)
            if not preserve_player_position:
                self.audio_player_current_frame = 0
        elif not preserve_player_position:
            self.audio_player_current_frame = 0
            self.audio_player_is_paused = False
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                self.logger.exception("Failed to stop sounddevice playback")
        self.audio_player_sd_start_frame = 0
        self.audio_player_sd_started_at = 0.0
        self._audio_player_cancel_tick()
        self._audio_player_update_progress()
        self._update_audio_player_buttons()

    def _render_history(self) -> None:
        if self.history_listbox is None:
            return
        self.history_listbox.delete(0, tk.END)
        for path in self.history_state:
            self.history_listbox.insert(tk.END, path)

    def _on_clear_history(self) -> None:
        if self.history_service is None:
            self.history_state = []
            self.audio_player_queue_index = None
            self._render_history()
            self._save_audio_player_state()
            return

        def work():
            return self.history_service.clear_history(self.history_state)

        def on_success(updated: list[str]) -> None:
            self.history_state = list(updated or [])
            self.audio_player_queue_index = None
            self._render_history()
            self.generate_status_var.set("History cleared.")
            self._save_audio_player_state()

        self._threaded(work, on_success)

    def _on_history_select_autoplay(self) -> None:
        selected_item = self._selected_history_item(show_errors=False)
        if selected_item is None:
            return
        index, target = selected_item
        self._load_audio_file_async(target, autoplay=True, history_index=index)

    def _on_history_double_click(self, event: tk.Event[Any]) -> None:
        if self.history_listbox is None:
            return
        try:
            index = int(self.history_listbox.nearest(event.y))
        except Exception:
            index = -1
        if index < 0 or index >= len(self.history_state):
            return
        self._select_history_index(index)
        selected_item = self._selected_history_item(show_errors=False)
        if selected_item is None:
            return
        history_index, target = selected_item
        self._open_player_tab(maximize_player=True)
        self._load_audio_file_async(target, autoplay=True, history_index=history_index)

    def _open_player_tab(self, *, maximize_player: bool) -> None:
        if self.notebook is not None and "generate" in self.tabs:
            try:
                self.notebook.select(self.tabs["generate"][0])
            except Exception:
                self.logger.exception("Failed to switch to Generate tab")
        if self.generate_detail_notebook is not None:
            player_tab = self.generate_detail_tabs.get("player")
            if player_tab is not None:
                try:
                    self.generate_detail_notebook.select(player_tab)
                except Exception:
                    self.logger.exception("Failed to switch to Player tab")
        if maximize_player and self.audio_player_minimal_var is not None:
            self.audio_player_minimal_var.set(False)
            self._apply_audio_player_minimal_mode()

    def _on_audio_player_open(self, *, autoplay: bool = False) -> None:
        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.ogg *.flac *.m4a *.aac *.opus"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        target = Path(path)
        history_index = self._find_history_index(target)
        if history_index is not None:
            self._select_history_index(history_index)
        self._load_audio_file_async(target, autoplay=bool(autoplay), history_index=history_index)

    def _autoplay_latest_history(self) -> bool:
        if not self.history_state:
            return False
        for index, value in enumerate(self.history_state):
            target = Path(value)
            if not target.is_file():
                continue
            self._select_history_index(index)
            self._load_audio_file_async(target, autoplay=True, history_index=index)
            return True
        return False

    def _selected_history_item(self, *, show_errors: bool) -> tuple[int, Path] | None:
        if self.history_listbox is None:
            return None
        selected = self.history_listbox.curselection()
        if not selected:
            if show_errors:
                self.generate_status_var.set("Select a history item first.")
            return None
        index = selected[0]
        if index < 0 or index >= len(self.history_state):
            if show_errors:
                self.generate_status_var.set("Selected history item is out of range.")
            return None
        target = Path(self.history_state[index])
        if not target.is_file():
            if show_errors:
                self.generate_status_var.set("History file does not exist.")
            return None
        return index, target

    def _select_history_index(self, index: int) -> None:
        if self.history_listbox is None:
            return
        try:
            self.history_listbox.selection_clear(0, tk.END)
            self.history_listbox.selection_set(index)
            self.history_listbox.activate(index)
            self.history_listbox.see(index)
        except Exception:
            self.logger.exception("Failed to update history selection")

    def _load_audio_file_async(
        self,
        path: Path,
        *,
        autoplay: bool,
        history_index: int | None = None,
        resume_seconds: float | None = None,
    ) -> None:
        assert self.audio_player_status_var is not None
        self.audio_player_status_var.set(f"Loading {path.name}...")

        def runner() -> None:
            waveform_audio: np.ndarray | None = None
            sample_rate = 0
            total_frames = 0
            waveform_warning = ""
            if sf is not None:
                try:
                    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
                    audio_np = np.asarray(audio, dtype=np.float32)
                    if audio_np.size > 0:
                        if audio_np.ndim not in (1, 2):
                            raise RuntimeError("Unsupported audio shape.")
                        waveform_audio = audio_np
                        sample_rate = int(sr)
                        total_frames = int(audio_np.shape[0])
                except Exception as exc:
                    self.logger.exception("Failed to read waveform data")
                    waveform_warning = str(exc)

            self._run_on_ui(
                lambda waveform_audio=waveform_audio, sample_rate=sample_rate, total_frames=total_frames, waveform_warning=waveform_warning: self._on_audio_file_loaded(
                    path=path,
                    waveform_audio=waveform_audio,
                    sample_rate=int(sample_rate),
                    total_frames=total_frames,
                    autoplay=autoplay,
                    history_index=history_index,
                    resume_seconds=resume_seconds,
                    waveform_warning=waveform_warning,
                )
            )

        threading.Thread(target=runner, daemon=True).start()

    def _on_audio_file_loaded(
        self,
        *,
        path: Path,
        waveform_audio: np.ndarray | None,
        sample_rate: int,
        total_frames: int,
        autoplay: bool,
        history_index: int | None = None,
        resume_seconds: float | None = None,
        waveform_warning: str = "",
    ) -> None:
        assert self.audio_player_status_var is not None
        assert self.audio_player_track_var is not None
        self._stop_audio(preserve_player_position=False)
        self.audio_player_loaded_path = path
        self.audio_player_track_var.set(path.name)
        if waveform_audio is not None and sample_rate > 0 and total_frames > 0:
            self.audio_player_pcm_data = np.asarray(waveform_audio, dtype=np.float32)
            self.audio_player_sample_rate = int(sample_rate)
            self.audio_player_total_frames = int(total_frames)
            self._audio_player_rebuild_waveform(waveform_audio)
        else:
            self.audio_player_pcm_data = None
            self.audio_player_sample_rate = 1000
            self.audio_player_total_frames = 0
            self.audio_player_waveform = None
            self._audio_player_redraw_waveform()
        self.audio_player_queue_index = history_index if history_index is not None else self._find_history_index(path)
        if self.audio_player_queue_index is not None:
            self._select_history_index(self.audio_player_queue_index)
        self.audio_player_media_length_ms = 0
        self.audio_player_backend = "vlc"
        self.audio_player_current_frame = 0
        restore_seconds = self._coerce_float(
            resume_seconds if resume_seconds is not None else 0.0,
            default=0.0,
            min_value=0.0,
        )
        self.audio_player_is_playing = False
        self.audio_player_is_paused = restore_seconds > 0
        media_ready = False
        if vlc is not None:
            media_ready = self._audio_player_set_media(path)
        if not media_ready and not self._audio_player_can_use_sounddevice():
            self.audio_player_status_var.set(
                "No playback backend available. Install VLC runtime or keep soundfile/sounddevice enabled."
            )
            self._audio_player_update_progress()
            self._update_audio_player_buttons()
            self._save_audio_player_state()
            return
        if not media_ready:
            self.audio_player_backend = "sounddevice"
        total_seconds = self._audio_player_total_seconds(refresh=True)
        if total_seconds > 0:
            if self.audio_player_waveform is None:
                self.audio_player_sample_rate = 1000
                self.audio_player_total_frames = int(total_seconds * 1000.0)
            restore_seconds = min(restore_seconds, total_seconds)
        self.audio_player_current_frame = int(restore_seconds * float(max(self.audio_player_sample_rate, 1)))
        if self.audio_player_total_frames > 0:
            self.audio_player_current_frame = max(0, min(self.audio_player_total_frames, self.audio_player_current_frame))
        self._audio_player_update_progress()
        duration_s = self._audio_player_total_seconds()
        resume_text = ""
        if self.audio_player_current_frame > 0 and not autoplay:
            position_text = self._audio_player_format_timestamp(restore_seconds)
            resume_text = f", resume {position_text}"
        status = f"Loaded {path.name} ({duration_s:.1f}s{resume_text})."
        if waveform_warning:
            status = f"{status} Waveform unavailable."
        self.audio_player_status_var.set(status)
        self.generate_status_var.set(f"Audio file loaded: {path.name}")
        self._update_audio_player_buttons()
        self._save_audio_player_state()
        if autoplay:
            self._on_audio_player_play()

    def _ensure_vlc_player(self) -> bool:
        assert self.audio_player_status_var is not None
        if vlc is None:
            self.audio_player_status_var.set("python-vlc is not installed. Install dependencies to enable Audio player.")
            return False
        if self.vlc_audio is None:
            try:
                self.vlc_audio = VlcAudioBackend()
            except Exception as exc:
                self.logger.exception("Failed to create VLC backend")
                self.audio_player_status_var.set(f"VLC init failed: {exc}")
                return False
        return True

    def _audio_player_release_vlc(self) -> None:
        if self.vlc_audio is not None:
            try:
                self.vlc_audio.release()
            except Exception:
                self.logger.exception("Failed to release VLC backend")
            self.vlc_audio = None

    def _audio_player_set_media(self, path: Path) -> bool:
        assert self.audio_player_status_var is not None
        assert self.audio_player_volume_var is not None
        if not self._ensure_vlc_player():
            return False
        try:
            assert self.vlc_audio is not None
            self.vlc_audio.load(str(path))
            volume = self._coerce_float(self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5)
            self.vlc_audio.set_volume(int(round(volume * 100.0)))
            self.audio_player_media_length_ms = 0
        except Exception as exc:
            self.logger.exception("Failed to set VLC media")
            self.audio_player_status_var.set(f"VLC media error: {exc}")
            return False
        return True

    def _on_audio_player_play(self) -> None:
        assert self.audio_player_status_var is not None
        if self.audio_player_loaded_path is None:
            self._on_audio_player_open(autoplay=True)
            return
        if self.audio_player_current_frame >= self.audio_player_total_frames:
            self.audio_player_current_frame = 0
        if self.audio_player_is_playing:
            return
        if not self._audio_player_start_playback(self.audio_player_current_frame):
            return
        total_stamp = self._audio_player_format_timestamp(self._audio_player_total_seconds(refresh=True))
        self.audio_player_status_var.set(
            f"Playing {self.audio_player_loaded_path.name} ({total_stamp})."
        )
        if self.audio_player_track_var is not None and self.audio_player_loaded_path is not None:
            self.audio_player_track_var.set(self.audio_player_loaded_path.name)
        self.generate_status_var.set(f"Playing {self.audio_player_loaded_path.name}")
        self._save_audio_player_state()

    def _audio_player_start_playback(self, start_frame: int) -> bool:
        assert self.audio_player_status_var is not None
        if self.audio_player_loaded_path is None:
            self.audio_player_status_var.set("Load a file first.")
            return False
        frame = int(max(0, start_frame))
        if vlc is None:
            return self._audio_player_start_playback_sounddevice(frame)
        if not self._ensure_vlc_player():
            if self._audio_player_can_use_sounddevice():
                return self._audio_player_start_playback_sounddevice(frame)
            return False
        start_seconds = float(frame) / float(max(self.audio_player_sample_rate, 1))
        target_ms = int(max(0.0, start_seconds * 1000.0))
        volume = self._coerce_float(self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5)
        try:
            assert self.vlc_audio is not None
            self.vlc_audio.set_volume(int(round(volume * 100.0)))
            self.vlc_audio.play()
            if target_ms > 0:
                # VLC may ignore immediate seek until playback thread is ready.
                self.vlc_audio.set_time_ms(target_ms)
                if self.root is not None:
                    self.root.after(140, lambda target_ms=target_ms: self._audio_player_seek_vlc_ms(target_ms))
        except Exception as exc:
            self.logger.exception("Audio player playback failed")
            if self._audio_player_can_use_sounddevice():
                return self._audio_player_start_playback_sounddevice(frame)
            self.audio_player_status_var.set(f"Playback failed: {exc}")
            return False
        self.audio_player_backend = "vlc"
        self.audio_player_sd_start_frame = 0
        self.audio_player_sd_started_at = 0.0
        self.audio_player_current_frame = frame
        self.audio_player_is_playing = True
        self.audio_player_is_paused = False
        self._audio_player_total_seconds(refresh=True)
        self._audio_player_update_progress()
        self._audio_player_schedule_tick()
        self._update_audio_player_buttons()
        return True

    def _audio_player_can_use_sounddevice(self) -> bool:
        return sd is not None and self.audio_player_pcm_data is not None and self.audio_player_sample_rate > 0

    def _audio_player_start_playback_sounddevice(self, frame: int) -> bool:
        assert self.audio_player_status_var is not None
        if not self._audio_player_can_use_sounddevice():
            self.audio_player_status_var.set(
                "Playback backend unavailable. Install VLC or ensure sounddevice+soundfile are available."
            )
            return False
        assert self.audio_player_volume_var is not None
        assert self.audio_player_pcm_data is not None
        frame = int(max(0, min(self.audio_player_total_frames, frame)))
        if frame >= self.audio_player_total_frames:
            frame = 0
        if self.audio_player_pcm_data.ndim == 1:
            chunk = self.audio_player_pcm_data[frame:]
        else:
            chunk = self.audio_player_pcm_data[frame:, :]
        if chunk.size == 0:
            self.audio_player_status_var.set("Nothing to play.")
            return False
        volume = self._coerce_float(self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5)
        prepared = np.asarray(chunk, dtype=np.float32)
        if abs(volume - 1.0) > 1e-6:
            prepared = np.clip(prepared * float(volume), -1.0, 1.0)
        try:
            sd.play(prepared, samplerate=int(self.audio_player_sample_rate), blocking=False)
        except Exception as exc:
            self.logger.exception("Sounddevice playback failed")
            self.audio_player_status_var.set(f"Playback failed: {exc}")
            return False
        self.audio_player_backend = "sounddevice"
        self.audio_player_sd_start_frame = frame
        self.audio_player_sd_started_at = time.monotonic()
        self.audio_player_current_frame = frame
        self.audio_player_is_playing = True
        self.audio_player_is_paused = False
        self._audio_player_update_progress()
        self._audio_player_schedule_tick()
        self._update_audio_player_buttons()
        return True

    def _audio_player_seek_vlc_ms(self, target_ms: int) -> None:
        if self.vlc_audio is None:
            return
        try:
            self.vlc_audio.set_time_ms(int(max(0, target_ms)))
        except Exception:
            self.logger.exception("Failed to seek VLC player")

    def _on_audio_player_pause(self) -> None:
        assert self.audio_player_status_var is not None
        if not self.audio_player_is_playing:
            self.audio_player_status_var.set("Nothing is currently playing.")
            return
        if self.vlc_audio is not None:
            try:
                self.vlc_audio.set_pause(True)
            except Exception:
                self.logger.exception("Failed to pause VLC playback")
        if self.audio_player_backend == "sounddevice" and sd is not None:
            try:
                sd.stop()
            except Exception:
                self.logger.exception("Failed to pause sounddevice playback")
        self.audio_player_current_frame = int(self._audio_player_current_seconds() * float(max(self.audio_player_sample_rate, 1)))
        self.audio_player_is_playing = False
        self.audio_player_is_paused = True
        self._audio_player_cancel_tick()
        self._audio_player_update_progress()
        position = self._audio_player_format_timestamp(self._audio_player_current_seconds())
        total = self._audio_player_format_timestamp(self._audio_player_total_seconds())
        self.audio_player_status_var.set(f"Paused at {position} / {total}.")
        self._update_audio_player_buttons()
        self._save_audio_player_state()

    def _on_audio_player_stop(self) -> None:
        assert self.audio_player_status_var is not None
        self._stop_audio(preserve_player_position=False)
        self.audio_player_current_frame = 0
        self.audio_player_is_paused = False
        self._audio_player_update_progress()
        if self.audio_player_loaded_path is not None:
            self.audio_player_status_var.set(f"Stopped: {self.audio_player_loaded_path.name}")
            if self.audio_player_track_var is not None:
                self.audio_player_track_var.set(self.audio_player_loaded_path.name)
        else:
            self.audio_player_status_var.set("Stopped.")
            if self.audio_player_track_var is not None:
                self.audio_player_track_var.set("No file loaded.")
        self._update_audio_player_buttons()
        self._save_audio_player_state()

    def _on_audio_player_seek_back(self) -> None:
        self._audio_player_seek_relative(-float(self.audio_player_seek_step_seconds))

    def _on_audio_player_seek_forward(self) -> None:
        self._audio_player_seek_relative(float(self.audio_player_seek_step_seconds))

    def _audio_player_seek_relative(self, delta_seconds: float) -> None:
        if self.audio_player_loaded_path is None:
            return
        current_seconds = self._audio_player_current_seconds()
        self._audio_player_seek_to_seconds(current_seconds + float(delta_seconds))

    def _on_audio_player_seek_press(self, _event: tk.Event[Any]) -> None:
        self.audio_player_seek_dragging = True

    def _on_audio_player_seek_release(self, _event: tk.Event[Any]) -> None:
        self.audio_player_seek_dragging = False
        if self.audio_player_progress_var is None:
            return
        self._audio_player_seek_to_seconds(self.audio_player_progress_var.get())

    def _on_audio_player_seek_change(self, value: str) -> None:
        if self.audio_player_seek_programmatic:
            return
        if self.audio_player_time_var is None:
            return
        try:
            position_seconds = float(value)
        except Exception:
            return
        if position_seconds < 0:
            position_seconds = 0.0
        total_seconds = self._audio_player_total_seconds()
        position_seconds = min(position_seconds, total_seconds)
        if self.audio_player_seek_dragging:
            position_label = self._audio_player_format_timestamp(position_seconds)
            total_label = self._audio_player_format_timestamp(total_seconds)
            self.audio_player_time_var.set(f"{position_label} / {total_label}")

    def _audio_player_seek_to_seconds(self, seconds: float) -> None:
        assert self.audio_player_status_var is not None
        if self.audio_player_loaded_path is None:
            return
        total_seconds = self._audio_player_total_seconds(refresh=True)
        clamped_seconds = self._coerce_float(
            seconds,
            default=0.0,
            min_value=0.0,
            max_value=total_seconds if total_seconds > 0 else None,
        )
        if self.audio_player_total_frames <= 0 and total_seconds > 0:
            self.audio_player_sample_rate = max(1, self.audio_player_sample_rate)
            self.audio_player_total_frames = int(total_seconds * float(self.audio_player_sample_rate))
        target_frame = int(clamped_seconds * float(max(self.audio_player_sample_rate, 1)))
        if self.audio_player_total_frames > 0:
            target_frame = max(0, min(self.audio_player_total_frames, target_frame))
        else:
            target_frame = max(0, target_frame)
        was_playing = self.audio_player_is_playing
        if self.audio_player_backend == "vlc":
            self._audio_player_seek_vlc_ms(int(round(clamped_seconds * 1000.0)))
        elif self.audio_player_backend == "sounddevice" and sd is not None:
            try:
                sd.stop()
            except Exception:
                self.logger.exception("Failed to seek sounddevice playback")
        self.audio_player_current_frame = target_frame
        self.audio_player_is_playing = bool(was_playing)
        self.audio_player_is_paused = (target_frame > 0) and (not was_playing)
        if self.audio_player_backend == "sounddevice":
            if was_playing:
                if not self._audio_player_start_playback_sounddevice(target_frame):
                    self.audio_player_is_playing = False
                    self.audio_player_is_paused = target_frame > 0
            else:
                self.audio_player_sd_start_frame = target_frame
                self.audio_player_sd_started_at = 0.0
        self._audio_player_update_progress()
        if was_playing:
            self._audio_player_schedule_tick()
        else:
            self._update_audio_player_buttons()
        position_label = self._audio_player_format_timestamp(clamped_seconds)
        total_label = self._audio_player_format_timestamp(total_seconds)
        self.audio_player_status_var.set(f"Seek: {position_label} / {total_label}")
        self._save_audio_player_state()

    def _on_audio_player_waveform_seek(self, event: tk.Event[Any]) -> None:
        if self.audio_player_waveform_canvas is None:
            return
        width = max(1, int(self.audio_player_waveform_canvas.winfo_width()))
        ratio = max(0.0, min(1.0, float(event.x) / float(width)))
        seconds = ratio * self._audio_player_total_seconds()
        self._audio_player_seek_to_seconds(seconds)

    def _audio_player_schedule_tick(self) -> None:
        if self.root is None:
            return
        self._audio_player_cancel_tick()
        self.audio_player_tick_job = self.root.after(120, self._on_audio_player_tick)

    def _audio_player_cancel_tick(self) -> None:
        if self.root is None:
            self.audio_player_tick_job = None
            return
        if self.audio_player_tick_job is None:
            return
        try:
            self.root.after_cancel(self.audio_player_tick_job)
        except Exception:
            pass
        self.audio_player_tick_job = None

    def _on_audio_player_tick(self) -> None:
        self.audio_player_tick_job = None
        if not self.audio_player_is_playing:
            return
        total_seconds = self._audio_player_total_seconds(refresh=True)
        current_seconds = self._audio_player_current_seconds()
        if total_seconds > 0:
            ratio = max(0.0, min(1.0, current_seconds / total_seconds))
            if self.audio_player_total_frames <= 0:
                self.audio_player_sample_rate = 1000
                self.audio_player_total_frames = int(total_seconds * 1000.0)
            self.audio_player_current_frame = int(ratio * float(max(1, self.audio_player_total_frames)))
        else:
            self.audio_player_current_frame = int(current_seconds * float(max(self.audio_player_sample_rate, 1)))
        self._audio_player_update_progress()
        state = None
        if self.vlc_audio is not None:
            try:
                state = self.vlc_audio.get_state()
            except Exception:
                state = None
        ended_states = set()
        if vlc is not None:
            ended_states = {vlc.State.Ended, vlc.State.Stopped, vlc.State.Error}
        reached_end = total_seconds > 0 and current_seconds >= max(0.0, total_seconds - 0.05)
        if state in ended_states or reached_end:
            if self.audio_player_total_frames > 0:
                self.audio_player_current_frame = self.audio_player_total_frames
            self.audio_player_is_playing = False
            self.audio_player_is_paused = False
            if self._audio_player_try_auto_next():
                return
            if self.audio_player_status_var is not None:
                if self.audio_player_loaded_path is not None:
                    self.audio_player_status_var.set(f"Playback complete: {self.audio_player_loaded_path.name}")
                else:
                    self.audio_player_status_var.set("Playback complete.")
            self._update_audio_player_buttons()
            self._save_audio_player_state()
            return
        self._audio_player_schedule_tick()

    def _audio_player_update_progress(self) -> None:
        if self.audio_player_progress_var is None or self.audio_player_time_var is None:
            return
        total_seconds = self._audio_player_total_seconds(refresh=True)
        current_seconds = self._audio_player_current_seconds()
        if total_seconds <= 0:
            self.audio_player_seek_programmatic = True
            try:
                self.audio_player_progress_var.set(0.0)
            finally:
                self.audio_player_seek_programmatic = False
            self.audio_player_time_var.set("00:00 / 00:00")
            self._audio_player_redraw_waveform()
            return
        if self.audio_player_total_frames <= 0:
            self.audio_player_sample_rate = 1000
            self.audio_player_total_frames = int(total_seconds * 1000.0)
        self.audio_player_current_frame = int(
            max(0.0, min(1.0, current_seconds / max(total_seconds, 0.001))) * float(max(1, self.audio_player_total_frames))
        )
        if self.audio_player_seek_scale is not None:
            self.audio_player_seek_scale.configure(to=max(0.001, total_seconds))
        if not self.audio_player_seek_dragging:
            self.audio_player_seek_programmatic = True
            try:
                self.audio_player_progress_var.set(current_seconds)
            finally:
                self.audio_player_seek_programmatic = False
        position_label = self._audio_player_format_timestamp(current_seconds)
        total_label = self._audio_player_format_timestamp(total_seconds)
        self.audio_player_time_var.set(f"{position_label} / {total_label}")
        self._audio_player_redraw_waveform()

    def _update_audio_player_buttons(self) -> None:
        loaded = self.audio_player_loaded_path is not None
        if self.audio_player_play_btn is not None:
            self.audio_player_play_btn.state(["!disabled"] if loaded and not self.audio_player_is_playing else ["disabled"])
        if self.audio_player_pause_btn is not None:
            self.audio_player_pause_btn.state(["!disabled"] if self.audio_player_is_playing else ["disabled"])
        if self.audio_player_stop_btn is not None:
            can_stop = loaded and (self.audio_player_is_playing or self.audio_player_is_paused or self.audio_player_current_frame > 0)
            self.audio_player_stop_btn.state(["!disabled"] if can_stop else ["disabled"])

    def _on_audio_player_minimal_toggle(self) -> None:
        self._apply_audio_player_minimal_mode()

    def _apply_audio_player_minimal_mode(self) -> None:
        minimal = bool(self.audio_player_minimal_var.get()) if self.audio_player_minimal_var is not None else False
        managed_frames = (
            self.audio_player_track_frame,
            self.audio_player_controls_frame,
            self.audio_player_volume_frame,
            self.audio_player_autonext_frame,
            self.audio_player_status_frame,
        )
        for frame in managed_frames:
            if frame is None or not frame.winfo_exists():
                continue
            if minimal:
                frame.grid_remove()
            else:
                frame.grid()
        if self.audio_player_waveform_frame is not None and self.audio_player_waveform_frame.winfo_exists():
            self.audio_player_waveform_frame.grid_configure(pady=(6, 4) if minimal else (8, 6))
        if self.audio_player_seek_frame is not None and self.audio_player_seek_frame.winfo_exists():
            self.audio_player_seek_frame.grid_configure(pady=(0, 6) if minimal else (6, 6))

    def _sync_audio_player_control_labels(self) -> None:
        assert self.audio_player_volume_var is not None
        volume = self._coerce_float(self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5)
        if self.audio_player_volume_value_label is not None:
            percent = int(round(volume * 100.0))
            self.audio_player_volume_value_label.configure(text=f"{percent:>3d}%")

    def _on_audio_player_volume_scale(self) -> None:
        self._sync_audio_player_control_labels()
        if self.vlc_audio is not None and self.audio_player_volume_var is not None:
            try:
                volume = self._coerce_float(self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5)
                self.vlc_audio.set_volume(int(round(volume * 100.0)))
            except Exception:
                self.logger.exception("Failed to update VLC volume")
        self._save_audio_player_state()

    def _on_audio_player_volume_var_updated(self) -> None:
        self._sync_audio_player_control_labels()

    def _audio_player_rebuild_waveform(self, audio: np.ndarray) -> None:
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        mono = np.abs(mono).flatten()
        if mono.size == 0:
            self.audio_player_waveform = None
            self._audio_player_redraw_waveform()
            return
        target_bins = 400
        if mono.size < target_bins:
            padded = np.pad(mono, (0, target_bins - mono.size), mode="constant")
            envelope = padded
        else:
            stride = int(np.ceil(float(mono.size) / float(target_bins)))
            padded_size = stride * target_bins
            padded = np.pad(mono, (0, padded_size - mono.size), mode="constant")
            envelope = padded.reshape(target_bins, stride).max(axis=1)
        peak = float(np.max(envelope)) if envelope.size else 0.0
        if peak > 1e-8:
            envelope = envelope / peak
        self.audio_player_waveform = envelope.astype(np.float32, copy=False)
        self._audio_player_redraw_waveform()

    def _audio_player_redraw_waveform(self) -> None:
        canvas = self.audio_player_waveform_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
        canvas.delete("all")
        waveform = self.audio_player_waveform
        if waveform is None or waveform.size == 0:
            canvas.create_text(
                width // 2,
                height // 2,
                text="No waveform",
                fill="#6f7888",
                font=("Segoe UI", 9),
            )
            return
        midpoint = height / 2.0
        amplitude = max(2.0, (height / 2.0) - 4.0)
        played_limit = int(round(self._audio_player_progress_fraction() * float(width - 1)))
        played_color = "#00ff41"
        pending_color = "#1f4f1f"
        samples = waveform
        sample_count = samples.size
        for x in range(width):
            sample_index = int((x / max(1, width - 1)) * (sample_count - 1))
            value = float(samples[sample_index])
            half = amplitude * value
            color = played_color if x <= played_limit else pending_color
            canvas.create_line(x, midpoint - half, x, midpoint + half, fill=color)

    def _audio_player_progress_fraction(self) -> float:
        total_seconds = self._audio_player_total_seconds()
        if total_seconds <= 0:
            return 0.0
        current_seconds = self._audio_player_current_seconds()
        return max(0.0, min(1.0, current_seconds / total_seconds))

    def _audio_player_try_auto_next(self) -> bool:
        assert self.audio_player_auto_next_var is not None
        if not self.audio_player_auto_next_var.get():
            return False
        if self.audio_player_queue_index is None:
            return False
        next_index = int(self.audio_player_queue_index) + 1
        while next_index < len(self.history_state):
            target = Path(self.history_state[next_index])
            if target.is_file():
                if self.audio_player_status_var is not None:
                    self.audio_player_status_var.set(f"Queue: loading {target.name}...")
                self._select_history_index(next_index)
                self._load_audio_file_async(target, autoplay=True, history_index=next_index)
                return True
            next_index += 1
        return False

    def _find_history_index(self, path: Path) -> int | None:
        target = str(path)
        for index, value in enumerate(self.history_state):
            if str(value) == target:
                return index
        return None

    def _audio_player_total_seconds(self, *, refresh: bool = False) -> float:
        if self.vlc_audio is not None and (refresh or self.audio_player_media_length_ms <= 0):
            try:
                length_ms = int(self.vlc_audio.get_length_ms())
            except Exception:
                length_ms = -1
            if length_ms > 0:
                self.audio_player_media_length_ms = length_ms
        if self.audio_player_media_length_ms > 0:
            return float(self.audio_player_media_length_ms) / 1000.0
        if self.audio_player_total_frames > 0 and self.audio_player_sample_rate > 0:
            return float(self.audio_player_total_frames) / float(self.audio_player_sample_rate)
        return 0.0

    def _audio_player_current_seconds(self) -> float:
        if self.vlc_audio is not None:
            try:
                current_ms = int(self.vlc_audio.get_time_ms())
            except Exception:
                current_ms = -1
            if current_ms >= 0:
                return float(current_ms) / 1000.0
        if (
            self.audio_player_backend == "sounddevice"
            and self.audio_player_is_playing
            and self.audio_player_sample_rate > 0
            and self.audio_player_sd_started_at > 0.0
        ):
            elapsed = max(0.0, time.monotonic() - self.audio_player_sd_started_at)
            current_frame = self.audio_player_sd_start_frame + int(elapsed * float(self.audio_player_sample_rate))
            if self.audio_player_total_frames > 0:
                current_frame = min(self.audio_player_total_frames, current_frame)
            return float(current_frame) / float(self.audio_player_sample_rate)
        if self.audio_player_sample_rate > 0:
            return float(self.audio_player_current_frame) / float(self.audio_player_sample_rate)
        return 0.0

    @staticmethod
    def _audio_player_format_timestamp(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        return f"{minutes:02d}:{secs:02d}"

    def _bind_audio_player_shortcuts(self) -> None:
        if self.root is None or self.audio_player_shortcuts_bound:
            return
        self.root.bind_all("<space>", self._on_audio_shortcut_play_pause, add="+")
        self.root.bind_all(
            "<Control-Left>",
            lambda event: self._on_audio_shortcut_seek(event, -float(self.audio_player_seek_step_seconds)),
            add="+",
        )
        self.root.bind_all(
            "<Control-Right>",
            lambda event: self._on_audio_shortcut_seek(event, float(self.audio_player_seek_step_seconds)),
            add="+",
        )
        self.root.bind_all("<Control-Up>", lambda event: self._on_audio_shortcut_volume(event, 0.05), add="+")
        self.root.bind_all("<Control-Down>", lambda event: self._on_audio_shortcut_volume(event, -0.05), add="+")
        self.audio_player_shortcuts_bound = True

    def _on_audio_shortcut_play_pause(self, event: tk.Event[Any]) -> str | None:
        if self._is_text_input_widget(getattr(event, "widget", None)):
            return None
        if self.audio_player_is_playing:
            self._on_audio_player_pause()
            return "break"
        if self.audio_player_loaded_path is not None:
            self._on_audio_player_play()
            return "break"
        return None

    def _on_audio_shortcut_seek(self, event: tk.Event[Any], delta_seconds: float) -> str | None:
        if self._is_text_input_widget(getattr(event, "widget", None)):
            return None
        if self.audio_player_loaded_path is None:
            return None
        self._audio_player_seek_relative(float(delta_seconds))
        return "break"

    def _on_audio_shortcut_volume(self, event: tk.Event[Any], delta: float) -> str | None:
        if self._is_text_input_widget(getattr(event, "widget", None)):
            return None
        if self.audio_player_volume_var is None:
            return None
        current = self._coerce_float(self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5)
        updated = self._coerce_float(current + float(delta), default=1.0, min_value=0.0, max_value=1.5)
        self.audio_player_volume_var.set(updated)
        self._save_audio_player_state()
        return "break"

    @staticmethod
    def _is_text_input_widget(widget: Any) -> bool:
        if widget is None:
            return False
        try:
            class_name = str(widget.winfo_class()).lower()
        except Exception:
            return False
        return class_name in {
            "entry",
            "tentry",
            "text",
            "spinbox",
            "listbox",
            "ttk::combobox",
            "combobox",
        }

    def _restore_audio_player_from_saved_state(self) -> None:
        path = self.audio_player_restore_path
        if path is None:
            return
        if not path.is_file():
            return
        history_index = self._find_history_index(path)
        if history_index is not None:
            self._select_history_index(history_index)
        self._load_audio_file_async(
            path,
            autoplay=False,
            history_index=history_index,
            resume_seconds=self.audio_player_restore_position_seconds,
        )

    def _load_audio_player_state(self) -> dict[str, Any]:
        path = self.audio_player_state_path
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.logger.exception("Failed to read audio player state")
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _save_audio_player_state(self) -> None:
        if self.audio_player_volume_var is None or self.audio_player_auto_next_var is None:
            return
        last_position_seconds = max(0.0, float(self._audio_player_current_seconds()))
        state = {
            "volume": self._coerce_float(self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5),
            "auto_next": bool(self.audio_player_auto_next_var.get()),
            "last_path": str(self.audio_player_loaded_path) if self.audio_player_loaded_path is not None else "",
            "last_position_seconds": last_position_seconds,
            "queue_index": self.audio_player_queue_index,
        }
        path = self.audio_player_state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            self.logger.exception("Failed to save audio player state")

    @staticmethod
    def _coerce_float(
        value: Any,
        *,
        default: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = float(default)
        if min_value is not None:
            parsed = max(float(min_value), parsed)
        if max_value is not None:
            parsed = min(float(max_value), parsed)
        return float(parsed)

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(default)

    def _on_stream_start(self) -> None:
        if self.stream_thread is not None and self.stream_thread.is_alive():
            self.stream_status_var.set("Stream is already running.")
            return
        self.stream_stop_event.clear()
        self.stream_status_var.set("Streaming...")
        self.stream_btn.state(["disabled"])
        self.stop_stream_btn.state(["!disabled"])
        self._stop_audio(preserve_player_position=True)
        kwargs = self._base_generation_kwargs()

        def worker() -> None:
            if sd is None:
                self._run_on_ui(
                    lambda: self.stream_status_var.set(
                        "sounddevice is not installed. Stream playback is unavailable."
                    )
                )
                self._run_on_ui(self._finalize_stream_buttons)
                return
            chunks = 0
            try:
                iterator = self.generate_all(**kwargs)
                for sample_rate, chunk in iterator:
                    if self.stream_stop_event.is_set():
                        break
                    audio = np.asarray(chunk, dtype=np.float32).flatten()
                    if audio.size == 0:
                        continue
                    chunks += 1
                    sd.play(audio, samplerate=int(sample_rate), blocking=True)
            except Exception as exc:  # pragma: no cover - UI threading path
                self.logger.exception("Stream playback failed")
                message = f"Stream failed: {exc}"
                self._run_on_ui(
                    lambda message=message: self.stream_status_var.set(message)
                )
            else:
                if self.stream_stop_event.is_set():
                    self._run_on_ui(lambda: self.stream_status_var.set("Stream stopped."))
                else:
                    self._run_on_ui(
                        lambda: self.stream_status_var.set(f"Stream complete: {chunks} chunk(s).")
                    )
            finally:
                try:
                    sd.stop()
                except Exception:
                    self.logger.exception("Failed to stop stream device")
                self._run_on_ui(self._finalize_stream_buttons)

        self.stream_thread = threading.Thread(target=worker, daemon=True)
        self.stream_thread.start()

    def _on_stream_stop(self) -> None:
        self.stream_stop_event.set()
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                self.logger.exception("Failed to stop stream playback")
        self.stream_status_var.set("Stopping stream...")

    def _finalize_stream_buttons(self) -> None:
        self.stream_btn.state(["!disabled"])
        self.stop_stream_btn.state(["disabled"])

    def _on_pronunciation_load(self) -> None:
        if not callable(self.load_pronunciation_rules):
            self.pronunciation_status_var.set("Pronunciation dictionary is not configured.")
            return

        def work():
            return self.load_pronunciation_rules()

        def on_success(payload: tuple[str, str]) -> None:
            json_text, status = payload
            self._write_text(self.pronunciation_json_text, json_text)
            self.pronunciation_status_var.set(status)

        self._threaded(work, on_success)

    def _on_pronunciation_apply(self) -> None:
        if not callable(self.apply_pronunciation_rules):
            self.pronunciation_status_var.set("Pronunciation dictionary is not configured.")
            return
        raw = self._read_text(self.pronunciation_json_text)

        def work():
            return self.apply_pronunciation_rules(raw)

        def on_success(payload: tuple[str, str]) -> None:
            json_text, status = payload
            self._write_text(self.pronunciation_json_text, json_text)
            self.pronunciation_status_var.set(status)

        self._threaded(work, on_success)

    def _on_pronunciation_import(self) -> None:
        if not callable(self.import_pronunciation_rules):
            self.pronunciation_status_var.set("Pronunciation dictionary is not configured.")
            return
        path = filedialog.askopenfilename(
            title="Import pronunciation rules",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        def work():
            return self.import_pronunciation_rules(path)

        def on_success(payload: tuple[str, str]) -> None:
            json_text, status = payload
            self._write_text(self.pronunciation_json_text, json_text)
            self.pronunciation_status_var.set(status)

        self._threaded(work, on_success)

    def _on_pronunciation_export(self) -> None:
        if not callable(self.export_pronunciation_rules):
            self.pronunciation_status_var.set("Pronunciation dictionary is not configured.")
            return

        def work():
            return self.export_pronunciation_rules()

        def on_success(payload: tuple[str | None, str]) -> None:
            _path, status = payload
            self.pronunciation_status_var.set(status)

        self._threaded(work, on_success)

    def _on_export_morphology(self) -> None:
        if not callable(self.export_morphology_sheet):
            self.export_status_var.set("Morphology DB export is not configured.")
            return
        dataset = self.export_dataset_var.get()
        export_format = self.export_format_var.get()

        def work():
            if self.export_supports_format:
                return self.export_morphology_sheet(dataset, export_format)
            return self.export_morphology_sheet(dataset)

        def on_success(payload: tuple[str | None, str]) -> None:
            path, status = payload
            self.export_status_var.set(status)
            self.export_path_var.set(path or "")
            self.generate_status_var.set(status if not path else f"{status} {os.path.basename(path)}")
            self._on_morphology_preview_dataset_change()

        self._threaded(work, on_success)

    def _on_morphology_preview_dataset_change(self) -> None:
        if self.morph_preview_status_var is None:
            return
        dataset = str(self.export_dataset_var.get() if self.export_dataset_var is not None else "lexemes").strip().lower()
        if not callable(self.morphology_db_view):
            self._set_morphology_preview_table(["No data"], [["No data"]], rows_count=0, unique_count=0)
            return
        self.morph_preview_status_var.set("Rows: 0 | Unique: 0 | Last updated: loading...")

        def work() -> tuple[str, dict[str, Any], str]:
            if dataset == "pos_table":
                table_update, status = self.morphology_db_view("lexemes", 1000, 0)
                pos_update = self._build_pos_table_preview_from_lexemes(table_update)
                return dataset, pos_update, status
            table_update, status = self.morphology_db_view(dataset, 50, 0)
            return dataset, table_update, status

        def on_success(payload: tuple[str, dict[str, Any], str]) -> None:
            selected_dataset, table_update, _status = payload
            headers, rows = self._project_morphology_preview_rows(selected_dataset, table_update)
            if not rows:
                fallback_headers = headers if headers else ["No data"]
                self._set_morphology_preview_table(fallback_headers, [["No data"]], rows_count=0, unique_count=0)
                return
            unique_values: set[str] = set()
            for row in rows:
                for cell in row:
                    cell_text = str(cell or "").strip()
                    if cell_text:
                        unique_values.add(cell_text)
            self._set_morphology_preview_table(
                headers,
                rows,
                rows_count=len(rows),
                unique_count=len(unique_values),
            )

        self._threaded(work, on_success)

    def _project_morphology_preview_rows(
        self,
        dataset: str,
        table_update: dict[str, Any],
    ) -> tuple[list[str], list[list[str]]]:
        headers = [str(item) for item in extract_morph_headers(table_update)]
        raw_rows = table_update.get("value", [])
        if not isinstance(raw_rows, list):
            raw_rows = []
        rows: list[list[str]] = []
        for row in raw_rows:
            if isinstance(row, (list, tuple)):
                rows.append([str(item or "") for item in row])
            else:
                rows.append([str(row or "")])

        column_specs: dict[str, list[tuple[str, tuple[str, ...]]]] = {
            "lexemes": [
                ("token_text", ("token_text", "dedup_key", "lemma")),
                ("lemma", ("lemma",)),
                ("upos", ("upos",)),
            ],
            "occurrences": [
                ("token_text", ("token_text", "token")),
                ("lemma", ("lemma",)),
            ],
            "expressions": [
                ("expression_text", ("expression_text",)),
                ("type", ("expression_type", "type")),
            ],
            "reviews": [
                ("sentence", ("sentence", "token_text", "source")),
                ("local_tag", ("local_tag", "local_upos", "status")),
                ("lm_tag", ("lm_tag", "lm_upos", "is_match")),
            ],
        }
        normalized_dataset = str(dataset or "").strip().lower()
        specs = column_specs.get(normalized_dataset)
        if specs is None:
            preview_headers = headers[:]
            preview_rows = [row[: len(preview_headers)] for row in rows[:50]]
            return preview_headers, preview_rows

        header_index = {name.strip().lower(): idx for idx, name in enumerate(headers)}
        preview_headers = [display for display, _aliases in specs]
        preview_rows: list[list[str]] = []
        for row in rows[:50]:
            projected_row: list[str] = []
            for _display, aliases in specs:
                selected_value = ""
                for alias in aliases:
                    idx = header_index.get(alias.lower())
                    if idx is None or idx >= len(row):
                        continue
                    selected_value = str(row[idx] or "")
                    break
                projected_row.append(selected_value)
            preview_rows.append(projected_row)
        return preview_headers, preview_rows

    def _build_pos_table_preview_from_lexemes(self, table_update: dict[str, Any]) -> dict[str, Any]:
        headers = [str(item) for item in extract_morph_headers(table_update)]
        raw_rows = table_update.get("value", [])
        if not isinstance(raw_rows, list):
            raw_rows = []
        if not headers or not raw_rows:
            return {"headers": ["No data"], "value": []}

        header_index = {name.strip().lower(): idx for idx, name in enumerate(headers)}
        upos_idx = header_index.get("upos")
        lemma_idx = header_index.get("lemma")
        if upos_idx is None or lemma_idx is None:
            return {"headers": ["No data"], "value": []}

        upos_buckets: dict[str, list[str]] = {}
        upos_seen: dict[str, set[str]] = {}
        for row in raw_rows:
            if not isinstance(row, (list, tuple)):
                continue
            upos_text = str(row[upos_idx] if upos_idx < len(row) else "").strip().upper()
            lemma_text = str(row[lemma_idx] if lemma_idx < len(row) else "").strip()
            if not upos_text or not lemma_text:
                continue
            seen_values = upos_seen.setdefault(upos_text, set())
            if lemma_text in seen_values:
                continue
            seen_values.add(lemma_text)
            upos_buckets.setdefault(upos_text, []).append(lemma_text)

        if not upos_buckets:
            return {"headers": ["No data"], "value": []}

        column_pairs: list[tuple[str, str]] = [
            ("Noun", "NOUN"),
            ("Verb", "VERB"),
            ("Adjective", "ADJ"),
            ("Adverb", "ADV"),
            ("Pronoun", "PRON"),
            ("ProperNoun", "PROPN"),
            ("Number", "NUM"),
            ("Determiner", "DET"),
            ("Adposition", "ADP"),
            ("CConj", "CCONJ"),
            ("SConj", "SCONJ"),
            ("Particle", "PART"),
            ("Interjection", "INTJ"),
            ("Symbol", "SYM"),
            ("Other", "X"),
        ]
        known_upos = {upos for _label, upos in column_pairs}
        for upos in sorted([name for name in upos_buckets.keys() if name not in known_upos]):
            column_pairs.append((upos, upos))

        preview_headers = [label for label, _upos in column_pairs]
        max_rows = max((len(upos_buckets.get(upos, [])) for _label, upos in column_pairs), default=0)
        preview_rows: list[list[str]] = []
        for row_idx in range(min(max_rows, 50)):
            values: list[str] = []
            for _label, upos in column_pairs:
                items = upos_buckets.get(upos, [])
                values.append(items[row_idx] if row_idx < len(items) else "")
            preview_rows.append(values)
        return {"headers": preview_headers, "value": preview_rows}

    def _set_morphology_preview_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        rows_count: int,
        unique_count: int,
    ) -> None:
        if self.morph_preview_tree is None or self.morph_preview_status_var is None:
            return
        safe_headers = [str(item) for item in (headers or ["No data"])]
        safe_rows: list[list[str]] = []
        for row in rows:
            row_values = list(row) if isinstance(row, (list, tuple)) else [str(row)]
            if len(row_values) < len(safe_headers):
                row_values.extend([""] * (len(safe_headers) - len(row_values)))
            safe_rows.append([str(item or "") for item in row_values[: len(safe_headers)]])

        self.morph_preview_headers = safe_headers
        self.morph_preview_tree.delete(*self.morph_preview_tree.get_children())
        self.morph_preview_tree.configure(columns=self.morph_preview_headers)
        for header in self.morph_preview_headers:
            self.morph_preview_tree.heading(header, text=header)
            self.morph_preview_tree.column(header, width=140, stretch=True, anchor="w")
        for row in safe_rows:
            self.morph_preview_tree.insert("", tk.END, values=row)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.morph_preview_status_var.set(
            f"Rows: {int(rows_count)} | Unique: {int(unique_count)} | Last updated: {timestamp}"
        )

    def _on_lesson_generate(self) -> None:
        if not callable(self.build_lesson_for_tts):
            self.lesson_status_var.set("Lesson builder is not configured.")
            return
        raw_text = self._read_text(self.lesson_raw_text)
        extra = self._read_text(self.llm_extra_instructions_text)
        self.lesson_status_var.set("Generating lesson...")

        def work():
            return self.build_lesson_for_tts(
                raw_text,
                self.llm_base_url_var.get(),
                self.llm_api_key_var.get(),
                self.llm_model_var.get(),
                self.llm_temperature_var.get(),
                self.llm_max_tokens_var.get(),
                self.llm_timeout_seconds_var.get(),
                extra,
            )

        def on_success(payload: tuple[str, str]) -> None:
            lesson_text, status = payload
            self._write_text(self.lesson_output_text, lesson_text)
            self.lesson_status_var.set(status)

        self._threaded(work, on_success)

    def _on_lesson_copy_to_input(self) -> None:
        text = self._read_text(self.lesson_output_text)
        if not text:
            self.lesson_status_var.set("Lesson output is empty.")
            return
        self._write_text(self.input_text, text)
        self.lesson_status_var.set("Lesson copied to main TTS input.")

    def _on_morph_refresh(self) -> None:
        if not callable(self.morphology_db_view):
            self.morph_status_var.set("Morphology DB is not configured.")
            return
        dataset = self.morph_dataset_var.get()
        limit = int(self.morph_limit_var.get())
        offset = int(self.morph_offset_var.get())
        self.morph_status_var.set("Loading...")

        def work():
            return self.morphology_db_view(dataset, limit, offset)

        def on_success(payload: tuple[dict[str, Any], str]) -> None:
            table_update, status = payload
            self._apply_table_update(table_update)
            self.morph_status_var.set(status)
            self.selected_morph_row_var.set("")
            self.morph_delete_armed = ""

        self._threaded(work, on_success)

    def _apply_table_update(self, table_update: dict[str, Any]) -> None:
        if self.morph_tree is None:
            return
        headers = extract_morph_headers(table_update)
        rows = table_update.get("value", [])
        if not isinstance(rows, list):
            rows = []
        if not headers:
            headers = ["No data"]
            rows = [[]]
        self.morph_headers = [str(item) for item in headers]
        self.morph_tree.delete(*self.morph_tree.get_children())
        self.morph_tree.configure(columns=self.morph_headers)
        for header in self.morph_headers:
            self.morph_tree.heading(header, text=header)
            self.morph_tree.column(header, width=120, stretch=True, anchor="w")
        for row in rows:
            row_values = list(row) if isinstance(row, (list, tuple)) else [str(row)]
            if len(row_values) < len(self.morph_headers):
                row_values.extend([""] * (len(self.morph_headers) - len(row_values)))
            self.morph_tree.insert("", tk.END, values=row_values[: len(self.morph_headers)])

    def _on_morph_select_row(self, _event: tk.Event[Any]) -> None:
        if self.morph_tree is None:
            return
        selected_items = self.morph_tree.selection()
        if not selected_items:
            return
        item_id = selected_items[0]
        row_values = self.morph_tree.item(item_id, "values")
        try:
            selected_row_id, payload = build_morph_update_payload(
                self.morph_dataset_var.get(),
                self.morph_headers,
                row_values,
            )
        except ValueError as exc:
            self.morph_status_var.set(f"Selection failed: {exc}")
            return
        self.selected_morph_row_var.set(selected_row_id)
        self._write_text(self.morph_update_json_text, json.dumps(payload, ensure_ascii=False))
        self.morph_status_var.set(f"Selected row id/key={selected_row_id}.")
        self.morph_delete_armed = ""

    def _on_morph_add(self) -> None:
        if not callable(self.morphology_db_add):
            self.morph_status_var.set("Morphology DB is not configured.")
            return
        dataset = self.morph_dataset_var.get()
        row_json = self._read_text(self.morph_add_json_text)
        limit = int(self.morph_limit_var.get())
        offset = int(self.morph_offset_var.get())
        self.morph_status_var.set("Adding row...")

        def work():
            return self.morphology_db_add(dataset, row_json, limit, offset)

        def on_success(payload: tuple[dict[str, Any], str]) -> None:
            table_update, status = payload
            self._apply_table_update(table_update)
            self.morph_status_var.set(status)
            self.selected_morph_row_var.set("")
            self.morph_delete_armed = ""

        self._threaded(work, on_success)

    def _on_morph_update(self) -> None:
        if not callable(self.morphology_db_update):
            self.morph_status_var.set("Morphology DB is not configured.")
            return
        dataset = self.morph_dataset_var.get()
        row_id = self.selected_morph_row_var.get()
        row_json = self._read_text(self.morph_update_json_text)
        limit = int(self.morph_limit_var.get())
        offset = int(self.morph_offset_var.get())
        self.morph_status_var.set("Updating row...")

        def work():
            return self.morphology_db_update(dataset, row_id, row_json, limit, offset)

        def on_success(payload: tuple[dict[str, Any], str]) -> None:
            table_update, status = payload
            self._apply_table_update(table_update)
            self.morph_status_var.set(status)
            self.selected_morph_row_var.set("")
            self.morph_delete_armed = ""

        self._threaded(work, on_success)

    def _on_morph_delete(self) -> None:
        if not callable(self.morphology_db_delete):
            self.morph_status_var.set("Morphology DB is not configured.")
            return
        selected = self.selected_morph_row_var.get()
        should_delete, next_armed, message = resolve_morph_delete_confirmation(
            selected,
            self.morph_delete_armed,
        )
        self.morph_delete_armed = next_armed
        if not should_delete:
            self.morph_status_var.set(message)
            return
        dataset = self.morph_dataset_var.get()
        limit = int(self.morph_limit_var.get())
        offset = int(self.morph_offset_var.get())
        self.morph_status_var.set("Deleting row...")

        def work():
            return self.morphology_db_delete(dataset, selected, limit, offset)

        def on_success(payload: tuple[dict[str, Any], str]) -> None:
            table_update, status = payload
            self._apply_table_update(table_update)
            self.morph_status_var.set(status)
            self.selected_morph_row_var.set("")
            self.morph_delete_armed = ""

        self._threaded(work, on_success)


def create_tkinter_app(
    *,
    config: AppConfig,
    cuda_available: bool,
    logger,
    generate_first,
    tokenize_first,
    generate_all,
    predict,
    export_morphology_sheet=None,
    morphology_db_view=None,
    morphology_db_add=None,
    morphology_db_update=None,
    morphology_db_delete=None,
    load_pronunciation_rules=None,
    apply_pronunciation_rules=None,
    import_pronunciation_rules=None,
    export_pronunciation_rules=None,
    build_lesson_for_tts=None,
    set_tts_only_mode=None,
    set_llm_only_mode=None,
    tts_only_mode_default: bool = False,
    llm_only_mode_default: bool = False,
    history_service=None,
    choices: Mapping[str, str] | None = None,
) -> DesktopApp:
    """Create the Tkinter desktop app instance."""
    return TkinterDesktopApp(
        config=config,
        cuda_available=cuda_available,
        logger=logger,
        generate_first=generate_first,
        tokenize_first=tokenize_first,
        generate_all=generate_all,
        predict=predict,
        export_morphology_sheet=export_morphology_sheet,
        morphology_db_view=morphology_db_view,
        morphology_db_add=morphology_db_add,
        morphology_db_update=morphology_db_update,
        morphology_db_delete=morphology_db_delete,
        load_pronunciation_rules=load_pronunciation_rules,
        apply_pronunciation_rules=apply_pronunciation_rules,
        import_pronunciation_rules=import_pronunciation_rules,
        export_pronunciation_rules=export_pronunciation_rules,
        build_lesson_for_tts=build_lesson_for_tts,
        set_tts_only_mode=set_tts_only_mode,
        set_llm_only_mode=set_llm_only_mode,
        tts_only_mode_default=tts_only_mode_default,
        llm_only_mode_default=llm_only_mode_default,
        history_service=history_service,
        choices=choices,
    )
