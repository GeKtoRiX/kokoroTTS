"""Tkinter desktop UI for Kokoro TTS."""
from __future__ import annotations

import os
import threading
import time
import tkinter as tk
import sys
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable, Mapping

from ..storage.morphology_projection import (
    build_pos_table_preview_from_lexemes,
    count_unique_non_empty_cells,
    format_morphology_preview_table,
    project_morphology_preview_rows,
)
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
    RUNTIME_MODE_TTS_MORPH,
    extract_morph_headers,
    normalize_runtime_mode,
    runtime_mode_from_flags,
    runtime_mode_status_text,
    runtime_mode_tab_visibility,
    supports_export_format_arg,
    tts_only_mode_status_text,
)
from .desktop_types import DesktopApp
from .features.audio_backend import VlcAudioBackend as _FeatureVlcAudioBackend
from .features.audio_player_feature import (
    AudioPlayerFeature,
    configure_runtime_modules as configure_audio_player_runtime_modules,
)
from .features.audio_player_runtime_state import AudioPlayerRuntimeState
from .features.generate_tab_feature import GenerateTabFeature
from .features.morphology_tab_feature import MorphologyTabFeature
from .features.stream_tab_feature import StreamTabFeature

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
    """Compatibility wrapper around the feature-module VLC backend."""
    def __init__(self) -> None:
        if vlc is None:
            raise RuntimeError("python-vlc is not available")
        self._backend = _FeatureVlcAudioBackend(vlc_module=vlc, platform_name=sys.platform)
        self.instance = self._backend.instance
        self.player = self._backend.player
        self.media = self._backend.media

    def load(self, path: str) -> None:
        self._backend.load(path)
        self.media = self._backend.media

    def play(self) -> None:
        self._backend.play()

    def pause_toggle(self) -> None:
        self._backend.pause_toggle()

    def set_pause(self, on: bool) -> None:
        self._backend.set_pause(on)

    def stop(self) -> None:
        self._backend.stop()

    def is_playing(self) -> bool:
        return self._backend.is_playing()

    def set_volume(self, vol_0_100: int) -> None:
        self._backend.set_volume(vol_0_100)

    def get_time_ms(self) -> int:
        return self._backend.get_time_ms()

    def get_length_ms(self) -> int:
        return self._backend.get_length_ms()

    def set_time_ms(self, ms: int) -> None:
        self._backend.set_time_ms(ms)

    def get_state(self):
        return self._backend.get_state()

    def release(self) -> None:
        self._backend.release()
        self.media = self._backend.media


class TkinterDesktopApp(DesktopApp):
    """Tkinter implementation of the Kokoro desktop UI."""

    _AUDIO_STATE_FIELD_MAP: dict[str, str] = {
        "audio_player_loaded_path": "loaded_path",
        "audio_player_pcm_data": "pcm_data",
        "audio_player_sample_rate": "sample_rate",
        "audio_player_total_frames": "total_frames",
        "audio_player_current_frame": "current_frame",
        "audio_player_media_length_ms": "media_length_ms",
        "audio_player_backend": "backend",
        "audio_player_sd_start_frame": "sd_start_frame",
        "audio_player_sd_started_at": "sd_started_at",
        "audio_player_is_playing": "is_playing",
        "audio_player_is_paused": "is_paused",
        "audio_player_tick_job": "tick_job",
        "audio_player_seek_dragging": "seek_dragging",
        "audio_player_seek_programmatic": "seek_programmatic",
        "audio_player_queue_index": "queue_index",
        "audio_player_waveform": "waveform",
        "audio_player_state_path": "state_path",
        "audio_player_restore_path": "restore_path",
        "audio_player_restore_position_seconds": "restore_position_seconds",
        "audio_player_shortcuts_bound": "shortcuts_bound",
        "audio_player_seek_step_seconds": "seek_step_seconds",
    }

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
        load_pronunciation_rules=None,
        apply_pronunciation_rules=None,
        import_pronunciation_rules=None,
        export_pronunciation_rules=None,
        set_tts_only_mode=None,
        tts_only_mode_default: bool = False,
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
        self.load_pronunciation_rules = load_pronunciation_rules
        self.apply_pronunciation_rules = apply_pronunciation_rules
        self.import_pronunciation_rules = import_pronunciation_rules
        self.export_pronunciation_rules = export_pronunciation_rules
        self.set_tts_only_mode = set_tts_only_mode
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
        if not self.runtime_tts_only_enabled:
            self.runtime_mode_value = RUNTIME_MODE_DEFAULT
            self.runtime_tts_only_enabled = True
        else:
            self.runtime_mode_value = runtime_mode_from_flags(
                tts_only_enabled=self.runtime_tts_only_enabled,
            )

        self.root: tk.Tk | None = None
        self.notebook: ttk.Notebook | None = None
        self.tabs: dict[str, tuple[ttk.Frame, str]] = {}
        self.generate_detail_notebook: ttk.Notebook | None = None
        self.generate_detail_tabs: dict[str, ttk.Frame] = {}
        self.history_state: list[str] = []
        self.morph_headers: list[str] = []
        self.stream_stop_event = threading.Event()
        self.stream_thread: threading.Thread | None = None
        self.accordion_setters: dict[str, Callable[[bool], None]] = {}
        self.audio_player_state = AudioPlayerRuntimeState(
            state_path=Path(self.config.output_dir) / ".audio_player_state.json",
        )
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
        self.morph_status_var: tk.StringVar | None = None
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
        self.morph_dataset_var: tk.StringVar | None = None
        self.morph_limit_var: tk.IntVar | None = None
        self.morph_offset_var: tk.IntVar | None = None
        self.morph_tree: ttk.Treeview | None = None
        self.morph_preview_tree: ttk.Treeview | None = None
        self.morph_preview_headers: list[str] = []
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

        self._generate_tab_feature = GenerateTabFeature(self)
        self._stream_tab_feature = StreamTabFeature(self)
        self._morphology_tab_feature = MorphologyTabFeature(self)
        self._audio_player_feature = AudioPlayerFeature(self)
        self._sync_audio_player_feature_runtime()

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
        button_disabled_bg = "#081308"
        tab_inactive = "#091809"
        tab_active = "#112a11"
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
        self.morph_status_var = tk.StringVar(master=self.root, value="")
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
            selectforeground=surface,
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
            selectforeground=surface,
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
        morph_tab = ttk.Frame(self.notebook)
        self.tabs = {
            "generate": (generate_tab, "Generate"),
            "stream": (stream_tab, "Stream"),
            "morph": (morph_tab, "Morphology DB"),
        }
        self.notebook.add(generate_tab, text="Generate")
        self.notebook.add(stream_tab, text="Stream")
        self.notebook.add(morph_tab, text="Morphology DB")

        self._build_generate_tab(generate_tab)
        self._build_stream_tab(stream_tab)
        self._build_morph_tab(morph_tab)
    def _build_generate_tab(self, parent: ttk.Frame) -> None:
        self._generate_tab_feature.build_tab(parent)

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
        self._stream_tab_feature.build_tab(parent)
    def _build_morph_tab(self, parent: ttk.Frame) -> None:
        self._morphology_tab_feature.build_tab(parent)

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
            else:
                self._set_tts_only_mode_wrapped(False)
        self.runtime_mode_status_var.set(runtime_mode_status_text(selected_mode))
        self._sync_runtime_tabs(selected_mode)
        self._sync_hardware_selector_visibility()

    def _sync_hardware_selector_visibility(self) -> None:
        if self.hardware_section_frame is None or self.hardware_var is None:
            return
        if not self.cuda_available:
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

    def _sync_runtime_tabs(self, selected_mode: str) -> None:
        if self.notebook is None:
            return
        morph_visible = runtime_mode_tab_visibility(selected_mode)
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
        return self._generate_tab_feature.base_generation_kwargs()
    def _on_generate(self) -> None:
        self._generate_tab_feature.on_generate()
    def _on_tokenize(self) -> None:
        self._generate_tab_feature.on_tokenize()
    def _sync_audio_player_feature_runtime(self) -> None:
        configure_audio_player_runtime_modules(
            sd_module=sd,
            vlc_module=vlc,
            sf_module=sf,
            filedialog_module=filedialog,
        )

    def __setattr__(self, name: str, value) -> None:
        state_field = type(self)._AUDIO_STATE_FIELD_MAP.get(name)
        if state_field is not None and "audio_player_state" in self.__dict__:
            setattr(self.audio_player_state, state_field, value)
            return
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        state_field = type(self)._AUDIO_STATE_FIELD_MAP.get(name)
        if state_field is not None and "audio_player_state" in self.__dict__:
            return getattr(self.audio_player_state, state_field)
        if name.startswith("_") and not name.startswith("__"):
            feature_attr = getattr(AudioPlayerFeature, name, None)
            if callable(feature_attr):
                def _delegated(*args, _name=name, **kwargs):
                    self._sync_audio_player_feature_runtime()
                    feature_method = getattr(self._audio_player_feature, _name)
                    return feature_method(*args, **kwargs)

                setattr(_delegated, "_audio_player_delegate_wrapper", True)
                return _delegated
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def _on_stream_start(self) -> None:
        self._stream_tab_feature.on_stream_start(sd_module=sd)
    def _on_stream_stop(self) -> None:
        self._stream_tab_feature.on_stream_stop(sd_module=sd)
    def _finalize_stream_buttons(self) -> None:
        self._stream_tab_feature.finalize_stream_buttons()

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
        self._morphology_tab_feature.on_export_morphology()
    def _on_morphology_preview_dataset_change(self) -> None:
        self._morphology_tab_feature.on_morphology_preview_dataset_change()
    def _project_morphology_preview_rows(
        self,
        dataset: str,
        table_update: dict[str, Any],
    ) -> tuple[list[str], list[list[str]]]:
        return self._morphology_tab_feature.project_morphology_preview_rows(dataset, table_update)

    def _project_morphology_preview_rows_impl(
        self,
        dataset: str,
        table_update: dict[str, Any],
    ) -> tuple[list[str], list[list[str]]]:
        return project_morphology_preview_rows(dataset, table_update, max_rows=50)
    def _build_pos_table_preview_from_lexemes(self, table_update: dict[str, Any]) -> dict[str, Any]:
        return self._morphology_tab_feature.build_pos_table_preview_from_lexemes(table_update)

    def _build_pos_table_preview_from_lexemes_impl(self, table_update: dict[str, Any]) -> dict[str, Any]:
        return build_pos_table_preview_from_lexemes(table_update, max_rows=50)
    def _set_morphology_preview_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        rows_count: int,
        unique_count: int,
    ) -> None:
        self._morphology_tab_feature.set_morphology_preview_table(
            headers,
            rows,
            rows_count=rows_count,
            unique_count=unique_count,
        )

    def _format_morphology_preview_table(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[list[str], list[list[str]]]:
        return format_morphology_preview_table(headers, rows)

    def _count_unique_non_empty_cells(self, rows: list[list[str]]) -> int:
        return count_unique_non_empty_cells(rows)
    def _on_morph_refresh(self) -> None:
        self._morphology_tab_feature.on_morph_refresh()
    def _apply_table_update(self, table_update: dict[str, Any]) -> None:
        self._morphology_tab_feature.apply_table_update(table_update)

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
    load_pronunciation_rules=None,
    apply_pronunciation_rules=None,
    import_pronunciation_rules=None,
    export_pronunciation_rules=None,
    set_tts_only_mode=None,
    tts_only_mode_default: bool = False,
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
        load_pronunciation_rules=load_pronunciation_rules,
        apply_pronunciation_rules=apply_pronunciation_rules,
        import_pronunciation_rules=import_pronunciation_rules,
        export_pronunciation_rules=export_pronunciation_rules,
        set_tts_only_mode=set_tts_only_mode,
        tts_only_mode_default=tts_only_mode_default,
        history_service=history_service,
        choices=choices,
    )
