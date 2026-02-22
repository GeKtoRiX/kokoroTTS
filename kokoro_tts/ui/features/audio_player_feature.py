"""Audio player feature extracted from TkinterDesktopApp."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog
from typing import Any

import numpy as np

from .audio_backend import VlcAudioBackend
from .audio_player_state import (
    coerce_bool as _coerce_bool_value,
    coerce_float as _coerce_float_value,
    load_audio_player_state as _load_audio_player_state_payload,
    save_audio_player_state as _save_audio_player_state_payload,
)

sd = None
vlc = None
sf = None
_DOUBLE_SPACE_STOP_WINDOW_SECONDS = 0.35


def configure_runtime_modules(*, sd_module, vlc_module, sf_module, filedialog_module) -> None:
    global sd
    global vlc
    global sf
    global filedialog
    sd = sd_module
    vlc = vlc_module
    sf = sf_module
    filedialog = filedialog_module


class AudioPlayerFeature:
    """Audio player and history playback behavior."""

    _INTERNAL_ATTRS = {"_host", "_INTERNAL_ATTRS"}

    def __init__(self, host) -> None:
        object.__setattr__(self, "_host", host)

    def __getattribute__(self, name: str):
        if name in object.__getattribute__(self, "_INTERNAL_ATTRS"):
            return object.__getattribute__(self, name)
        if name.startswith("_") and not name.startswith("__"):
            host = object.__getattribute__(self, "_host")
            host_dict = object.__getattribute__(host, "__dict__")
            if name in host_dict:
                override = host_dict[name]
                if not getattr(override, "_audio_player_delegate_wrapper", False):
                    return override
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_host"), name)

    def __setattr__(self, name: str, value):
        if name in object.__getattribute__(self, "_INTERNAL_ATTRS"):
            object.__setattr__(self, name, value)
            return
        setattr(object.__getattribute__(self, "_host"), name, value)

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
            self.audio_player_current_frame = int(
                current_seconds * float(max(self.audio_player_sample_rate, 1))
            )
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
        if self.history_listbox is None:
            return
        selected = self.history_listbox.curselection()
        if len(selected) != 1:
            return
        selected_item = self._selected_history_item(show_errors=False)
        if selected_item is None:
            return
        index, target = selected_item
        self._load_audio_file_async(target, autoplay=True, history_index=index)

    def _on_history_delete_selected(self) -> None:
        if self.history_listbox is None:
            return
        selected_indices = sorted(
            {
                int(index)
                for index in self.history_listbox.curselection()
                if 0 <= int(index) < len(self.history_state)
            }
        )
        if not selected_indices:
            self.generate_status_var.set("Select one or more history items first.")
            return
        if self.history_service is None:
            selected_set = set(selected_indices)
            self.history_state = [
                value for index, value in enumerate(self.history_state) if index not in selected_set
            ]
            self.audio_player_queue_index = None
            self._render_history()
            self.generate_status_var.set(
                f"Deleted {len(selected_indices)} selected history item(s)."
            )
            self._save_audio_player_state()
            return

        removed_count = len(selected_indices)

        def work():
            if callable(getattr(self.history_service, "remove_selected_history", None)):
                return self.history_service.remove_selected_history(
                    self.history_state, selected_indices
                )
            selected_set = set(selected_indices)
            return [
                value for index, value in enumerate(self.history_state) if index not in selected_set
            ]

        def on_success(updated: list[str]) -> None:
            self.history_state = list(updated or [])
            if self.audio_player_loaded_path is None:
                self.audio_player_queue_index = None
            else:
                self.audio_player_queue_index = self._find_history_index(
                    self.audio_player_loaded_path
                )
            self._render_history()
            self.generate_status_var.set(f"Deleted {removed_count} selected history item(s).")
            self._save_audio_player_state()

        self._threaded(work, on_success)

    def _on_history_context_menu(self, event: tk.Event[Any]) -> str:
        if self.history_listbox is None:
            return "break"
        try:
            index = int(self.history_listbox.nearest(event.y))
        except Exception:
            index = -1
        if index < 0 or index >= len(self.history_state):
            return "break"
        try:
            already_selected = bool(self.history_listbox.selection_includes(index))
        except Exception:
            already_selected = False
        if not already_selected:
            self._select_history_index(index)
        self.history_context_index = index
        if self.root is None:
            return "break"
        if self.history_context_menu is None:
            create_styled_menu = getattr(self, "_create_styled_menu", None)
            style_popup_menu = getattr(self, "_style_popup_menu", None)
            if callable(create_styled_menu):
                try:
                    self.history_context_menu = create_styled_menu(self.root)
                except Exception:
                    self.history_context_menu = tk.Menu(self.root, tearoff=0)
            else:
                self.history_context_menu = tk.Menu(self.root, tearoff=0)
                if callable(style_popup_menu):
                    try:
                        style_popup_menu(self.history_context_menu)
                    except Exception:
                        pass
            self.history_context_menu.add_command(
                label="Delete selected",
                command=self._on_history_delete_selected,
            )
            self.history_context_menu.add_separator()
            self.history_context_menu.add_command(
                label="Open Containing Folder",
                command=self._on_history_open_selected_folder,
            )
        try:
            self.history_context_menu.tk_popup(int(event.x_root), int(event.y_root))
        finally:
            try:
                self.history_context_menu.grab_release()
            except Exception:
                pass
        return "break"

    def _on_history_open_selected_folder(self) -> None:
        index = self.history_context_index
        if index is None:
            selected_item = self._selected_history_item(show_errors=True)
            if selected_item is None:
                return
            index, _target = selected_item
        if index < 0 or index >= len(self.history_state):
            self.generate_status_var.set("Selected history item is out of range.")
            return
        target = Path(self.history_state[index])
        if not target.exists():
            self.generate_status_var.set("History file does not exist.")
            return
        folder = target.parent
        if not folder.exists():
            self.generate_status_var.set("History folder does not exist.")
            return
        try:
            self._open_path_in_file_manager(folder)
            self.generate_status_var.set(f"Opened folder: {folder}")
        except Exception:
            self.logger.exception("Failed to open history folder: %s", folder)
            self.generate_status_var.set(f"Failed to open folder: {folder}")

    @staticmethod
    def _open_path_in_file_manager(path: Path) -> None:
        folder_path = Path(path).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        folder = str(folder_path)
        if sys.platform.startswith("win"):
            if hasattr(os, "startfile"):
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                explorer = shutil.which("explorer.exe") or shutil.which("explorer")
                if not explorer:
                    raise FileNotFoundError("Windows Explorer executable was not found.")
                subprocess.Popen([explorer, folder])
            return
        if sys.platform == "darwin":
            open_cmd = shutil.which("open")
            if not open_cmd:
                raise FileNotFoundError("macOS open executable was not found.")
            subprocess.Popen([open_cmd, folder])
            return
        xdg_open_cmd = shutil.which("xdg-open")
        if not xdg_open_cmd:
            raise FileNotFoundError("xdg-open executable was not found.")
        subprocess.Popen([xdg_open_cmd, folder])

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
                lambda waveform_audio=waveform_audio,
                sample_rate=sample_rate,
                total_frames=total_frames,
                waveform_warning=waveform_warning: self._on_audio_file_loaded(
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
        self.audio_player_queue_index = (
            history_index if history_index is not None else self._find_history_index(path)
        )
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
        self.audio_player_current_frame = int(
            restore_seconds * float(max(self.audio_player_sample_rate, 1))
        )
        if self.audio_player_total_frames > 0:
            self.audio_player_current_frame = max(
                0, min(self.audio_player_total_frames, self.audio_player_current_frame)
            )
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
            self.audio_player_status_var.set(
                "python-vlc is not installed. Install dependencies to enable Audio player."
            )
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
            volume = self._coerce_float(
                self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5
            )
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
        total_stamp = self._audio_player_format_timestamp(
            self._audio_player_total_seconds(refresh=True)
        )
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
        volume = self._coerce_float(
            self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5
        )
        try:
            assert self.vlc_audio is not None
            self.vlc_audio.set_volume(int(round(volume * 100.0)))
            self.vlc_audio.play()
            if target_ms > 0:
                # VLC may ignore immediate seek until playback thread is ready.
                self.vlc_audio.set_time_ms(target_ms)
                if self.root is not None:
                    self.root.after(
                        140, lambda target_ms=target_ms: self._audio_player_seek_vlc_ms(target_ms)
                    )
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
        return (
            sd is not None
            and self.audio_player_pcm_data is not None
            and self.audio_player_sample_rate > 0
        )

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
        volume = self._coerce_float(
            self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5
        )
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
        self.audio_player_current_frame = int(
            self._audio_player_current_seconds() * float(max(self.audio_player_sample_rate, 1))
        )
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
            self.audio_player_total_frames = int(
                total_seconds * float(self.audio_player_sample_rate)
            )
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
            self.audio_player_current_frame = int(
                ratio * float(max(1, self.audio_player_total_frames))
            )
        else:
            self.audio_player_current_frame = int(
                current_seconds * float(max(self.audio_player_sample_rate, 1))
            )
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
                    self.audio_player_status_var.set(
                        f"Playback complete: {self.audio_player_loaded_path.name}"
                    )
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
            max(0.0, min(1.0, current_seconds / max(total_seconds, 0.001)))
            * float(max(1, self.audio_player_total_frames))
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
            self.audio_player_play_btn.state(
                ["!disabled"] if loaded and not self.audio_player_is_playing else ["disabled"]
            )
        if self.audio_player_pause_btn is not None:
            self.audio_player_pause_btn.state(
                ["!disabled"] if self.audio_player_is_playing else ["disabled"]
            )
        if self.audio_player_stop_btn is not None:
            can_stop = loaded and (
                self.audio_player_is_playing
                or self.audio_player_is_paused
                or self.audio_player_current_frame > 0
            )
            self.audio_player_stop_btn.state(["!disabled"] if can_stop else ["disabled"])

    def _on_audio_player_minimal_toggle(self) -> None:
        self._apply_audio_player_minimal_mode()

    def _apply_audio_player_minimal_mode(self) -> None:
        minimal = (
            bool(self.audio_player_minimal_var.get())
            if self.audio_player_minimal_var is not None
            else False
        )
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
        if (
            self.audio_player_waveform_frame is not None
            and self.audio_player_waveform_frame.winfo_exists()
        ):
            self.audio_player_waveform_frame.grid_configure(pady=(6, 4) if minimal else (8, 6))
        if self.audio_player_seek_frame is not None and self.audio_player_seek_frame.winfo_exists():
            self.audio_player_seek_frame.grid_configure(pady=(0, 6) if minimal else (6, 6))

    def _sync_audio_player_control_labels(self) -> None:
        assert self.audio_player_volume_var is not None
        volume = self._coerce_float(
            self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5
        )
        if self.audio_player_volume_value_label is not None:
            percent = int(round(volume * 100.0))
            self.audio_player_volume_value_label.configure(text=f"{percent:>3d}%")

    def _on_audio_player_volume_scale(self) -> None:
        self._sync_audio_player_control_labels()
        if self.vlc_audio is not None and self.audio_player_volume_var is not None:
            try:
                volume = self._coerce_float(
                    self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5
                )
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
            current_frame = self.audio_player_sd_start_frame + int(
                elapsed * float(self.audio_player_sample_rate)
            )
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
            lambda event: self._on_audio_shortcut_seek(
                event, -float(self.audio_player_seek_step_seconds)
            ),
            add="+",
        )
        self.root.bind_all(
            "<Control-Right>",
            lambda event: self._on_audio_shortcut_seek(
                event, float(self.audio_player_seek_step_seconds)
            ),
            add="+",
        )
        self.root.bind_all(
            "<Control-Up>", lambda event: self._on_audio_shortcut_volume(event, 0.05), add="+"
        )
        self.root.bind_all(
            "<Control-Down>", lambda event: self._on_audio_shortcut_volume(event, -0.05), add="+"
        )
        for button in (
            self.audio_player_play_btn,
            self.audio_player_pause_btn,
            self.audio_player_stop_btn,
            self.audio_player_minimal_check_btn,
            self.audio_player_auto_next_check_btn,
        ):
            if button is None:
                continue
            button.bind(
                "<space>", self._on_audio_shortcut_play_pause_from_transport_button, add="+"
            )
            button.bind("<KeyRelease-space>", self._on_audio_shortcut_consume_key_release, add="+")
        self.audio_player_shortcuts_bound = True

    def _on_audio_shortcut_play_pause_from_transport_button(self, event: tk.Event[Any]) -> str:
        result = self._on_audio_shortcut_play_pause(event)
        if result is None:
            return "break"
        return result

    @staticmethod
    def _on_audio_shortcut_consume_key_release(_event: tk.Event[Any]) -> str:
        return "break"

    def _on_audio_shortcut_play_pause(self, event: tk.Event[Any]) -> str | None:
        if self._is_text_input_widget(getattr(event, "widget", None)):
            return None
        if self.audio_player_loaded_path is None:
            self.audio_player_last_space_pressed_at = 0.0
            return None

        now = float(time.monotonic())
        stop_window = self._coerce_float(
            getattr(
                self,
                "audio_player_double_space_stop_window_seconds",
                _DOUBLE_SPACE_STOP_WINDOW_SECONDS,
            ),
            default=_DOUBLE_SPACE_STOP_WINDOW_SECONDS,
            min_value=0.15,
            max_value=1.5,
        )
        last_pressed = self._coerce_float(
            getattr(self, "audio_player_last_space_pressed_at", 0.0),
            default=0.0,
            min_value=0.0,
        )
        self.audio_player_last_space_pressed_at = now
        elapsed = now - last_pressed
        can_stop = (
            self.audio_player_is_playing
            or self.audio_player_is_paused
            or self.audio_player_current_frame > 0
        )
        if can_stop and last_pressed > 0.0 and 0.0 < elapsed <= stop_window:
            self.audio_player_last_space_pressed_at = 0.0
            self._on_audio_player_stop()
            return "break"

        if self.audio_player_is_playing:
            self._on_audio_player_pause()
            return "break"
        self._on_audio_player_play()
        return "break"

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
        current = self._coerce_float(
            self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5
        )
        updated = self._coerce_float(
            current + float(delta), default=1.0, min_value=0.0, max_value=1.5
        )
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
            "ttk::entry",
            "text",
            "spinbox",
            "ttk::spinbox",
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
        return _load_audio_player_state_payload(self.audio_player_state_path, self.logger)

    def _save_audio_player_state(self) -> None:
        if self.audio_player_volume_var is None or self.audio_player_auto_next_var is None:
            return
        last_position_seconds = max(0.0, float(self._audio_player_current_seconds()))
        state = {
            "volume": self._coerce_float(
                self.audio_player_volume_var.get(), default=1.0, min_value=0.0, max_value=1.5
            ),
            "auto_next": bool(self.audio_player_auto_next_var.get()),
            "last_path": str(self.audio_player_loaded_path)
            if self.audio_player_loaded_path is not None
            else "",
            "last_position_seconds": last_position_seconds,
            "queue_index": self.audio_player_queue_index,
        }
        _save_audio_player_state_payload(self.audio_player_state_path, state, self.logger)

    @staticmethod
    def _coerce_float(
        value: Any,
        *,
        default: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        return _coerce_float_value(
            value,
            default=default,
            min_value=min_value,
            max_value=max_value,
        )

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        return _coerce_bool_value(value, default=default)
