"""Stream tab feature wiring and actions."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

import numpy as np


class StreamTabFeature:
    """Build and handle Stream tab interactions."""

    def __init__(self, host) -> None:
        self.host = host

    def build_tab(self, parent: ttk.Frame) -> None:
        ui = self.host
        assert ui.stream_status_var is not None
        frame = ttk.Frame(parent, padding=8)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, textvariable=ui.stream_status_var, wraplength=760).pack(anchor="w")
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", pady=(8, 8))
        ui.stream_btn = ttk.Button(
            btn_row, text="Stream", style="Primary.TButton", command=ui._on_stream_start
        )
        ui.stop_stream_btn = ttk.Button(btn_row, text="Stop", command=ui._on_stream_stop)
        ui.stream_btn.pack(side="left")
        ui.stop_stream_btn.pack(side="left", padx=(8, 0))
        ui.stop_stream_btn.state(["disabled"])
        ttk.Label(
            frame,
            text="Desktop stream runs in worker threads with Stop support.",
            wraplength=760,
            justify="left",
        ).pack(anchor="w")

    def on_stream_start(self, *, sd_module) -> None:
        ui = self.host
        if ui.stream_thread is not None and ui.stream_thread.is_alive():
            ui.stream_status_var.set("Stream is already running.")
            return
        ui.stream_stop_event.clear()
        ui.stream_status_var.set("Streaming...")
        ui.stream_btn.state(["disabled"])
        ui.stop_stream_btn.state(["!disabled"])
        ui._stop_audio(preserve_player_position=True)
        kwargs = ui._base_generation_kwargs()

        def worker() -> None:
            if sd_module is None:
                ui._run_on_ui(
                    lambda: ui.stream_status_var.set(
                        "sounddevice is not installed. Stream playback is unavailable."
                    )
                )
                ui._run_on_ui(ui._finalize_stream_buttons)
                return
            chunks = 0
            try:
                iterator = ui.generate_all(**kwargs)
                for sample_rate, chunk in iterator:
                    if ui.stream_stop_event.is_set():
                        break
                    audio = np.asarray(chunk, dtype=np.float32).flatten()
                    if audio.size == 0:
                        continue
                    chunks += 1
                    sd_module.play(audio, samplerate=int(sample_rate), blocking=True)
            except Exception as exc:  # pragma: no cover - UI threading path
                ui.logger.exception("Stream playback failed")
                message = f"Stream failed: {exc}"
                ui._run_on_ui(lambda message=message: ui.stream_status_var.set(message))
            else:
                if ui.stream_stop_event.is_set():
                    ui._run_on_ui(lambda: ui.stream_status_var.set("Stream stopped."))
                else:
                    ui._run_on_ui(
                        lambda: ui.stream_status_var.set(f"Stream complete: {chunks} chunk(s).")
                    )
            finally:
                try:
                    sd_module.stop()
                except Exception:
                    ui.logger.exception("Failed to stop stream device")
                ui._run_on_ui(ui._finalize_stream_buttons)

        ui.stream_thread = threading.Thread(target=worker, daemon=True)
        ui.stream_thread.start()

    def on_stream_stop(self, *, sd_module) -> None:
        ui = self.host
        ui.stream_stop_event.set()
        if sd_module is not None:
            try:
                sd_module.stop()
            except Exception:
                ui.logger.exception("Failed to stop stream playback")
        ui.stream_status_var.set("Stopping stream...")

    def finalize_stream_buttons(self) -> None:
        ui = self.host
        ui.stream_btn.state(["!disabled"])
        ui.stop_stream_btn.state(["disabled"])
