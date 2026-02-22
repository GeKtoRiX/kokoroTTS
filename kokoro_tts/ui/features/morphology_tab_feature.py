"""Morphology tab feature wiring and actions."""

from __future__ import annotations

from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk
from typing import Any

from ..common import extract_morph_headers


class MorphologyTabFeature:
    """Build and handle Morphology tab interactions."""

    def __init__(self, host) -> None:
        self.host = host

    def build_tab(self, parent: ttk.Frame) -> None:
        ui = self.host
        assert ui.morph_dataset_var is not None
        assert ui.morph_limit_var is not None
        assert ui.morph_offset_var is not None
        assert ui.morph_status_var is not None
        frame = ttk.Frame(parent, padding=8)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=(
                "Read-only preview for morphology.sqlite3. "
                "Use an external database tool to edit rows."
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w")

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(8, 8))
        ttk.Label(controls, text="Dataset").pack(side="left")
        dataset_combo = ttk.Combobox(
            controls,
            textvariable=ui.morph_dataset_var,
            state="readonly",
            values=["occurrences", "lexemes", "expressions"],
            width=16,
        )
        dataset_combo.pack(side="left", padx=(8, 12))
        dataset_combo.bind("<<ComboboxSelected>>", lambda _e: ui._on_morph_refresh())
        ttk.Label(controls, text="Limit").pack(side="left")
        ttk.Spinbox(
            controls,
            from_=1,
            to=1000,
            increment=1,
            textvariable=ui.morph_limit_var,
            width=8,
        ).pack(side="left", padx=(8, 12))
        ttk.Label(controls, text="Offset").pack(side="left")
        ttk.Spinbox(
            controls,
            from_=0,
            to=1000000,
            increment=1,
            textvariable=ui.morph_offset_var,
            width=8,
        ).pack(side="left", padx=(8, 12))
        ttk.Button(controls, text="Refresh", command=ui._on_morph_refresh).pack(side="left")

        table_wrap = ttk.Frame(frame)
        table_wrap.pack(fill="both", expand=True)
        ui.morph_tree = ttk.Treeview(table_wrap, show="headings", style="Treeview")
        y_scroll = ui._create_scrollbar(table_wrap, orient=tk.VERTICAL, command=ui.morph_tree.yview)
        x_scroll = ui._create_scrollbar(
            table_wrap, orient=tk.HORIZONTAL, command=ui.morph_tree.xview
        )
        ui.morph_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        ui.morph_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_wrap.rowconfigure(0, weight=1)
        table_wrap.columnconfigure(0, weight=1)

        ttk.Label(frame, textvariable=ui.morph_status_var, wraplength=760).pack(
            fill="x",
            pady=(6, 6),
        )

        ui._apply_table_update({"headers": ["No data"], "value": [[]]})

    def on_export_morphology(self) -> None:
        ui = self.host
        if not callable(ui.export_morphology_sheet):
            ui.export_status_var.set("Morphology DB export is not configured.")
            return
        dataset = ui.export_dataset_var.get()
        export_format = ui.export_format_var.get()

        def work():
            if ui.export_supports_format:
                return ui.export_morphology_sheet(dataset, export_format)
            return ui.export_morphology_sheet(dataset)

        def on_success(payload: tuple[str | None, str]) -> None:
            path, status = payload
            ui.export_status_var.set(status)
            ui.export_path_var.set(path or "")
            ui.generate_status_var.set(status if not path else f"{status} {os.path.basename(path)}")
            ui._on_morphology_preview_dataset_change()

        ui._threaded(work, on_success)

    def on_morphology_preview_dataset_change(self) -> None:
        ui = self.host
        if ui.morph_preview_status_var is None:
            return
        dataset = (
            str(ui.export_dataset_var.get() if ui.export_dataset_var is not None else "lexemes")
            .strip()
            .lower()
        )
        if not callable(ui.morphology_db_view):
            ui._set_morphology_preview_table(
                ["No data"], [["No data"]], rows_count=0, unique_count=0
            )
            return
        ui.morph_preview_status_var.set("Rows: 0 | Unique: 0 | Last updated: loading...")

        def work() -> tuple[str, dict[str, Any], str]:
            if dataset == "pos_table":
                table_update, status = ui.morphology_db_view("lexemes", 1000, 0)
                pos_update = ui._build_pos_table_preview_from_lexemes(table_update)
                return dataset, pos_update, status
            table_update, status = ui.morphology_db_view(dataset, 50, 0)
            return dataset, table_update, status

        def on_success(payload: tuple[str, dict[str, Any], str]) -> None:
            selected_dataset, table_update, _status = payload
            headers, rows = ui._project_morphology_preview_rows(selected_dataset, table_update)
            if not rows:
                fallback_headers = headers if headers else ["No data"]
                ui._set_morphology_preview_table(
                    fallback_headers, [["No data"]], rows_count=0, unique_count=0
                )
                return
            ui._set_morphology_preview_table(
                headers,
                rows,
                rows_count=len(rows),
                unique_count=ui._count_unique_non_empty_cells(rows),
            )

        ui._threaded(work, on_success)

    def project_morphology_preview_rows(
        self,
        dataset: str,
        table_update: dict[str, Any],
    ) -> tuple[list[str], list[list[str]]]:
        ui = self.host
        return ui._project_morphology_preview_rows_impl(dataset, table_update)

    def build_pos_table_preview_from_lexemes(self, table_update: dict[str, Any]) -> dict[str, Any]:
        ui = self.host
        return ui._build_pos_table_preview_from_lexemes_impl(table_update)

    def set_morphology_preview_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        rows_count: int,
        unique_count: int,
    ) -> None:
        ui = self.host
        if ui.morph_preview_tree is None or ui.morph_preview_status_var is None:
            return
        safe_headers, safe_rows = ui._format_morphology_preview_table(headers, rows)

        ui.morph_preview_headers = safe_headers
        ui.morph_preview_tree.delete(*ui.morph_preview_tree.get_children())
        ui.morph_preview_tree.configure(columns=ui.morph_preview_headers)
        for header in ui.morph_preview_headers:
            ui.morph_preview_tree.heading(header, text=header)
            ui.morph_preview_tree.column(header, width=140, stretch=True, anchor="w")
        for row in safe_rows:
            ui.morph_preview_tree.insert("", tk.END, values=row)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ui.morph_preview_status_var.set(
            f"Rows: {int(rows_count)} | Unique: {int(unique_count)} | Last updated: {timestamp}"
        )

    def on_morph_refresh(self) -> None:
        ui = self.host
        if not callable(ui.morphology_db_view):
            ui.morph_status_var.set("Morphology DB is not configured.")
            return
        dataset = ui.morph_dataset_var.get()
        limit = int(ui.morph_limit_var.get())
        offset = int(ui.morph_offset_var.get())
        ui.morph_status_var.set("Loading...")

        def work():
            return ui.morphology_db_view(dataset, limit, offset)

        def on_success(payload: tuple[dict[str, Any], str]) -> None:
            table_update, status = payload
            ui._apply_table_update(table_update)
            ui.morph_status_var.set(status)

        ui._threaded(work, on_success)

    def apply_table_update(self, table_update: dict[str, Any]) -> None:
        ui = self.host
        if ui.morph_tree is None:
            return
        headers = extract_morph_headers(table_update)
        rows = table_update.get("value", [])
        if not isinstance(rows, list):
            rows = []
        if not headers:
            headers = ["No data"]
            rows = [[]]
        ui.morph_headers = [str(item) for item in headers]
        ui.morph_tree.delete(*ui.morph_tree.get_children())
        ui.morph_tree.configure(columns=ui.morph_headers)
        for header in ui.morph_headers:
            ui.morph_tree.heading(header, text=header)
            ui.morph_tree.column(header, width=120, stretch=True, anchor="w")
        for row in rows:
            row_values = list(row) if isinstance(row, (list, tuple)) else [str(row)]
            if len(row_values) < len(ui.morph_headers):
                row_values.extend([""] * (len(ui.morph_headers) - len(row_values)))
            ui.morph_tree.insert("", tk.END, values=row_values[: len(ui.morph_headers)])
