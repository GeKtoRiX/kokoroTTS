"""Application layer orchestration."""

from .history_service import HistoryService
from .bootstrap import AppServices, initialize_app_services
from .context import AppContext
from .local_api import LocalKokoroApi
from .morphology_projection import (
    build_pos_table_preview_from_lexemes,
    count_unique_non_empty_cells,
    format_morphology_preview_table,
    project_morphology_preview_rows,
)
from .ports import KokoroTtsPort
from .state import KokoroState
from .ui_hooks import UiHooks

__all__ = [
    "AppContext",
    "AppServices",
    "HistoryService",
    "KokoroTtsPort",
    "KokoroState",
    "LocalKokoroApi",
    "UiHooks",
    "build_pos_table_preview_from_lexemes",
    "count_unique_non_empty_cells",
    "format_morphology_preview_table",
    "initialize_app_services",
    "project_morphology_preview_rows",
]
