"""Gradio UI construction for the Kokoro TTS app."""
from __future__ import annotations

import inspect
import json
import os
from typing import Any, Sequence

import gradio as gr

from ..config import AppConfig
from ..constants import OUTPUT_FORMATS
from ..domain.morphology_datasets import (
    morphology_primary_key,
    normalize_morphology_dataset,
)
from ..domain.style import DEFAULT_STYLE_PRESET, STYLE_PRESET_CHOICES
from ..domain.voice import (
    DEFAULT_VOICE,
    LANGUAGE_CHOICES,
    default_voice_for_lang,
    get_voice_choices,
    normalize_lang_code,
    voice_language,
)

UI_PRIMARY_HUE = os.getenv("UI_PRIMARY_HUE", "green").strip() or "green"
APP_TITLE = "KokoroTTS"
APP_THEME = gr.themes.Base(primary_hue=UI_PRIMARY_HUE)
APP_CSS = """
#runtime-mode-selector .wrap {
  flex-direction: column;
  align-items: stretch;
}

#runtime-mode-selector .wrap > label {
  width: 100%;
}

#main-layout {
  align-items: flex-start;
}

#generate-stack {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
  margin-bottom: 8px;
}

#app-title {
  margin-bottom: 8px;
}

"""

TOKEN_NOTE = (
    "\nCustomize pronunciation with Markdown link syntax and /slashes/ like "
    "`[Kokoro](/k o k o r o/)`\n\n"
    "To adjust intonation, try punctuation `;:,.!?\\\"()` and stress markers.\n\n"
    "Lower stress: `[1 level](-1)` or `[2 levels](-2)`\n\n"
    "Raise stress: `[or](+2)` (or +1 where supported)\n"
)

DIALOGUE_NOTE = (
    "\nUse [voice=af_heart] to switch speakers inside the text.\n"
    "Use [style=neutral|narrator|energetic] to switch style per segment.\n"
    "Use [pause=0.35] (or [pause=350ms], [pause=default]) to control pauses per segment.\n"
    "Mix voices with commas: [voice=af_heart,am_michael].\n"
)


RUNTIME_MODE_DEFAULT = "default"
RUNTIME_MODE_TTS_MORPH = "tts_morph"
RUNTIME_MODE_FULL = "full"
RUNTIME_MODE_CHOICES: list[tuple[str, str]] = [
    ("Default", RUNTIME_MODE_DEFAULT),
    ("TTS + Morphology", RUNTIME_MODE_TTS_MORPH),
    ("Full", RUNTIME_MODE_FULL),
]


def _tts_only_mode_status_text(enabled: bool) -> str:
    if enabled:
        return "TTS-only mode is ON: Morphology DB and LLM requests are disabled."
    return "TTS-only mode is OFF: Morphology DB and LLM requests are enabled."


def _llm_only_mode_status_text(enabled: bool, *, tts_only_enabled: bool) -> str:
    if tts_only_enabled:
        return "TTS + Morphology mode is overridden by TTS-only mode."
    if enabled:
        return "TTS + Morphology mode is ON: LLM requests are disabled, Morphology DB stays enabled."
    return "TTS + Morphology mode is OFF: LLM requests are enabled."


def _runtime_mode_from_flags(*, tts_only_enabled: bool, llm_only_enabled: bool) -> str:
    if tts_only_enabled:
        return RUNTIME_MODE_DEFAULT
    if llm_only_enabled:
        return RUNTIME_MODE_TTS_MORPH
    return RUNTIME_MODE_FULL


def _normalize_runtime_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == RUNTIME_MODE_TTS_MORPH:
        return RUNTIME_MODE_TTS_MORPH
    if normalized == RUNTIME_MODE_DEFAULT:
        return RUNTIME_MODE_DEFAULT
    return RUNTIME_MODE_FULL


def _runtime_mode_status_text(mode_value: Any) -> str:
    mode = _normalize_runtime_mode(mode_value)
    if mode == RUNTIME_MODE_DEFAULT:
        return "Default mode is active: TTS only."
    if mode == RUNTIME_MODE_TTS_MORPH:
        return "TTS + Morphology mode is active."
    return "Full mode is active."


def _runtime_mode_tab_visibility(mode_value: Any) -> tuple[bool, bool]:
    mode = _normalize_runtime_mode(mode_value)
    if mode == RUNTIME_MODE_DEFAULT:
        return False, False
    if mode == RUNTIME_MODE_TTS_MORPH:
        return False, True
    return True, True


def _normalize_morph_dataset(dataset: Any) -> str:
    return normalize_morphology_dataset(dataset)


def _extract_morph_headers(table_update: Any) -> list[str]:
    if not isinstance(table_update, dict):
        return []
    headers = table_update.get("headers")
    if not isinstance(headers, list):
        return []
    return [str(header) for header in headers]


def _morph_primary_key(dataset: Any) -> str:
    return morphology_primary_key(dataset)


def _supports_export_format_arg(callback: Any) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return True

    parameters = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return True
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return True
    positional = [
        param
        for param in parameters
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return True
    return any(param.name == "file_format" for param in parameters)


def _coerce_int_value(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    raw = str(value or "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _coerce_wordnet_hit(value: Any) -> int:
    raw = str(value or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return 0
    if raw in ("1", "true", "yes", "on"):
        return 1
    return 1 if _coerce_int_value(value, 0) != 0 else 0


def _coerce_morph_cell_value(dataset: str, column: str, value: Any) -> Any:
    numeric_columns = {
        "occurrences": {"part_index", "segment_index", "token_index", "start_offset", "end_offset"},
        "expressions": {"part_index", "segment_index", "expression_index", "start_offset", "end_offset"},
        "reviews": {
            "part_index",
            "segment_index",
            "token_index",
            "start_offset",
            "end_offset",
            "is_match",
            "attempt_count",
        },
    }
    if dataset == "expressions" and column == "wordnet_hit":
        return _coerce_wordnet_hit(value)
    if column in numeric_columns.get(dataset, set()):
        return _coerce_int_value(value, 0)
    if value is None:
        return ""
    return str(value)


def _build_morph_update_payload(
    dataset: Any,
    headers: Sequence[Any],
    row_value: Sequence[Any] | None,
) -> tuple[str, dict[str, Any]]:
    if not row_value:
        raise ValueError("No row selected.")

    normalized_dataset = _normalize_morph_dataset(dataset)
    normalized_headers = [str(header) for header in (headers or [])]
    if not normalized_headers:
        raise ValueError("No table headers available.")

    primary_key = _morph_primary_key(normalized_dataset)
    if primary_key not in normalized_headers:
        raise ValueError(f"Primary key column '{primary_key}' is missing.")

    row_items = list(row_value)
    key_index = normalized_headers.index(primary_key)
    if key_index >= len(row_items):
        raise ValueError("Selected row does not include a primary key value.")

    selected_row_id = str(row_items[key_index] or "").strip()
    if not selected_row_id:
        raise ValueError("Selected row id/key is empty.")

    payload: dict[str, Any] = {}
    for index, column in enumerate(normalized_headers):
        if index >= len(row_items):
            continue
        if column in ("id", "created_at"):
            continue
        if normalized_dataset == "lexemes" and column == "dedup_key":
            continue
        payload[column] = _coerce_morph_cell_value(normalized_dataset, column, row_items[index])

    return selected_row_id, payload


def _resolve_morph_delete_confirmation(
    selected_row_id: Any,
    armed_row_id: Any,
) -> tuple[bool, str, str]:
    selected = str(selected_row_id or "").strip()
    armed = str(armed_row_id or "").strip()
    if not selected:
        return False, "", "Select a row before deleting."
    if armed != selected:
        return (
            False,
            selected,
            f"Press Delete row again to confirm deleting id/key={selected}.",
        )
    return True, "", ""


def _wire_generation_events(
    *,
    language,
    voice,
    voice_mix,
    mix_enabled,
    text,
    speed,
    use_gpu,
    pause_between,
    output_format,
    normalize_times_toggle,
    normalize_numbers_toggle,
    style_preset,
    out_audio,
    out_ps,
    out_stream,
    generate_btn,
    tokenize_btn,
    stream_btn,
    stop_btn,
    predict_btn,
    api_name: str | None | bool,
    generate_first,
    tokenize_first,
    generate_all,
    predict,
    clear_stream_output,
    update_history,
    clear_history,
    history_state,
    history_audios: Sequence[Any],
    clear_history_btn,
    on_language_change,
    on_voice_change,
    on_mix_change,
    toggle_mix_controls,
) -> None:
    language.change(
        fn=on_language_change,
        inputs=[language, voice, voice_mix],
        outputs=[language, voice, voice_mix],
    )
    voice.change(fn=on_voice_change, inputs=[voice], outputs=[language])
    voice_mix.change(fn=on_mix_change, inputs=[voice_mix, language], outputs=[language])
    mix_enabled.change(fn=toggle_mix_controls, inputs=[mix_enabled], outputs=[voice_mix, voice])

    generate_event = generate_btn.click(
        fn=generate_first,
        inputs=[
            text,
            voice,
            mix_enabled,
            voice_mix,
            speed,
            use_gpu,
            pause_between,
            output_format,
            normalize_times_toggle,
            normalize_numbers_toggle,
            style_preset,
        ],
        outputs=[out_audio, out_ps],
        api_name=api_name,
    )
    generate_event.then(
        fn=update_history,
        inputs=[history_state],
        outputs=[history_state] + list(history_audios),
    )
    if clear_history_btn is not None:
        clear_history_btn.click(
            fn=clear_history,
            inputs=[history_state],
            outputs=[history_state] + list(history_audios),
        )
    tokenize_btn.click(
        fn=tokenize_first,
        inputs=[
            text,
            voice,
            mix_enabled,
            voice_mix,
            speed,
            normalize_times_toggle,
            normalize_numbers_toggle,
            style_preset,
        ],
        outputs=[out_ps],
        api_name=api_name,
    )
    stream_prepare_event = stream_btn.click(
        fn=clear_stream_output,
        outputs=[out_stream],
        queue=False,
        api_name=False,
    )
    stream_event = stream_prepare_event.then(
        fn=generate_all,
        inputs=[
            text,
            voice,
            mix_enabled,
            voice_mix,
            speed,
            use_gpu,
            pause_between,
            normalize_times_toggle,
            normalize_numbers_toggle,
            style_preset,
        ],
        outputs=[out_stream],
        api_name=api_name,
    )
    stop_btn.click(
        fn=clear_stream_output,
        outputs=[out_stream],
        cancels=stream_event,
        queue=False,
        api_name=False,
    )
    predict_btn.click(
        fn=predict,
        inputs=[
            text,
            voice,
            mix_enabled,
            voice_mix,
            speed,
            normalize_times_toggle,
            normalize_numbers_toggle,
            style_preset,
        ],
        outputs=[out_audio],
        api_name=api_name,
    )


def _wire_morphology_events(
    *,
    export_csv_btn,
    export_morphology_sheet,
    export_csv_dataset,
    export_csv_format,
    export_csv_file,
    export_csv_status,
    morph_db_refresh_btn,
    morph_db_dataset,
    morph_db_limit,
    morph_db_offset,
    morph_db_table,
    morph_db_status,
    morph_db_headers_state,
    morph_db_selected_row,
    morph_db_delete_armed_state,
    morph_db_update_json,
    morph_db_add_btn,
    morph_db_add_json,
    morph_db_update_btn,
    morph_db_delete_btn,
    morph_db_view,
    morph_db_view_wrapped,
    morph_db_add,
    morph_db_add_wrapped,
    morph_db_select_row,
    morph_db_update,
    morph_db_update_selected,
    morph_db_delete,
    morph_db_delete_selected,
) -> None:
    if export_csv_btn is not None:
        export_csv_btn.click(
            fn=export_morphology_sheet,
            inputs=[export_csv_dataset, export_csv_format],
            outputs=[export_csv_file, export_csv_status],
            api_name=False,
        )
    if callable(morph_db_view):
        morph_db_refresh_btn.click(
            fn=morph_db_view_wrapped,
            inputs=[morph_db_dataset, morph_db_limit, morph_db_offset],
            outputs=[
                morph_db_table,
                morph_db_status,
                morph_db_headers_state,
                morph_db_selected_row,
                morph_db_delete_armed_state,
            ],
            api_name=False,
        )
        morph_db_dataset.change(
            fn=morph_db_view_wrapped,
            inputs=[morph_db_dataset, morph_db_limit, morph_db_offset],
            outputs=[
                morph_db_table,
                morph_db_status,
                morph_db_headers_state,
                morph_db_selected_row,
                morph_db_delete_armed_state,
            ],
            api_name=False,
        )
        morph_db_table.select(
            fn=morph_db_select_row,
            inputs=[morph_db_dataset, morph_db_headers_state, morph_db_update_json],
            outputs=[
                morph_db_selected_row,
                morph_db_update_json,
                morph_db_status,
                morph_db_delete_armed_state,
            ],
            api_name=False,
        )
    if callable(morph_db_add):
        morph_db_add_btn.click(
            fn=morph_db_add_wrapped,
            inputs=[morph_db_dataset, morph_db_add_json, morph_db_limit, morph_db_offset],
            outputs=[
                morph_db_table,
                morph_db_status,
                morph_db_headers_state,
                morph_db_selected_row,
                morph_db_delete_armed_state,
            ],
            api_name=False,
        )
    if callable(morph_db_update):
        morph_db_update_btn.click(
            fn=morph_db_update_selected,
            inputs=[
                morph_db_dataset,
                morph_db_selected_row,
                morph_db_update_json,
                morph_db_limit,
                morph_db_offset,
                morph_db_headers_state,
            ],
            outputs=[
                morph_db_table,
                morph_db_status,
                morph_db_headers_state,
                morph_db_selected_row,
                morph_db_delete_armed_state,
            ],
            api_name=False,
        )
    if callable(morph_db_delete):
        morph_db_delete_btn.click(
            fn=morph_db_delete_selected,
            inputs=[
                morph_db_dataset,
                morph_db_selected_row,
                morph_db_limit,
                morph_db_offset,
                morph_db_headers_state,
                morph_db_delete_armed_state,
            ],
            outputs=[
                morph_db_table,
                morph_db_status,
                morph_db_headers_state,
                morph_db_selected_row,
                morph_db_delete_armed_state,
            ],
            api_name=False,
        )


def _wire_pronunciation_and_lesson_events(
    *,
    pronunciation_load_btn,
    pronunciation_apply_btn,
    pronunciation_import_btn,
    pronunciation_export_btn,
    pronunciation_rules_json,
    pronunciation_status,
    pronunciation_import_file,
    pronunciation_export_file,
    load_pronunciation_rules,
    apply_pronunciation_rules,
    import_pronunciation_rules,
    export_pronunciation_rules,
    build_lesson_for_tts,
    llm_generate_btn,
    llm_raw_text,
    llm_base_url,
    llm_api_key,
    llm_model,
    llm_temperature,
    llm_max_tokens,
    llm_timeout_seconds,
    llm_extra_instructions,
    llm_output_text,
    llm_status,
    llm_to_tts_btn,
    text,
) -> None:
    if callable(load_pronunciation_rules):
        pronunciation_load_btn.click(
            fn=load_pronunciation_rules,
            outputs=[pronunciation_rules_json, pronunciation_status],
            api_name=False,
        )
    if callable(apply_pronunciation_rules):
        pronunciation_apply_btn.click(
            fn=apply_pronunciation_rules,
            inputs=[pronunciation_rules_json],
            outputs=[pronunciation_rules_json, pronunciation_status],
            api_name=False,
        )
    if callable(import_pronunciation_rules):
        pronunciation_import_btn.click(
            fn=import_pronunciation_rules,
            inputs=[pronunciation_import_file],
            outputs=[pronunciation_rules_json, pronunciation_status],
            api_name=False,
        )
    if callable(export_pronunciation_rules):
        pronunciation_export_btn.click(
            fn=export_pronunciation_rules,
            outputs=[pronunciation_export_file, pronunciation_status],
            api_name=False,
        )
    if callable(build_lesson_for_tts):
        llm_generate_btn.click(
            fn=build_lesson_for_tts,
            inputs=[
                llm_raw_text,
                llm_base_url,
                llm_api_key,
                llm_model,
                llm_temperature,
                llm_max_tokens,
                llm_timeout_seconds,
                llm_extra_instructions,
            ],
            outputs=[llm_output_text, llm_status],
            api_name=False,
        )
    if llm_to_tts_btn is not None:
        llm_to_tts_btn.click(
            fn=lambda generated_text: generated_text or "",
            inputs=[llm_output_text],
            outputs=[text],
            api_name=False,
        )


def create_gradio_app(
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
    history_service,
    choices,
) -> tuple[gr.Blocks, bool]:
    _ = choices  # Preserved for backwards compatibility with existing call sites.
    default_output_format = (
        config.default_output_format
        if config.default_output_format in OUTPUT_FORMATS
        else "wav"
    )
    default_lang = voice_language(DEFAULT_VOICE)
    default_voice = default_voice_for_lang(default_lang)
    default_voice_choices = get_voice_choices(default_lang)

    history_state = None
    history_audios: list[gr.Audio] = []
    clear_history_btn = None
    export_csv_btn = None
    export_csv_dataset = None
    export_csv_format = None
    export_csv_file = None
    export_csv_status = None
    out_audio = None
    out_ps = None
    tokenize_btn = None
    predict_btn = None
    generate_btn = None
    out_stream = None
    stream_btn = None
    stop_btn = None
    style_preset = None
    pronunciation_rules_json = None
    pronunciation_status = None
    pronunciation_load_btn = None
    pronunciation_apply_btn = None
    pronunciation_import_file = None
    pronunciation_import_btn = None
    pronunciation_export_btn = None
    pronunciation_export_file = None
    runtime_mode_selector = None
    runtime_mode_status = None
    llm_raw_text = None
    llm_base_url = None
    llm_api_key = None
    llm_model = None
    llm_temperature = None
    llm_max_tokens = None
    llm_timeout_seconds = None
    llm_extra_instructions = None
    llm_generate_btn = None
    llm_to_tts_btn = None
    llm_output_text = None
    llm_status = None
    lesson_tab = None
    morphology_tab = None
    morph_db_dataset = None
    morph_db_limit = None
    morph_db_offset = None
    morph_db_refresh_btn = None
    morph_db_table = None
    morph_db_status = None
    morph_db_add_json = None
    morph_db_add_btn = None
    morph_db_selected_row = None
    morph_db_update_json = None
    morph_db_update_btn = None
    morph_db_delete_btn = None
    morph_db_headers_state = None
    morph_db_delete_armed_state = None

    stream_note = [
        "There is a known Gradio issue that may produce no audio the first time you click `Stream`."
    ]
    if config.char_limit is not None:
        stream_note.append(f"Each stream is capped at {config.char_limit} characters.")
        stream_note.append(
            "Need more characters? You can "
            "[use Kokoro directly](https://huggingface.co/hexgrad/Kokoro-82M#usage) "
            "or duplicate this space:"
        )
    stream_note = "\n\n".join(stream_note)
    export_supports_format = bool(
        callable(export_morphology_sheet) and _supports_export_format_arg(export_morphology_sheet)
    )
    runtime_tts_only_enabled = bool(tts_only_mode_default)
    runtime_llm_only_enabled = bool(llm_only_mode_default)
    if not runtime_tts_only_enabled and not runtime_llm_only_enabled:
        # Prefer plain TTS by default unless a mode is explicitly enabled.
        runtime_mode_value = RUNTIME_MODE_DEFAULT
        runtime_tts_only_enabled = True
    else:
        runtime_mode_value = _runtime_mode_from_flags(
            tts_only_enabled=runtime_tts_only_enabled,
            llm_only_enabled=runtime_llm_only_enabled,
        )
    lesson_tab_visible, morphology_tab_visible = _runtime_mode_tab_visibility(runtime_mode_value)

    def toggle_mix_controls(enabled):
        return gr.update(visible=enabled), gr.update(interactive=not enabled)

    def on_language_change(selected_lang, selected_voice, selected_mix):
        normalized_lang = normalize_lang_code(
            selected_lang,
            default=voice_language(selected_voice or default_voice),
        )
        options = get_voice_choices(normalized_lang)
        valid_ids = {voice_id for _, voice_id in options}
        next_voice = selected_voice if selected_voice in valid_ids else default_voice_for_lang(normalized_lang)
        next_mix = [voice_id for voice_id in (selected_mix or []) if voice_id in valid_ids]
        return (
            gr.update(value=normalized_lang),
            gr.update(choices=options, value=next_voice),
            gr.update(choices=options, value=next_mix),
        )

    def on_voice_change(selected_voice):
        return gr.update(value=voice_language(selected_voice, default=default_lang))

    def on_mix_change(selected_mix, current_lang):
        if selected_mix:
            return gr.update(value=voice_language(selected_mix[0], default=default_lang))
        return gr.update(value=normalize_lang_code(current_lang, default=default_lang))

    def update_history(history):
        updated = history_service.update_history(history)
        values = updated + [None] * (config.history_limit - len(updated))
        return (updated, *values)

    def clear_history(history):
        updated = history_service.clear_history(history)
        values = [None] * config.history_limit
        return (updated, *values)

    def clear_stream_output():
        return None

    def set_tts_only_mode_wrapped(enabled):
        nonlocal runtime_tts_only_enabled
        runtime_tts_only_enabled = bool(enabled)
        if not callable(set_tts_only_mode):
            return _tts_only_mode_status_text(runtime_tts_only_enabled)
        status = set_tts_only_mode(runtime_tts_only_enabled)
        return str(status or _tts_only_mode_status_text(runtime_tts_only_enabled))

    def set_llm_only_mode_wrapped(enabled):
        nonlocal runtime_llm_only_enabled
        runtime_llm_only_enabled = bool(enabled)
        if not callable(set_llm_only_mode):
            return _llm_only_mode_status_text(
                runtime_llm_only_enabled,
                tts_only_enabled=runtime_tts_only_enabled,
            )
        status = set_llm_only_mode(runtime_llm_only_enabled)
        return str(
            status
            or _llm_only_mode_status_text(
                runtime_llm_only_enabled,
                tts_only_enabled=runtime_tts_only_enabled,
            )
        )

    def set_runtime_mode_wrapped(mode_value):
        selected_mode = _normalize_runtime_mode(mode_value)
        if selected_mode == RUNTIME_MODE_DEFAULT:
            set_tts_only_mode_wrapped(True)
            set_llm_only_mode_wrapped(False)
        elif selected_mode == RUNTIME_MODE_TTS_MORPH:
            set_tts_only_mode_wrapped(False)
            set_llm_only_mode_wrapped(True)
        else:
            set_tts_only_mode_wrapped(False)
            set_llm_only_mode_wrapped(False)
        lesson_visible, morph_visible = _runtime_mode_tab_visibility(selected_mode)
        return (
            _runtime_mode_status_text(selected_mode),
            gr.update(visible=lesson_visible),
            gr.update(visible=morph_visible),
        )

    # Apply selected runtime mode once during UI assembly to keep backend flags in sync.
    _ = set_runtime_mode_wrapped(runtime_mode_value)

    def export_morphology_sheet_wrapped(dataset, export_format):
        if not callable(export_morphology_sheet):
            return None, "Morphology DB export is not configured."
        if export_supports_format:
            return export_morphology_sheet(dataset, export_format)
        return export_morphology_sheet(dataset)

    def morph_db_view_wrapped(dataset, limit, offset):
        table_update, status = morphology_db_view(dataset, limit, offset)
        return table_update, status, _extract_morph_headers(table_update), "", ""

    def morph_db_add_wrapped(dataset, row_json, limit, offset):
        table_update, status = morphology_db_add(dataset, row_json, limit, offset)
        return table_update, status, _extract_morph_headers(table_update), "", ""

    def morph_db_select_row(
        dataset,
        headers,
        current_update_json,
        evt: gr.SelectData = None,
    ):
        row_value = getattr(evt, "row_value", None)
        try:
            selected_row_id, payload = _build_morph_update_payload(dataset, headers, row_value)
        except ValueError as exc:
            return "", current_update_json, f"Selection failed: {exc}", ""
        return (
            selected_row_id,
            json.dumps(payload, ensure_ascii=False),
            f"Selected row id/key={selected_row_id}.",
            "",
        )

    def morph_db_update_selected(dataset, selected_row_id, row_json, limit, offset, headers):
        row_id = str(selected_row_id or "").strip()
        current_headers = [str(header) for header in (headers or [])]
        if not row_id:
            return gr.update(), "Select a row before updating.", current_headers, "", ""
        table_update, status = morphology_db_update(dataset, row_id, row_json, limit, offset)
        return table_update, status, _extract_morph_headers(table_update), "", ""

    def morph_db_delete_selected(dataset, selected_row_id, limit, offset, headers, armed_row_id):
        should_delete, next_armed, confirm_status = _resolve_morph_delete_confirmation(
            selected_row_id,
            armed_row_id,
        )
        current_headers = [str(header) for header in (headers or [])]
        selected = str(selected_row_id or "").strip()
        if not should_delete:
            return gr.update(), confirm_status, current_headers, selected, next_armed
        table_update, status = morphology_db_delete(dataset, selected, limit, offset)
        return table_update, status, _extract_morph_headers(table_update), "", ""

    api_open = config.space_id != "hexgrad/Kokoro-TTS"
    api_name = None if api_open else False
    logger.debug("API_OPEN=%s", api_open)
    with gr.Blocks(title=APP_TITLE) as app:
        history_state = gr.State([])
        gr.Markdown(f"# {APP_TITLE}", elem_id="app-title")
        with gr.Row(elem_id="main-layout"):
            with gr.Column():
                with gr.Accordion("Input", open=True):
                    text = gr.Textbox(
                        label="Input Text",
                        lines=6,
                        min_width=360,
                        info="Use | to split into separate files. Use [voice=af_heart], [style=narrator], [pause=0.3].",
                    )
                    language = gr.Dropdown(
                        LANGUAGE_CHOICES,
                        value=default_lang,
                        label="Language",
                        info="Filters voices by language. Dialog tags can still override per segment.",
                    )
                    voice = gr.Dropdown(
                        default_voice_choices,
                        value=default_voice,
                        label="Voice",
                        info="Quality and availability vary by language",
                    )
                    mix_enabled = gr.Checkbox(label="Mix voices", value=False)
                    voice_mix = gr.Dropdown(
                        default_voice_choices,
                        value=[],
                        multiselect=True,
                        label="Voice mix",
                        info="Select multiple voices to average",
                        visible=False,
                    )
                with gr.Accordion("Hardware", open=False):
                    with gr.Row():
                        use_gpu = gr.Dropdown(
                            [("ZeroGPU", True), ("CPU", False)],
                            value=cuda_available,
                            label="Hardware",
                            info="GPU is usually faster, but has a usage quota",
                            interactive=cuda_available,
                        )
                with gr.Accordion("Generation settings", open=False):
                    speed = gr.Slider(minimum=0.5, maximum=2, value=0.8, step=0.1, label="Speed")
                    style_preset = gr.Dropdown(
                        STYLE_PRESET_CHOICES,
                        value=DEFAULT_STYLE_PRESET,
                        label="Style preset",
                        info=(
                            "Kokoro has no native emotion controls. Presets tune runtime speed/pause "
                            "and pass style to the pipeline only if supported."
                        ),
                    )
                    pause_between = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=0.3,
                        step=0.1,
                        label="Pause between sentences (s)",
                        info="Applies to Generate and Stream output",
                    )
                    output_format = gr.Dropdown(
                        OUTPUT_FORMATS,
                        value=default_output_format,
                        label="Output format",
                        info="Applies to saved files in History/outputs (mp3/ogg require ffmpeg)",
                    )
                with gr.Accordion("Text normalization", open=False):
                    normalize_times_toggle = gr.Checkbox(
                        label="Normalize times (12:30 -> twelve thirty)",
                        value=config.normalize_times,
                    )
                    normalize_numbers_toggle = gr.Checkbox(
                        label="Normalize numbers (0-9999, decimals, %, ordinals)",
                        value=config.normalize_numbers,
                    )
                with gr.Accordion("Runtime mode", open=False):
                    runtime_mode_selector = gr.Radio(
                        RUNTIME_MODE_CHOICES,
                        value=runtime_mode_value,
                        label="Mode",
                        interactive=callable(set_tts_only_mode) or callable(set_llm_only_mode),
                        elem_id="runtime-mode-selector",
                    )
                    runtime_mode_status = gr.Textbox(
                        label="Mode status",
                        value=_runtime_mode_status_text(runtime_mode_value),
                        interactive=False,
                    )
                with gr.Accordion("Dialog tags", open=False):
                    gr.Markdown(DIALOGUE_NOTE)
                with gr.Accordion("Pronunciation dictionary", open=False):
                    gr.Markdown(
                        "Persistent JSON rules by language code (`a,b,e,f,h,i,j,p,z`). "
                        "Example: `{ \"a\": { \"OpenAI\": \"OW P AH N EY AY\" } }`"
                    )
                    pronunciation_rules_json = gr.Textbox(
                        label="Rules JSON",
                        value="{}",
                        lines=12,
                    )
                    pronunciation_status = gr.Textbox(
                        label="Dictionary status",
                        interactive=False,
                    )
                    with gr.Row():
                        pronunciation_load_btn = gr.Button(
                            "Load current",
                            variant="secondary",
                            interactive=callable(load_pronunciation_rules),
                        )
                        pronunciation_apply_btn = gr.Button(
                            "Apply",
                            variant="primary",
                            interactive=callable(apply_pronunciation_rules),
                        )
                    with gr.Row(equal_height=True):
                        pronunciation_import_file = gr.File(
                            label="Import JSON",
                            file_types=[".json"],
                            type="filepath",
                            height=160,
                        )
                        pronunciation_export_file = gr.File(
                            label="Exported file",
                            interactive=False,
                            height=160,
                        )
                    with gr.Row(equal_height=True):
                        pronunciation_import_btn = gr.Button(
                            "Import file",
                            variant="secondary",
                            interactive=callable(import_pronunciation_rules),
                        )
                        pronunciation_export_btn = gr.Button(
                            "Export JSON",
                            variant="secondary",
                            interactive=callable(export_pronunciation_rules),
                        )
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Generate"):
                        with gr.Column(elem_id="generate-stack"):
                            out_audio = gr.Audio(
                                label="Output Audio",
                                interactive=False,
                                streaming=False,
                                autoplay=True,
                            )
                            generate_btn = gr.Button("Generate", variant="primary")
                            if config.history_limit > 0:
                                with gr.Accordion("History", open=False):
                                    clear_history_btn = gr.Button("Clear history", variant="secondary")
                                    for index in range(1, config.history_limit + 1):
                                        history_audios.append(
                                            gr.Audio(
                                                label=f"History {index}",
                                                interactive=False,
                                                streaming=False,
                                            )
                                        )
                            with gr.Accordion("Output Tokens", open=False):
                                out_ps = gr.Textbox(
                                    interactive=False,
                                    show_label=False,
                                    info="Tokens used to generate the audio, up to 510 context length.",
                                )
                                tokenize_btn = gr.Button("Tokenize", variant="secondary")
                                gr.Markdown(TOKEN_NOTE)
                                predict_btn = gr.Button("Predict", variant="secondary", visible=False)
                            if config.morph_db_enabled and callable(export_morphology_sheet):
                                with gr.Accordion("Morphology DB Export", open=False):
                                    gr.Markdown(
                                        "Export selected dataset as `.ods`, `.csv`, `.txt`, or `.xlsx`."
                                    )
                                    export_csv_dataset = gr.Dropdown(
                                        [
                                            ("Lexemes", "lexemes"),
                                            ("Token occurrences", "occurrences"),
                                            ("Expressions (phrasal verbs and idioms)", "expressions"),
                                            ("LM reviews", "reviews"),
                                            ("General table", "pos_table"),
                                        ],
                                        value="lexemes",
                                        label="Dataset",
                                    )
                                    export_csv_format = gr.Dropdown(
                                        [
                                            ("ODS spreadsheet (.ods)", "ods"),
                                            ("CSV (.csv)", "csv"),
                                            ("Plain text table (.txt)", "txt"),
                                            ("Microsoft Excel (.xlsx)", "xlsx"),
                                        ],
                                        value="ods",
                                        label="Format",
                                    )
                                    export_csv_btn = gr.Button("Export file", variant="secondary")
                                    export_csv_file = gr.File(label="Exported file", interactive=False)
                                    export_csv_status = gr.Textbox(label="Export status", interactive=False)
                    with gr.Tab("Stream"):
                        out_stream = gr.Audio(
                            label="Output Audio Stream",
                            interactive=False,
                            streaming=True,
                            autoplay=True,
                        )
                        with gr.Row():
                            stream_btn = gr.Button("Stream", variant="primary")
                            stop_btn = gr.Button("Stop", variant="stop")
                        with gr.Accordion("Note", open=True):
                            gr.Markdown(stream_note)
                            gr.DuplicateButton()
                    with gr.Tab("Lesson Builder (LLM)", visible=lesson_tab_visible) as lesson_tab:
                        gr.Markdown(
                            "Transform raw material into an English lesson script with detailed "
                            "exercise explanations for TTS narration."
                        )
                        llm_raw_text = gr.Textbox(
                            label="Raw source text",
                            lines=12,
                            placeholder=(
                                "Paste lesson source text with tasks/exercises. "
                                "The model will rewrite it into a clear spoken lesson."
                            ),
                        )
                        with gr.Accordion("LM Studio settings", open=False):
                            llm_base_url = gr.Textbox(
                                label="Base URL",
                                value=config.lm_studio_base_url,
                                info="OpenAI-compatible endpoint (for example http://127.0.0.1:1234/v1)",
                            )
                            with gr.Row():
                                llm_model = gr.Textbox(
                                    label="Model",
                                    value=config.lm_studio_model,
                                )
                                llm_api_key = gr.Textbox(
                                    label="API key",
                                    value=config.lm_studio_api_key,
                                    type="password",
                                )
                            with gr.Row():
                                llm_temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=config.lm_studio_temperature,
                                    step=0.05,
                                    label="Temperature",
                                )
                                llm_max_tokens = gr.Slider(
                                    minimum=64,
                                    maximum=32768,
                                    value=config.lm_studio_max_tokens,
                                    step=64,
                                    label="Max tokens",
                                )
                                llm_timeout_seconds = gr.Slider(
                                    minimum=0,
                                    maximum=86400,
                                    value=config.lm_studio_timeout_seconds,
                                    step=5,
                                    label="Timeout (s)",
                                    info="Set 0 to disable timeout and wait until the model responds.",
                                )
                            llm_extra_instructions = gr.Textbox(
                                label="Extra instructions (optional)",
                                lines=4,
                                placeholder="Optional extra constraints for the lesson output.",
                            )
                        with gr.Row():
                            llm_generate_btn = gr.Button(
                                "Generate lesson",
                                variant="primary",
                                interactive=callable(build_lesson_for_tts),
                            )
                            llm_to_tts_btn = gr.Button(
                                "Use lesson as TTS input",
                                variant="secondary",
                            )
                        llm_output_text = gr.Textbox(
                            label="Lesson output (English)",
                            lines=16,
                        )
                        llm_status = gr.Textbox(
                            label="LLM status",
                            interactive=False,
                        )
                    with gr.Tab("Morphology DB", visible=morphology_tab_visible) as morphology_tab:
                        gr.Markdown(
                            "Simple CRUD for `morphology.sqlite3`: browse rows, add JSON row, "
                            "select a row in the table, then click Update/Delete."
                        )
                        with gr.Accordion("CRUD examples", open=False):
                            gr.Markdown(
                                "Add example (`occurrences`):\n"
                                "```json\n"
                                "{\"source\":\"manual\",\"token_text\":\"Cats\",\"lemma\":\"cat\",\"upos\":\"NOUN\"}\n"
                                "```\n\n"
                                "Update example (`occurrences`):\n"
                                "1. Select row with `id=15` in the table.\n"
                                "2. Edit generated Update JSON.\n"
                                "3. Click `Update row`.\n"
                                "```json\n"
                                "{\"token_text\":\"Dogs\",\"lemma\":\"dog\"}\n"
                                "```\n\n"
                                "Delete example (`occurrences`):\n"
                                "1. Select row with `id=15` in the table.\n"
                                "2. Click `Delete row` twice.\n\n"
                                "For `lexemes`, selection uses `dedup_key` (example: `run|verb`)."
                            )
                        with gr.Row():
                            morph_db_dataset = gr.Dropdown(
                                [
                                    ("Token occurrences", "occurrences"),
                                    ("Lexemes", "lexemes"),
                                    ("Expressions", "expressions"),
                                    ("LM reviews", "reviews"),
                                ],
                                value="occurrences",
                                label="Dataset",
                            )
                            morph_db_limit = gr.Slider(
                                minimum=1,
                                maximum=1000,
                                value=100,
                                step=1,
                                label="Limit",
                            )
                            morph_db_offset = gr.Number(
                                value=0,
                                precision=0,
                                label="Offset",
                            )
                            morph_db_refresh_btn = gr.Button(
                                "Refresh",
                                variant="secondary",
                                interactive=callable(morphology_db_view),
                            )
                        morph_db_headers_state = gr.State([])
                        morph_db_delete_armed_state = gr.State("")
                        morph_db_table = gr.Dataframe(
                            headers=["No data"],
                            value=[[]],
                            interactive=False,
                            wrap=True,
                            label="Rows",
                        )
                        morph_db_status = gr.Textbox(
                            label="DB status",
                            interactive=False,
                        )
                        with gr.Accordion("Add row", open=False):
                            morph_db_add_json = gr.Textbox(
                                label="Row JSON",
                                lines=8,
                                value='{"source":"manual"}',
                                info=(
                                    "Provide JSON object. Minimal fields are enough: "
                                    "missing values are auto-filled."
                                ),
                            )
                            morph_db_add_btn = gr.Button(
                                "Add row",
                                variant="primary",
                                interactive=callable(morphology_db_add),
                            )
                        with gr.Accordion("Update / Delete row", open=False):
                            morph_db_selected_row = gr.Textbox(
                                label="Selected row id/key",
                                interactive=False,
                            )
                            morph_db_update_json = gr.Textbox(
                                label="Update JSON",
                                lines=8,
                                value='{"source":"manual"}',
                                info="Select a row to auto-fill. Only provided fields are updated.",
                            )
                            with gr.Row():
                                morph_db_update_btn = gr.Button(
                                    "Update row",
                                    variant="secondary",
                                    interactive=callable(morphology_db_update),
                                )
                                morph_db_delete_btn = gr.Button(
                                    "Delete row",
                                    variant="stop",
                                    interactive=callable(morphology_db_delete),
                                )

        _wire_generation_events(
            language=language,
            voice=voice,
            voice_mix=voice_mix,
            mix_enabled=mix_enabled,
            text=text,
            speed=speed,
            use_gpu=use_gpu,
            pause_between=pause_between,
            output_format=output_format,
            normalize_times_toggle=normalize_times_toggle,
            normalize_numbers_toggle=normalize_numbers_toggle,
            style_preset=style_preset,
            out_audio=out_audio,
            out_ps=out_ps,
            out_stream=out_stream,
            generate_btn=generate_btn,
            tokenize_btn=tokenize_btn,
            stream_btn=stream_btn,
            stop_btn=stop_btn,
            predict_btn=predict_btn,
            api_name=api_name,
            generate_first=generate_first,
            tokenize_first=tokenize_first,
            generate_all=generate_all,
            predict=predict,
            clear_stream_output=clear_stream_output,
            update_history=update_history,
            clear_history=clear_history,
            history_state=history_state,
            history_audios=history_audios,
            clear_history_btn=clear_history_btn,
            on_language_change=on_language_change,
            on_voice_change=on_voice_change,
            on_mix_change=on_mix_change,
            toggle_mix_controls=toggle_mix_controls,
        )
        _wire_morphology_events(
            export_csv_btn=export_csv_btn,
            export_morphology_sheet=export_morphology_sheet_wrapped,
            export_csv_dataset=export_csv_dataset,
            export_csv_format=export_csv_format,
            export_csv_file=export_csv_file,
            export_csv_status=export_csv_status,
            morph_db_refresh_btn=morph_db_refresh_btn,
            morph_db_dataset=morph_db_dataset,
            morph_db_limit=morph_db_limit,
            morph_db_offset=morph_db_offset,
            morph_db_table=morph_db_table,
            morph_db_status=morph_db_status,
            morph_db_headers_state=morph_db_headers_state,
            morph_db_selected_row=morph_db_selected_row,
            morph_db_delete_armed_state=morph_db_delete_armed_state,
            morph_db_update_json=morph_db_update_json,
            morph_db_add_btn=morph_db_add_btn,
            morph_db_add_json=morph_db_add_json,
            morph_db_update_btn=morph_db_update_btn,
            morph_db_delete_btn=morph_db_delete_btn,
            morph_db_view=morphology_db_view,
            morph_db_view_wrapped=morph_db_view_wrapped,
            morph_db_add=morphology_db_add,
            morph_db_add_wrapped=morph_db_add_wrapped,
            morph_db_select_row=morph_db_select_row,
            morph_db_update=morphology_db_update,
            morph_db_update_selected=morph_db_update_selected,
            morph_db_delete=morphology_db_delete,
            morph_db_delete_selected=morph_db_delete_selected,
        )
        _wire_pronunciation_and_lesson_events(
            pronunciation_load_btn=pronunciation_load_btn,
            pronunciation_apply_btn=pronunciation_apply_btn,
            pronunciation_import_btn=pronunciation_import_btn,
            pronunciation_export_btn=pronunciation_export_btn,
            pronunciation_rules_json=pronunciation_rules_json,
            pronunciation_status=pronunciation_status,
            pronunciation_import_file=pronunciation_import_file,
            pronunciation_export_file=pronunciation_export_file,
            load_pronunciation_rules=load_pronunciation_rules,
            apply_pronunciation_rules=apply_pronunciation_rules,
            import_pronunciation_rules=import_pronunciation_rules,
            export_pronunciation_rules=export_pronunciation_rules,
            build_lesson_for_tts=build_lesson_for_tts,
            llm_generate_btn=llm_generate_btn,
            llm_raw_text=llm_raw_text,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_extra_instructions=llm_extra_instructions,
            llm_output_text=llm_output_text,
            llm_status=llm_status,
            llm_to_tts_btn=llm_to_tts_btn,
            text=text,
        )
        if (
            runtime_mode_selector is not None
            and runtime_mode_status is not None
            and lesson_tab is not None
            and morphology_tab is not None
        ):
            runtime_mode_selector.change(
                fn=set_runtime_mode_wrapped,
                inputs=[runtime_mode_selector],
                outputs=[runtime_mode_status, lesson_tab, morphology_tab],
                api_name=False,
                queue=False,
            )

    logger.debug("UI wiring complete")
    return app, api_open
