# Route Map

## Public App API Surface (`app.py`)

- `set_tts_only_mode(enabled)`
- `load_pronunciation_rules_json()`
- `apply_pronunciation_rules_json(raw_json)`
- `import_pronunciation_rules_json(uploaded_file)`
- `export_pronunciation_rules_json()`
- `forward_gpu(ps, ref_s, speed)`
- `generate_first(text, voice='af_heart', mix_enabled=False, voice_mix=None, speed=1, use_gpu=CUDA_AVAILABLE, pause_seconds=0.0, output_format='wav', normalize_times_enabled=None, normalize_numbers_enabled=None, style_preset='neutral')`
- `predict(text, voice='af_heart', mix_enabled=False, voice_mix=None, speed=1, normalize_times_enabled=None, normalize_numbers_enabled=None, style_preset='neutral')`
- `tokenize_first(text, voice='af_heart', mix_enabled=False, voice_mix=None, speed=1, normalize_times_enabled=None, normalize_numbers_enabled=None, style_preset='neutral')`
- `generate_all(text, voice='af_heart', mix_enabled=False, voice_mix=None, speed=1, use_gpu=CUDA_AVAILABLE, pause_seconds=0.0, normalize_times_enabled=None, normalize_numbers_enabled=None, style_preset='neutral')`
- `export_morphology_sheet(dataset='lexemes', file_format='ods')`
- `morphology_db_view(dataset='occurrences', limit=100, offset=0)`
- `launch()`

## UI Callback Surface (Tkinter)

- Total `_on_*` callbacks discovered: `50`

### `kokoro_tts/ui/features/audio_player_feature.py`
- `AudioPlayerFeature._on_audio_file_loaded`
- `AudioPlayerFeature._on_audio_player_minimal_toggle`
- `AudioPlayerFeature._on_audio_player_open`
- `AudioPlayerFeature._on_audio_player_pause`
- `AudioPlayerFeature._on_audio_player_play`
- `AudioPlayerFeature._on_audio_player_seek_back`
- `AudioPlayerFeature._on_audio_player_seek_change`
- `AudioPlayerFeature._on_audio_player_seek_forward`
- `AudioPlayerFeature._on_audio_player_seek_press`
- `AudioPlayerFeature._on_audio_player_seek_release`
- `AudioPlayerFeature._on_audio_player_stop`
- `AudioPlayerFeature._on_audio_player_tick`
- `AudioPlayerFeature._on_audio_player_volume_scale`
- `AudioPlayerFeature._on_audio_player_volume_var_updated`
- `AudioPlayerFeature._on_audio_player_waveform_seek`
- `AudioPlayerFeature._on_audio_shortcut_consume_key_release`
- `AudioPlayerFeature._on_audio_shortcut_play_pause`
- `AudioPlayerFeature._on_audio_shortcut_play_pause_from_transport_button`
- `AudioPlayerFeature._on_audio_shortcut_seek`
- `AudioPlayerFeature._on_audio_shortcut_volume`
- `AudioPlayerFeature._on_clear_history`
- `AudioPlayerFeature._on_history_context_menu`
- `AudioPlayerFeature._on_history_delete_selected`
- `AudioPlayerFeature._on_history_double_click`
- `AudioPlayerFeature._on_history_open_selected_folder`
- `AudioPlayerFeature._on_history_select_autoplay`

### `kokoro_tts/ui/features/generate_tab_feature.py`
- `_HoverTooltip._on_destroy`
- `_HoverTooltip._on_enter`
- `_HoverTooltip._on_leave`

### `kokoro_tts/ui/tkinter_app.py`
- `TkinterDesktopApp._on_close`
- `TkinterDesktopApp._on_export_morphology`
- `TkinterDesktopApp._on_generate`
- `TkinterDesktopApp._on_input_text_context_menu`
- `TkinterDesktopApp._on_input_text_redo`
- `TkinterDesktopApp._on_input_text_undo`
- `TkinterDesktopApp._on_language_change`
- `TkinterDesktopApp._on_mix_change`
- `TkinterDesktopApp._on_mix_toggle`
- `TkinterDesktopApp._on_morph_refresh`
- `TkinterDesktopApp._on_morphology_preview_dataset_change`
- `TkinterDesktopApp._on_pronunciation_apply`
- `TkinterDesktopApp._on_pronunciation_export`
- `TkinterDesktopApp._on_pronunciation_import`
- `TkinterDesktopApp._on_pronunciation_load`
- `TkinterDesktopApp._on_root_resize_sync_input_text_height`
- `TkinterDesktopApp._on_runtime_mode_change`
- `TkinterDesktopApp._on_stream_start`
- `TkinterDesktopApp._on_stream_stop`
- `TkinterDesktopApp._on_tokenize`
- `TkinterDesktopApp._on_voice_change`

## CLI Entrypoints (`scripts/*.py`)

- `scripts/benchmark_morph_ingest.py`: Script entrypoint
- `scripts/check_english_lexemes.py`: Script entrypoint
- `scripts/checkout_from_git_objects.py`: Restore a Git working tree directly from .git/objects without git executable.
- `scripts/generate_project_map.py`: Script entrypoint
- `scripts/profile_tts_inference.py`: Script entrypoint
