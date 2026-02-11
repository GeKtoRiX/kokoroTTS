"""Gradio UI construction for the Kokoro TTS app."""
from __future__ import annotations

import os

import gradio as gr

from ..config import AppConfig
from ..constants import OUTPUT_FORMATS

UI_PRIMARY_HUE = os.getenv("UI_PRIMARY_HUE", "green").strip() or "green"
APP_THEME = gr.themes.Base(primary_hue=UI_PRIMARY_HUE)

TOKEN_NOTE = (
    "\n\U0001f4a1 Customize pronunciation with Markdown link syntax and /slashes/ like "
    "`[Kokoro](/k\u02c8Ok\u0259\u0279O/)`\n\n"
    "\U0001f4ac To adjust intonation, try punctuation `;:,.!?\u2014\u2026\"()\u201c\u201d` "
    "or stress `\u02c8` and `\u02cc`\n\n"
    "\u2b07\ufe0f Lower stress `[1 level](-1)` or `[2 levels](-2)`\n\n"
    "\u2b06\ufe0f Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)\n"
)

DIALOGUE_NOTE = (
    "\nUse [voice=af_heart] to switch speakers inside the text.\n"
    "Mix voices with commas: [voice=af_heart,am_michael].\n"
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
    history_service,
    choices,
) -> tuple[gr.Blocks, bool]:
    default_output_format = (
        config.default_output_format
        if config.default_output_format in OUTPUT_FORMATS
        else "wav"
    )
    history_state = None
    history_audios: list[gr.Audio] = []
    clear_history_btn = None

    with gr.Blocks(theme=APP_THEME) as generate_tab:
        out_audio = gr.Audio(label="Output Audio", interactive=False, streaming=False, autoplay=True)
        generate_btn = gr.Button("Generate", variant="primary")
        history_state = gr.State([])
        if config.history_limit > 0:
            with gr.Accordion("History", open=False):
                clear_history_btn = gr.Button("Clear history", variant="secondary")
                for index in range(1, config.history_limit + 1):
                    history_audios.append(
                        gr.Audio(label=f"History {index}", interactive=False, streaming=False)
                    )
        with gr.Accordion("Output Tokens", open=True):
            out_ps = gr.Textbox(
                interactive=False,
                show_label=False,
                info="Tokens used to generate the audio, up to 510 context length.",
            )
            tokenize_btn = gr.Button("Tokenize", variant="secondary")
            gr.Markdown(TOKEN_NOTE)
            predict_btn = gr.Button("Predict", variant="secondary", visible=False)

    stream_note = [
        "\u26a0\ufe0f There is an unknown Gradio bug that might yield no audio the first time you click `Stream`."
    ]
    if config.char_limit is not None:
        stream_note.append(f"\u2702\ufe0f Each stream is capped at {config.char_limit} characters.")
        stream_note.append(
            "\U0001f680 Want more characters? You can "
            "[use Kokoro directly](https://huggingface.co/hexgrad/Kokoro-82M#usage) "
            "or duplicate this space:"
        )
    stream_note = "\n\n".join(stream_note)

    def toggle_mix_controls(enabled):
        return gr.update(visible=enabled), gr.update(interactive=not enabled)

    def update_history(history):
        updated = history_service.update_history(history)
        values = updated + [None] * (config.history_limit - len(updated))
        return (updated, *values)

    def clear_history(history):
        updated = history_service.clear_history(history)
        values = [None] * config.history_limit
        return (updated, *values)

    with gr.Blocks(theme=APP_THEME) as stream_tab:
        out_stream = gr.Audio(label="Output Audio Stream", interactive=False, streaming=True, autoplay=True)
        with gr.Row():
            stream_btn = gr.Button("Stream", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
        with gr.Accordion("Note", open=True):
            gr.Markdown(stream_note)
            gr.DuplicateButton()

    api_open = config.space_id != "hexgrad/Kokoro-TTS"
    api_name = None if api_open else False
    logger.debug("API_OPEN=%s", api_open)
    with gr.Blocks(theme=APP_THEME) as app:
        with gr.Row():
            with gr.Column():
                stream_cap = "\u221e" if config.char_limit is None else config.char_limit
                text = gr.Textbox(
                    label="Input Text",
                    info=(
                        f"Up to ~{config.max_chunk_chars} characters per chunk for Generate, "
                        f"or {stream_cap} characters per Stream. Use | to split into "
                        "separate files. Use [voice=af_heart] to switch speakers."
                    ),
                )
                with gr.Row():
                    voice = gr.Dropdown(
                        list(choices.items()),
                        value="af_heart",
                        label="Voice",
                        info="Quality and availability vary by language",
                    )
                    mix_enabled = gr.Checkbox(label="Mix voices", value=False)
                voice_mix = gr.Dropdown(
                    list(choices.items()),
                    value=[],
                    multiselect=True,
                    label="Voice mix",
                    info="Select multiple voices to average",
                    visible=False,
                )
                with gr.Row():
                    use_gpu = gr.Dropdown(
                        [("ZeroGPU \U0001f680", True), ("CPU \U0001f40c", False)],
                        value=cuda_available,
                        label="Hardware",
                        info="GPU is usually faster, but has a usage quota",
                        interactive=cuda_available,
                    )
                speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="Speed")
                pause_between = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0,
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
                with gr.Accordion("Dialog tags", open=False):
                    gr.Markdown(DIALOGUE_NOTE)
            with gr.Column():
                gr.TabbedInterface([generate_tab, stream_tab], ["Generate", "Stream"])
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
            ],
            outputs=[out_audio, out_ps],
            api_name=api_name,
        )
        generate_event.then(
            fn=update_history,
            inputs=[history_state],
            outputs=[history_state] + history_audios,
        )
        if clear_history_btn is not None:
            clear_history_btn.click(
                fn=clear_history,
                inputs=[history_state],
                outputs=[history_state] + history_audios,
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
            ],
            outputs=[out_ps],
            api_name=api_name,
        )
        stream_event = stream_btn.click(
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
            ],
            outputs=[out_stream],
            api_name=api_name,
        )
        stop_btn.click(fn=None, cancels=stream_event)
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
            ],
            outputs=[out_audio],
            api_name=api_name,
        )

    logger.debug("UI wiring complete")
    return app, api_open
