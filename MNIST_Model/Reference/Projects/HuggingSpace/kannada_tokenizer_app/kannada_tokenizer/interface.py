from __future__ import annotations

import os
from typing import Any, Dict, List

import gradio as gr

from .highlighter import highlight_tokens
from .metrics import compute_compression_metrics
from .tokenizer import KannadaBPETokenizer


class TokenizerApp:
    def __init__(self, tokenizer: KannadaBPETokenizer, css_path: str | None = None):
        self.tokenizer = tokenizer
        self.custom_css = self._load_css(css_path) if css_path else None

    @staticmethod
    def _load_css(css_path: str) -> str:
        if not os.path.exists(css_path):
            return ""
        with open(css_path, "r", encoding="utf-8") as fh:
            return fh.read()

    def process(self, text: str) -> Dict[str, Any]:
        text = text or ""
        if not text.strip():
            empty_response = {
                "highlight": "<em>Enter Kannada text to view tokenization.</em>",
                "table": [],
                "metrics": {},
                "decoded": "",
            }
            return empty_response

        analysis = self.tokenizer.analyze(text)
        highlighted = highlight_tokens(analysis.token_strings, analysis.token_ids)
        metrics = compute_compression_metrics(analysis)
        rows: List[List[Any]] = []
        for token_str, token_id, byte_len in zip(
            analysis.token_strings, analysis.token_ids, analysis.byte_lengths
        ):
            display_token = token_str.replace("\n", "\\n")
            rows.append([display_token, token_id, byte_len])

        decoded = self.tokenizer.decode(analysis.token_ids)
        return {
            "highlight": highlighted,
            "table": rows,
            "metrics": metrics,
            "decoded": decoded,
        }

    def build(self) -> gr.Blocks:
        with gr.Blocks(theme=gr.themes.Soft(), css=self.custom_css) as demo:
            gr.Markdown(
                """
                # Kannada Tokenizer Viewer
                Enter Kannada text to inspect byte-pair encoding tokens, visualize
                segmentation with color-coded highlights, and view compression
                metrics compared to raw UTF-8 bytes.
                """.strip()
            )

            with gr.Row():
                text_input = gr.Textbox(
                    label="Kannada Text",
                    lines=6,
                    placeholder="ಕನ್ನಡ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ನಮೂದಿಸಿ…",
                )

            run_button = gr.Button("Tokenize", variant="primary")

            with gr.Row():
                highlight_output = gr.HTML(label="Highlighted Tokens")

            with gr.Row():
                table_output = gr.Dataframe(
                    headers=["Token", "Token ID", "Byte Length"],
                    datatype=["str", "number", "number"],
                    col_count=(3, "fixed"),
                    wrap=True,
                    label="Token Breakdown",
                )

            with gr.Row():
                metrics_output = gr.JSON(label="Compression Metrics")
                decoded_output = gr.Textbox(
                    label="Decoded text",
                    interactive=False,
                    lines=4,
                )

            def _handler(text: str) -> tuple:
                response = self.process(text)
                return (
                    response["highlight"],
                    response["table"],
                    response["metrics"],
                    response["decoded"],
                )

            run_button.click(
                fn=_handler,
                inputs=[text_input],
                outputs=[highlight_output, table_output, metrics_output, decoded_output],
            )

        return demo


def build_interface(tokenizer: KannadaBPETokenizer, css_path: str | None = None) -> gr.Blocks:
    app = TokenizerApp(tokenizer, css_path=css_path)
    return app.build()


