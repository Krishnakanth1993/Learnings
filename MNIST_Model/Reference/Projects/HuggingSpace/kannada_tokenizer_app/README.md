# Kannada Tokenizer Gradio App

A lightweight Gradio application that visualises byte-pair encoding tokenisation for Kannada text, inspired by the OpenAI *tiktoken* viewer. Enter Kannada phrases to inspect colour-highlighted tokens, view token IDs, and understand compression achieved versus the raw UTF-8 byte representation.

## Features

- Byte-level BPE tokenizer fine-tuned on a Kannada news corpus
- Deterministic colour highlighting per token
- Token table with IDs and byte lengths
- Compression metrics (token count, compression ratio, average bytes per token)
- Modern Gradio Blocks UI packaged for Hugging Face Spaces deployment

## Project Structure

```
kannada_tokenizer_app/
├── app.py                # Entry point used by Gradio / Hugging Face Spaces
├── requirements.txt      # Runtime dependencies
├── README.md
├── assets/
│   └── styles.css        # Custom styling for the Blocks interface
├── data/
│   ├── news_test.txt     # Kannada corpus used to build BPE merges
│   └── merges.json       # Generated automatically on first run
└── kannada_tokenizer/
    ├── __init__.py
    ├── tokenizer.py      # `KannadaBPETokenizer` (train/load/encode/decode/analyse)
    ├── highlighter.py    # Token → colour HTML utilities
    ├── metrics.py        # Compression statistics helpers
    └── interface.py      # Gradio Blocks interface assembly
```

## Getting Started Locally

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python app.py
```

The first run trains the tokenizer on `data/news_test.txt` and stores merges in `data/merges.json`. Subsequent launches reuse the cached merges for faster startup.

## Deployment to Hugging Face Spaces

1. Create a new **Gradio** Space.
2. Commit all files in this directory to the Space repository (or connect to this Git project).
3. Ensure `app.py` is selected as the entry file. Hugging Face will automatically install dependencies from `requirements.txt` and start the app.

## Updating the Tokenizer

If you modify the training corpus or want a different vocabulary size, update `app.py` to pass `refresh=True` to `create_app()` once. This retrains the merges and rewrites `data/merges.json`.

```python
demo = create_app(refresh=True)
```

Switch back to `refresh=False` (or remove the flag) afterwards to avoid retraining on every run.

## License

MIT License. Feel free to adapt for your own Kannada NLP experiments.


