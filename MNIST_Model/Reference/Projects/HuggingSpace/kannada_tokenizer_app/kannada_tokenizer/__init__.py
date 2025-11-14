"""Application package for the Kannada tokenizer Gradio app."""

from .tokenizer import KannadaBPETokenizer


def build_interface(*args, **kwargs):
    from .interface import build_interface as _build_interface

    return _build_interface(*args, **kwargs)


__all__ = ["KannadaBPETokenizer", "build_interface"]


