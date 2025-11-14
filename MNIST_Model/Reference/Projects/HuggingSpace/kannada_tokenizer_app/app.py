from __future__ import annotations

import os

from kannada_tokenizer import KannadaBPETokenizer, build_interface


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def create_app(refresh: bool = False):
    corpus_path = os.path.join(DATA_DIR, "news_test.txt")
    merges_path = os.path.join(DATA_DIR, "merges.json")
    css_path = os.path.join(BASE_DIR, "assets", "styles.css")

    tokenizer = KannadaBPETokenizer(
        corpus_path=corpus_path,
        merges_path=merges_path,
        vocab_size=512,
        min_frequency=2,
        refresh=refresh,
    )
    return build_interface(tokenizer=tokenizer, css_path=css_path)


demo = create_app()


if __name__ == "__main__":
    demo.launch()


