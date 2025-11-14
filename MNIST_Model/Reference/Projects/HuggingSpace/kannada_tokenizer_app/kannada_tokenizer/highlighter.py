from __future__ import annotations

import hashlib
import html
from typing import Iterable, Sequence


def _id_to_color(token_id: int) -> str:
    """Map a token id to a pastel background color."""
    digest = hashlib.md5(str(token_id).encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16)
    saturation = 65
    lightness = 80
    return f"hsl({hue * 3}, {saturation}%, {lightness}%)"


def highlight_tokens(
    token_strings: Sequence[str],
    token_ids: Sequence[int],
    *,
    css_class: str = "token",
) -> str:
    """Return HTML markup that highlights each token with a deterministic color."""
    spans = []
    for token_str, token_id in zip(token_strings, token_ids):
        color = _id_to_color(token_id)
        safe_token = html.escape(token_str) or "&nbsp;"
        display_token = safe_token.replace(" ", "&nbsp;")
        spans.append(
            f'<span class="{css_class}" style="background:{color}" '
            f'data-token-id="{token_id}">{display_token}</span>'
        )
    return "".join(spans)


