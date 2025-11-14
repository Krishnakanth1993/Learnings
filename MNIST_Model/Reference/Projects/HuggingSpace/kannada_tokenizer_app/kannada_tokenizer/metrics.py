from __future__ import annotations

from typing import Dict

from .tokenizer import TokenAnalysis


def compute_compression_metrics(analysis: TokenAnalysis) -> Dict[str, float]:
    original = max(analysis.original_byte_length, 1)
    token_count = max(analysis.token_count, 1)
    compression_ratio = original / token_count
    compression_percent = (1 - (token_count / original)) * 100
    avg_bytes_per_token = analysis.original_byte_length / token_count
    return {
        "original_bytes": float(analysis.original_byte_length),
        "token_count": float(analysis.token_count),
        "compression_ratio": round(compression_ratio, 3),
        "compression_percent": round(compression_percent, 2),
        "avg_bytes_per_token": round(avg_bytes_per_token, 3),
    }


