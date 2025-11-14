from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


Pair = Tuple[int, int]


@dataclass
class TokenAnalysis:
    token_ids: List[int]
    token_strings: List[str]
    byte_lengths: List[int]
    original_byte_length: int

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


class KannadaBPETokenizer:
    """Simple byte-pair encoding tokenizer tailored for Kannada text."""

    base_vocab_size: int = 256

    def __init__(
        self,
        corpus_path: str,
        merges_path: str,
        vocab_size: int = 512,
        min_frequency: int = 2,
        refresh: bool = False,
    ) -> None:
        self.corpus_path = corpus_path
        self.merges_path = merges_path
        self.vocab_size = max(vocab_size, self.base_vocab_size)
        self.min_frequency = max(min_frequency, 2)
        self.merges: List[Pair] = []
        self.token_bytes: Dict[int, bytes] = {
            i: bytes([i]) for i in range(self.base_vocab_size)
        }

        if refresh or not os.path.exists(self.merges_path):
            self._train()
        else:
            self._load_merges()

        self._rebuild_token_bytes()

    # --------------------------------------------------------------------- #
    # Training utilities
    # --------------------------------------------------------------------- #
    def _load_corpus(self) -> bytes:
        with open(self.corpus_path, "rb") as fh:
            return fh.read()

    def _train(self) -> None:
        corpus = self._load_corpus()
        ids = list(corpus)
        self.merges = []

        for _ in range(self.vocab_size - self.base_vocab_size):
            stats = self._get_stats(ids)
            if not stats:
                break
            pair, count = max(stats.items(), key=lambda item: item[1])
            if count < self.min_frequency:
                break
            new_token_id = self.base_vocab_size + len(self.merges)
            ids = self._merge(ids, pair, new_token_id)
            self.merges.append(pair)
            self.token_bytes[new_token_id] = (
                self.token_bytes[pair[0]] + self.token_bytes[pair[1]]
            )

        self._save_merges()

    @staticmethod
    def _get_stats(ids: Sequence[int]) -> Counter:
        pairs = zip(ids, ids[1:])
        return Counter(pairs)

    def _merge(self, ids: Sequence[int], pair: Pair, new_token_id: int) -> List[int]:
        merged: List[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def _save_merges(self) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "merges": [list(pair) for pair in self.merges],
        }
        os.makedirs(os.path.dirname(self.merges_path), exist_ok=True)
        with open(self.merges_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _load_merges(self) -> None:
        with open(self.merges_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        stored_vocab = payload.get("vocab_size", self.vocab_size)
        if stored_vocab < self.vocab_size:
            # Retrain if stored vocab is insufficient for requested size
            self._train()
            return

        self.merges = [tuple(pair) for pair in payload.get("merges", [])]  # type: ignore[list-item]

    def _rebuild_token_bytes(self) -> None:
        # reset to base bytes to avoid duplicates
        self.token_bytes = {i: bytes([i]) for i in range(self.base_vocab_size)}
        for offset, pair in enumerate(self.merges):
            token_id = self.base_vocab_size + offset
            left, right = pair
            self.token_bytes[token_id] = (
                self.token_bytes[left] + self.token_bytes[right]
            )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def encode(self, text: str) -> List[int]:
        byte_ids = list(text.encode("utf-8"))
        return self._apply_merges(byte_ids)

    def decode(self, token_ids: Iterable[int]) -> str:
        buffer = bytearray()
        for token_id in token_ids:
            buffer.extend(self.token_bytes[token_id])
        return buffer.decode("utf-8", errors="replace")

    def analyze(self, text: str) -> TokenAnalysis:
        token_ids = self.encode(text)
        token_strings = [
            self.token_bytes[token_id].decode("utf-8", errors="replace")
            for token_id in token_ids
        ]
        byte_lengths = [len(self.token_bytes[token_id]) for token_id in token_ids]
        original_byte_length = len(text.encode("utf-8"))
        return TokenAnalysis(
            token_ids=token_ids,
            token_strings=token_strings,
            byte_lengths=byte_lengths,
            original_byte_length=original_byte_length,
        )

    # ------------------------------------------------------------------ #
    # Helper routines
    # ------------------------------------------------------------------ #
    def _apply_merges(self, ids: Sequence[int]) -> List[int]:
        token_ids = list(ids)
        for offset, pair in enumerate(self.merges):
            new_token_id = self.base_vocab_size + offset
            token_ids = self._merge(token_ids, pair, new_token_id)
        return token_ids

    def available_vocab_size(self) -> int:
        return self.base_vocab_size + len(self.merges)


