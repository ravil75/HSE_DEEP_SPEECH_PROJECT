# src/utils/tokenizer.py
from typing import List


class CharTokenizer:
    def __init__(self, chars: str = " abcdefghijklmnopqrstuvwxyz'.,?"):
        self.blank = 0
        self.chars = chars
        # idx->char: 1..N
        self.idx2char = {i + 1: ch for i, ch in enumerate(self.chars)}
        self.char2idx = {ch: i + 1 for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.idx2char) + 1  # +1 for blank

    def encode(self, text: str) -> List[int]:
        if text is None:
            return []
        text = text.lower()
        out = []
        for ch in text:
            if ch in self.char2idx:
                out.append(self.char2idx[ch])
        return out

    def decode(self, indices: List[int]) -> str:
        out = []
        prev = None
        for i in indices:
            if i == prev:
                prev = i
                continue
            if i != self.blank and i in self.idx2char:
                out.append(self.idx2char[i])
            prev = i
        return "".join(out)
