"""Tokenizer abstraction so the data pipeline works with both the real CLIP
tokenizer and the offline `tiny` backbone (Section: backbone.py)."""
from __future__ import annotations

import hashlib
import re
from typing import List

import torch


class SimpleWhitespaceTokenizer:
    """Offline fallback tokenizer -- deterministic hashing vocab, no external files.

    Only used when backbone.kind == 'tiny'. For real experiments always use
    `HFTokenizerWrapper` (the CLIP/BPE tokenizer that matches the chosen
    pretrained backbone).
    """

    def __init__(self, vocab_size: int = 30522, max_len: int = 32):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_id = 0
        self.cls_id = 1

    def _hash(self, token: str) -> int:
        # md5, not Python's builtin hash() (fixed 2026-09-08, code-audit
        # finding): CPython randomizes string hash seeds by default
        # (PYTHONHASHSEED), so builtin hash() is NOT stable across process
        # restarts despite this class's own docstring claiming "deterministic
        # hashing vocab" -- two separate test runs could tokenize identical
        # text to different input_ids. md5 (already this project's pattern
        # for the same purpose, see attribute_parser.hash_attributes_to_vocab)
        # is stable across processes/machines. Only affects the offline
        # `tiny` smoke-test backbone, never real CLIP training.
        return (int(hashlib.md5(token.encode()).hexdigest(), 16) % (self.vocab_size - 2)) + 2

    def __call__(self, texts: List[str]):
        input_ids = torch.zeros(len(texts), self.max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), self.max_len, dtype=torch.long)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z']+", text.lower())[: self.max_len - 1]
            ids = [self.cls_id] + [self._hash(t) for t in tokens]
            ids = ids[: self.max_len]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class HFTokenizerWrapper:
    def __init__(self, pretrained_name: str = "openai/clip-vit-large-patch14", max_len: int = 32):
        from transformers import CLIPTokenizerFast

        self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_name)
        self.max_len = max_len

    def __call__(self, texts: List[str]):
        enc = self.tokenizer(texts, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def build_tokenizer(backbone_cfg):
    if backbone_cfg.kind == "tiny":
        return SimpleWhitespaceTokenizer(backbone_cfg.tiny_vocab_size)
    return HFTokenizerWrapper(backbone_cfg.pretrained_name)
