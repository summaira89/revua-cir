"""Collation: stacks image tensors, tokenizes edit text as a batch (much
faster than per-sample tokenization), and stacks/omits attribute targets."""
from __future__ import annotations

from typing import List

import torch


class CIRCollator:
    """Picklable collate callable.

    Deliberately a class, not a closure: Windows (and macOS `spawn`) pickle the
    collate_fn to send it to DataLoader workers, and a locally-defined function
    raises `Can't pickle local object`. With `num_workers > 0` in the config that
    made training unstartable on Windows.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[dict]) -> dict:
        reference = torch.stack([b["reference_pixel_values"] for b in batch])
        target = torch.stack([b["target_pixel_values"] for b in batch])
        texts = [b["edit_text"] for b in batch]
        tok = self.tokenizer(texts)

        attribute_targets = None
        if batch[0].get("attribute_targets") is not None:
            attribute_targets = torch.stack([b["attribute_targets"] for b in batch])

        return {
            "reference_pixel_values": reference,
            "target_pixel_values": target,
            "edit_input_ids": tok["input_ids"],
            "edit_attention_mask": tok["attention_mask"],
            "attribute_targets": attribute_targets,
            "meta": [b["meta"] for b in batch],
        }


def build_collate_fn(tokenizer):
    return CIRCollator(tokenizer)
