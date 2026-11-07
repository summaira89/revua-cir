"""
Attribute Edit Parser (Section 4.2).

Two independent pieces, on purpose:

  1. `RuleBasedAttributeExtractor` / `LLMAttributeExtractor` /
     `HybridAttributeExtractor` -- symbolic parsing of the raw modification
     text into {attribute, old_value, new_value} triples. This runs as a
     *data-preprocessing* step (CPU, no gradients) and is pluggable: the
     paper framework's AttributeParserConfig.parser_backend selects "rule"
     (fully offline spaCy dependency-parse rules), "llm" (delegate to a
     caller-supplied function -- e.g. an OpenAI/local-LLM prompt), or
     "hybrid" (try LLM, fall back to rules on failure/timeout/missing key).
     The extracted triples are turned into a fixed-size weak-supervision
     target (hashed into `attribute_vocab_size` bins) for an auxiliary loss.

  2. `AttributeEditParserHead` -- the *learned* neural module (used inside
     ReVUACIR.forward) that reads the fused text/image token sequences and
     splits them into an editable-attribute embedding and an
     invariant-attribute embedding via an orthogonality-regularized
     two-branch projection. The invariant branch is exactly what the
     Reverse Verifier (Section 4.3) and the connective loss (Section 4.5)
     check must be preserved by the reverse edit.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from revua_cir.config import AttributeParserConfig


# --------------------------------------------------------------------------- #
# Symbolic parsing (data-preprocessing side)
# --------------------------------------------------------------------------- #
@dataclass
class AttributeTriple:
    attribute: str
    old_value: Optional[str]
    new_value: Optional[str]


_REPLACE_PATTERNS = [
    re.compile(r"(?:change|make|turn)\s+(?:it|the)?\s*(?P<attr>[\w\- ]+?)\s+(?:to|into)\s+(?P<new>[\w\- ]+)", re.I),
    re.compile(r"(?:instead of|rather than)\s+(?P<old>[\w\- ]+?)[, ]+(?:go with|use|have)?\s*(?P<new>[\w\- ]+)", re.I),
    re.compile(r"with\s+(?P<new>[\w\- ]+?)\s+instead(?:\s+of\s+(?P<old>[\w\- ]+))?", re.I),
    re.compile(r"(?:more|less)\s+(?P<attr>[\w\-]+)", re.I),
]

_INVARIANT_MARKERS = [
    re.compile(r"(?:but|while)\s+(?:keep|keeping|preserve|preserving|same)\s+(?P<attr>[\w\- ]+)", re.I),
    re.compile(r"same\s+(?P<attr>[\w\- ]+)", re.I),
]


class RuleBasedAttributeExtractor:
    """Fully offline, dependency-parse-free regex/heuristic extractor.

    Deliberately simple and dependency-light (no spaCy model download
    required at import time) so the pipeline runs anywhere; if spaCy with an
    English model is available it is used to additionally pull out
    adjectival modifiers as extra attribute candidates.
    """

    def __init__(self, use_spacy: bool = True):
        self._nlp = None
        if use_spacy:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self._nlp = spacy.blank("en")
            except ImportError:
                self._nlp = None

    def extract(self, text: str) -> List[AttributeTriple]:
        triples: List[AttributeTriple] = []
        for pat in _REPLACE_PATTERNS:
            for m in pat.finditer(text):
                gd = m.groupdict()
                attr = (gd.get("attr") or gd.get("new") or "").strip()
                if attr:
                    triples.append(AttributeTriple(attribute=attr, old_value=(gd.get("old") or None), new_value=(gd.get("new") or None)))
        invariants = []
        for pat in _INVARIANT_MARKERS:
            for m in pat.finditer(text):
                attr = (m.groupdict().get("attr") or "").strip()
                if attr:
                    invariants.append(attr)

        if self._nlp is not None and self._nlp.vocab.vectors_length >= 0:
            try:
                doc = self._nlp(text)
                for tok in doc:
                    if tok.pos_ == "ADJ" and tok.text.lower() not in {t.attribute.lower() for t in triples}:
                        triples.append(AttributeTriple(attribute=tok.text, old_value=None, new_value=tok.text))
            except Exception:
                pass

        for inv in invariants:
            triples.append(AttributeTriple(attribute=inv, old_value=inv, new_value=inv))  # old==new marks "keep"
        return triples


class LLMAttributeExtractor:
    """Delegates to a caller-supplied `fn(text) -> List[AttributeTriple]`.

    Kept as a thin adapter so a production run can plug in a real LLM
    (OpenAI/Claude/local model) prompted to return structured
    {attribute, old_value, new_value} JSON, without this module depending
    on any specific API.
    """

    def __init__(self, fn: Callable[[str], List[AttributeTriple]]):
        self.fn = fn

    def extract(self, text: str) -> List[AttributeTriple]:
        return self.fn(text)


class HybridAttributeExtractor:
    def __init__(self, llm_extractor: Optional[LLMAttributeExtractor], rule_extractor: Optional[RuleBasedAttributeExtractor] = None):
        self.llm_extractor = llm_extractor
        self.rule_extractor = rule_extractor or RuleBasedAttributeExtractor()

    def extract(self, text: str) -> List[AttributeTriple]:
        if self.llm_extractor is not None:
            try:
                triples = self.llm_extractor.extract(text)
                if triples:
                    return triples
            except Exception:
                pass
        return self.rule_extractor.extract(text)


def build_attribute_extractor(cfg: AttributeParserConfig, llm_parser_fn: Optional[Callable[[str], List[AttributeTriple]]] = None,
                              use_spacy: bool = False):
    """`use_spacy=False` by default (not RuleBasedAttributeExtractor's own
    default of True): matches the deliberate choice `data/datasets.py`
    always made before this function existed as the training-data code path
    (fixed 2026-09-08, see train.py's build_train_dataset) -- offline,
    dependency-light, "runs anywhere" without an optional spaCy model
    download. Pass use_spacy=True explicitly if you want the extra
    adjectival-modifier extraction spaCy enables."""
    if cfg.parser_backend == "rule":
        return RuleBasedAttributeExtractor(use_spacy=use_spacy)
    if cfg.parser_backend == "llm":
        if llm_parser_fn is None:
            raise ValueError("parser_backend='llm' requires an llm_parser_fn callable.")
        return LLMAttributeExtractor(llm_parser_fn)
    if cfg.parser_backend == "hybrid":
        llm = LLMAttributeExtractor(llm_parser_fn) if llm_parser_fn is not None else None
        return HybridAttributeExtractor(llm, rule_extractor=RuleBasedAttributeExtractor(use_spacy=use_spacy))
    raise ValueError(f"Unknown parser_backend: {cfg.parser_backend}")


def hash_attributes_to_vocab(triples: List[AttributeTriple], vocab_size: int, max_attributes: int) -> torch.Tensor:
    """Turns symbolic triples into a fixed-size multi-hot weak-supervision target."""
    target = torch.zeros(vocab_size)
    for t in triples[:max_attributes]:
        idx = int(hashlib.md5(t.attribute.lower().encode()).hexdigest(), 16) % vocab_size
        target[idx] = 1.0
    return target


# --------------------------------------------------------------------------- #
# Learned neural module (model side)
# --------------------------------------------------------------------------- #
class AttributeEditParserHead(nn.Module):
    """Splits the (raw edit text, raw reference image) pooled embeddings into
    editable vs invariant sub-spaces.

    Deliberately fed CLIP's raw pretrained `text_out.pooled` (clarified
    2026-09-08), not the Forward Composer's cross-attention-fused
    `text_attended` -- decomposing an already-image-mixed representation
    would risk circularity, since AEP's invariant branch is exactly what the
    Reverse Verifier later compares against a CANDIDATE image (see below);
    that comparison needs to start from the edit instruction alone, not a
    version already blended with the reference image's own content.

    Orthogonality regularizer (Section 4.2 / Ablation #8) discourages the two
    branches from encoding the same information, which is what lets the
    Reverse Verifier check "did the reverse edit preserve everything the
    query said to keep?" using only the invariant branch.
    """

    def __init__(self, embed_dim: int, cfg: AttributeParserConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        self.shared = nn.Sequential(nn.Linear(embed_dim * 2, hidden), nn.GELU())
        self.editable_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        self.invariant_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        self.attribute_classifier = nn.Linear(embed_dim, cfg.attribute_vocab_size)

    def forward(self, text_pooled: torch.Tensor, image_pooled: torch.Tensor):
        h = self.shared(torch.cat([text_pooled, image_pooled], dim=-1))
        editable = F.normalize(self.editable_head(h), dim=-1)
        invariant = F.normalize(self.invariant_head(h), dim=-1)
        attribute_logits = self.attribute_classifier(editable)
        return {"editable_embed": editable, "invariant_embed": invariant, "attribute_logits": attribute_logits}

    @staticmethod
    def orthogonality_loss(editable: torch.Tensor, invariant: torch.Tensor) -> torch.Tensor:
        """||cos(editable, invariant)||^2, averaged over the batch -- pulls the two branches apart."""
        cos = F.cosine_similarity(editable, invariant, dim=-1)
        return (cos ** 2).mean()
