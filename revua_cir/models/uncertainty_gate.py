"""
Uncertainty Gate (Section 4.4).

Combines three signals into a single calibrated confidence score used to
rerank the top-K candidates coming out of the Forward Composer / Reverse
Verifier:

  1. Semantic-ambiguity variance -- a Gaussian-embedding-style variance head
     (COPE/HUG-inspired) trained with a Neighborhood-Deviation-style
     regularizer: predicted variance should track how dispersed a query's
     nearest neighbors are in embedding space (approximated in-batch, since
     a full-corpus k-NN pass every step is not tractable for training).
  2. Multimodal coordination quality -- a small MLP scoring how well the
     reference image and edit text agree/complement each other (HUG-style
     "coordination" signal, decomposed from the raw query representation).
  3. Reverse-verification disagreement -- fed in directly from the Reverse
     Verifier (Section 4.5's connective mechanism); this is the input no
     prior uncertainty-CIR paper ingests, per the novelty scan.

All three are fused by a small learned logistic layer into a final
confidence in [0, 1], with a separate temperature parameter for calibration
(Ablation #9, Expected Calibration Error).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from revua_cir.config import UncertaintyGateConfig


class GaussianVarianceHead(nn.Module):
    """Predicts a scalar log-variance (total uncertainty) for an embedding."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)  # log-variance, (B,)


class CoordinationQualityHead(nn.Module):
    """Scores how well image and text modalities agree (HUG-style coordination signal)."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, image_pooled: torch.Tensor, text_pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([image_pooled, text_pooled], dim=-1)).squeeze(-1))


class UncertaintyGate(nn.Module):
    def __init__(self, embed_dim: int, cfg: UncertaintyGateConfig):
        super().__init__()
        self.cfg = cfg
        self.variance_head = GaussianVarianceHead(embed_dim, cfg.variance_hidden_dim)
        self.coordination_head = CoordinationQualityHead(embed_dim, cfg.coordination_hidden_dim)
        n_signals = 2 + (1 if cfg.fuse_reverse_disagreement else 0)
        self.fusion = nn.Linear(n_signals, 1)
        # Positive parameterization (fixed 2026-09-09, code-audit finding):
        # previously a raw `nn.Parameter` clamped at use-site
        # (`self.temperature.clamp(min=1e-3)`). A clamped parameter that
        # gets pushed below the clamp floor by gradient descent sits in a
        # zero-gradient region (clamp's gradient is 0 there) and can never
        # recover on its own. Storing log(temperature) and exponentiating at
        # use-site keeps the *effective* temperature always positive with no
        # boundary to get stuck against -- gradient flows through `.exp()`
        # everywhere. `math.log` (not `torch.log`) since this only runs once
        # at construction on a Python float, not a tensor.
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(cfg.calibration_temperature_init))))

    def forward(self, query_pooled: torch.Tensor, image_pooled: torch.Tensor, text_pooled: torch.Tensor,
                reverse_disagreement: Optional[torch.Tensor] = None):
        log_var = self.variance_head(query_pooled)                      # (B,)
        variance = log_var.exp()
        norm_uncertainty = torch.sigmoid(log_var)                        # squashed to [0,1] for fusion
        coordination = self.coordination_head(image_pooled, text_pooled)  # (B,) higher = better coordinated

        signals = [norm_uncertainty, 1.0 - coordination]  # both "higher = more uncertain"
        if self.cfg.fuse_reverse_disagreement:
            if reverse_disagreement is None:
                reverse_disagreement = torch.zeros_like(norm_uncertainty)
            signals.append(reverse_disagreement)
        signal_tensor = torch.stack(signals, dim=-1)  # (B, n_signals)

        uncertainty_logit = self.fusion(signal_tensor).squeeze(-1) / self.log_temperature.exp()
        confidence_logit = -uncertainty_logit
        confidence = torch.sigmoid(confidence_logit)  # confidence = 1 - calibrated uncertainty prob
        return {
            "log_var": log_var,
            "variance": variance,
            "coordination": coordination,
            "confidence": confidence,
            # Pre-sigmoid logit kept so calibration_loss can use the
            # autocast-safe BCE-with-logits form (plain BCE errors under AMP).
            "confidence_logit": confidence_logit,
        }

    @staticmethod
    def neighborhood_deviation_loss(query_pooled: torch.Tensor, log_var: torch.Tensor, k: int,
                                     full_batch_query_pooled: Optional[torch.Tensor] = None,
                                     full_batch_log_var: Optional[torch.Tensor] = None,
                                     chunk_offset: int = 0) -> torch.Tensor:
        """Regresses predicted variance toward in-batch k-NN embedding dispersion (COPE-style).

        `full_batch_query_pooled`/`full_batch_log_var` (detached, no_grad),
        when given, are the FULL logical batch's embeddings/variance
        predictions -- gradcache's per-chunk aux-loss recompute calls this on
        a tiny chunk (e.g. size 2), and both the k-NN neighborhood (line
        "dist = cdist(...)") and the mean/std normalization below are
        genuinely batch-wide statistics: computed over 1-2 chunk-mates
        instead of the real batch, they silently regress `log_var` toward
        near-meaningless per-chunk noise (measured up to ~195x gradient
        error on `variance_head`, 2026-09, via
        tests/test_smoke.py::test_gradcache_trainable_backbone_matches_plain_step
        extended to fc+aep+rv+ug). `chunk_offset` locates this call's rows
        within the full-batch tensors. `pred` -- the only gradient-carrying
        term -- still comes from this chunk's own `log_var`; only the
        target/normalization statistics switch to the full-batch view.
        """
        knn_source = full_batch_query_pooled if full_batch_query_pooled is not None else query_pooled
        b = knn_source.size(0)
        k = min(k, max(b - 1, 1))
        with torch.no_grad():
            dist = torch.cdist(knn_source, knn_source, p=2)               # (full_B, full_B) or (B, B)
            dist.fill_diagonal_(float("inf"))
            knn_dist, _ = dist.topk(k, dim=-1, largest=False)
            target_dispersion_full = knn_dist.mean(dim=-1)                 # (full_B,) or (B,)
            target_dispersion_full = (target_dispersion_full - target_dispersion_full.mean()) / (
                target_dispersion_full.std() + 1e-6)
            if full_batch_query_pooled is not None:
                n = query_pooled.size(0)
                target_dispersion = target_dispersion_full[chunk_offset:chunk_offset + n]
            else:
                target_dispersion = target_dispersion_full

        norm_source = full_batch_log_var if full_batch_log_var is not None else log_var
        norm_mean, norm_std = norm_source.mean(), norm_source.std()
        pred = (log_var - norm_mean) / (norm_std + 1e-6)
        return F.mse_loss(pred, target_dispersion.detach())

    @staticmethod
    def calibration_loss(confidence_logit: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy between predicted confidence and whether top-1
        retrieval was correct.

        Takes the *pre-sigmoid logit*, not the probability: `F.binary_cross_entropy`
        raises under `torch.autocast` (and `optim.amp: true` is the default), while
        the fused with-logits form is autocast-safe and numerically stabler.
        """
        return F.binary_cross_entropy_with_logits(confidence_logit, correct.float())
