"""
Retrieval loss (the L_retrieval term of the connective joint loss, Section 4.5).

Symmetric (image->text-conditioned-query <-> target) InfoNCE with:
  * a learned temperature (owned by ForwardComposer, passed in here),
  * optional label smoothing,
  * optional in-batch hard-negative up-weighting (Section: Ablation-adjacent
    knob, `loss.hard_negative_weight` in config) -- up-weights negatives
    that are already close to the query, which is what actually separates
    "visually similar but attribute-unrelated distractors" during training.
    Currently 0.0 in every shipped config and forbidden under gradcache
    (its in-batch reweighting is not chunk-decomposable), so this path is
    dormant, not exercised by any current result -- fixed anyway (2026-09-08:
    additive log-space reweighting instead of a sign-flipping multiplicative
    rescale; 2026-09-09: i2t and t2i directions now compute independent
    hardness weights from their own matrices instead of reusing i2t's
    weighted matrix via transpose for t2i -- see symmetric_info_nce's own
    comments) so it is mathematically sound whenever it IS used.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_info_nce(query: torch.Tensor, target: torch.Tensor, logit_scale: torch.Tensor,
                        label_smoothing: float = 0.0, hard_negative_weight: float = 0.0) -> dict:
    """query, target: (B, D) L2-normalized embeddings. Returns dict with 'loss' and 'logits' (for metrics)."""
    logits = logit_scale * query @ target.t()          # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)

    def _apply_hard_weights(m: torch.Tensor) -> torch.Tensor:
        # Additive in LOG-space (fixed 2026-09-08, code-audit finding), not a
        # direct multiplicative rescale of the logits. z_j' = z_j * w_j
        # behaves unintuitively whenever z_j < 0 (a real, common case here --
        # logits are logit_scale * cosine similarity, and cosine similarity
        # is frequently negative for unrelated in-batch pairs): multiplying a
        # negative logit by w_j > 1 makes it MORE negative, i.e. moves away
        # from the positive class instead of the intended "up-weight this
        # candidate's competition for the softmax". The principled
        # reweighted-softmax form is z_j' = z_j + log(w_j) -- this is exactly
        # what a per-class prior/weight does inside a softmax denominator
        # (equivalent to `w_j * exp(z_j) = exp(z_j + log(w_j))`), and it
        # preserves the up-weighting DIRECTION regardless of z_j's sign.
        # The positive (diagonal) entry keeps weight=1 (log(1)=0, no change)
        # -- only negatives are reweighted.
        with torch.no_grad():
            pos = torch.diagonal(m).unsqueeze(1)
            # up-weight negatives whose similarity is close to the positive's similarity
            closeness = (-(pos - m).abs()).exp()
            weights = 1.0 + hard_negative_weight * closeness
            weights.fill_diagonal_(1.0)
            log_weights = weights.log()
        return m + log_weights

    if hard_negative_weight > 0:
        # Each direction computes its OWN hardness weights from its OWN
        # matrix (fixed 2026-09-09, code-audit finding #12): previously
        # `logits_weighted` (i2t's own row-wise-weighted matrix) was reused
        # for BOTH `loss_i2t` and `loss_t2i` (via `logits_weighted.t()`) --
        # transposing an ALREADY-WEIGHTED matrix is not the same as
        # computing t2i's hardness weights from `logits.t()`'s own diagonal/
        # closeness structure. `logits.t()`'s diagonal equals `logits`'s
        # diagonal (both are the true-pair similarity for row/col i), but
        # `logits.t()`'s off-diagonal "closeness to the positive" values
        # differ from `logits`'s row-wise closeness values in general (a
        # non-symmetric similarity matrix's transpose is not simply itself
        # relabeled) -- so the two directions need independently-computed
        # weights, not one direction's weights reused via transpose. Dormant
        # in every shipped config (hard_negative_weight: 0.0 everywhere) but
        # fixed so the mechanism is mathematically sound if ever enabled.
        logits_i2t = _apply_hard_weights(logits)
        logits_t2i = _apply_hard_weights(logits.t())
    else:
        logits_i2t = logits
        logits_t2i = logits.t()

    loss_i2t = F.cross_entropy(logits_i2t, labels, label_smoothing=label_smoothing)
    loss_t2i = F.cross_entropy(logits_t2i, labels, label_smoothing=label_smoothing)
    loss = 0.5 * (loss_i2t + loss_t2i)
    return {"loss": loss, "logits": logits, "labels": labels}


def attribute_bce_loss(attribute_logits: torch.Tensor, attribute_targets: torch.Tensor) -> torch.Tensor:
    """Optional weak-supervision auxiliary loss for the Attribute Edit Parser's classifier head."""
    return F.binary_cross_entropy_with_logits(attribute_logits, attribute_targets)
